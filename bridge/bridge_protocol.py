"""
bridge_protocol.py -- Python subprocess wrapper for the BHuman C++ bridge.

Protocol: JSON-line over stdin/stdout.
  - Python sends one JSON line per frame (GroundTruthWorldState)
  - C++ bridge responds with one JSON line (decisions)
  - stderr is used for logging/errors

Frame input format (Python → C++):
{
  "type": "frame",
  "frame_id": 123,
  "timestamp_ms": 4059,
  "ball": {"x": 100.0, "y": 200.0, "vx": 0.0, "vy": 0.0},
  "own_team": [
    {"number": 1, "x": -2000, "y": 0, "theta": 0.0, "upright": true},
    ...
  ],
  "opp_team": [
    {"number": 1, "x": 2000, "y": 0, "theta": 3.14, "upright": true},
    ...
  ],
  "game_state": "playing",
  "own_score": 0,
  "opp_score": 0
}

Frame output format (C++ → Python):
{
  "type": "decision",
  "frame_id": 123,
  "player_number": 1,
  "skill_request": "shoot",
  "motion_request": "walkToBallAndKick",
  "motion_speed": [0.0, 0.0, 0.0],
  "kick_type": "forwardFastLeftLong",
  "kick_length_mm": 5000.0,
  "last_kick_type": "forwardFastLeftLong",
  "last_kick_timestamp_ms": 3990,
  "strategy_status": "s4v4_scaled",
  "current_tactic": "t121_4v4_scaled",
  "behavior_status": "playing",
  "activation_graph": [
    {"option": "PlayBall", "depth": 0, "state": "execute",
     "option_time": 100, "state_time": 50, "parameters": []}
  ],
  "active_agent_number": 2,
  "decision_time_us": 450
}

Control messages:
  {"type": "init", "scenario": "2D", "location": "Default",
   "team_number": 1, "player_number": 1, "num_players": 4}
  {"type": "reset"}
  {"type": "shutdown"}

Responses:
  {"type": "ready", "version": "1.0"}
  {"type": "ack", "command": "reset"}
  {"type": "error", "message": "..."}
"""

from __future__ import annotations

import json
import logging
import os
import queue
import subprocess
import tempfile
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from runtime_paths import (
    preferred_bridge_transport,
    resolve_bhuman_root,
    resolve_bridge_executable,
    stable_bridge_transports,
)

logger = logging.getLogger(__name__)

ONESHOT_ARTIFACT_PREFIX = "rlbridge_oneshot_"


def _oneshot_runtime_dir() -> Path:
    """
    Keep one-shot transport artifacts out of the git workspace.

    These files are purely operational scratch space for the Windows fallback
    transport. Writing them inside `RL/bridge` polluted `git status` whenever
    cleanup failed on Windows due to transient file-handle races.
    """
    runtime_dir = Path(tempfile.gettempdir()) / "rlbridge_oneshot_runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    return runtime_dir


def _extract_json_objects_from_mixed_output(text: str) -> List[Dict[str, Any]]:
    """
    Recover JSON objects from bridge output even when stdout/stderr logs are
    interleaved on the same line as the JSON payload.

    The RLBridge Windows one-shot transport merges stdout and stderr into a
    single file. BHuman logs can appear immediately before or after a JSON
    object, which makes line-based parsing unreliable. This scanner walks the
    whole text, extracts balanced `{...}` spans while respecting quoted strings
    and escapes, and returns every valid JSON object it can decode.
    """
    objects: List[Dict[str, Any]] = []
    length = len(text)
    index = 0

    while index < length:
        if text[index] != "{":
            index += 1
            continue

        start = index
        depth = 0
        in_string = False
        escape = False

        while index < length:
            ch = text[index]

            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == "\"":
                    in_string = False
            else:
                if ch == "\"":
                    in_string = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = text[start:index + 1]
                        try:
                            obj = json.loads(candidate)
                        except json.JSONDecodeError:
                            break
                        if isinstance(obj, dict):
                            objects.append(obj)
                        break

            index += 1

        index = start + 1 if depth != 0 else index + 1

    return objects

def _resolve_default_bridge_path() -> Path:
    """Resolve the preferred RLBridge executable for the current workspace."""
    return resolve_bridge_executable()


def expected_transport_mode() -> str:
    """Preferred transport mode for the current platform/configuration."""
    return preferred_bridge_transport()


def is_stable_transport_mode(mode: str) -> bool:
    """Return True when the mode is considered valid for the current platform."""
    return mode in stable_bridge_transports()


# Default path to the compiled bridge executable
_DEFAULT_BRIDGE_PATH = _resolve_default_bridge_path()

# Timeout for bridge startup (seconds)
_STARTUP_TIMEOUT = 10.0

# Timeout per frame decision (seconds)
_FRAME_TIMEOUT = 1.0


# ---------------------------------------------------------------------------
# EXCEPTIONS
# ---------------------------------------------------------------------------

class BridgeNotFoundError(Exception):
    """Raised when the compiled bridge executable is not found."""
    pass


class BridgeStartupError(Exception):
    """Raised when the bridge fails to start or handshake."""
    pass


class BridgeTimeoutError(Exception):
    """Raised when the bridge doesn't respond within timeout."""
    pass


class BridgeProtocolError(Exception):
    """Raised on unexpected bridge response format."""
    pass


# ---------------------------------------------------------------------------
# BRIDGE CONFIGURATION
# ---------------------------------------------------------------------------

@dataclass
class BridgeConfig:
    """Configuration for the BHuman C++ bridge subprocess."""
    bridge_executable: Path = field(default_factory=resolve_bridge_executable)
    bhuman_root: Path = field(default_factory=resolve_bhuman_root)
    scenario: str = "2D"
    location: str = "Default"
    team_number: int = 1
    player_number: int = 1
    num_players: int = 4
    startup_timeout: float = _STARTUP_TIMEOUT
    frame_timeout: float = _FRAME_TIMEOUT
    preferred_transport: str = field(default_factory=preferred_bridge_transport)
    allow_oneshot_fallback: bool = True


# ---------------------------------------------------------------------------
# FRAME BUILDER — converts GroundTruthFrame to JSON
# ---------------------------------------------------------------------------

def frame_to_json(
    frame_id: int,
    ball_pos: np.ndarray,
    ball_vel: np.ndarray,
    own_team: List[Dict[str, Any]],
    opp_team: List[Dict[str, Any]],
    game_state: str = "playing",
    own_score: int = 0,
    opp_score: int = 0,
    timestamp_ms: int = 0,
) -> str:
    """
    Build a JSON-line frame message for the C++ bridge.

    own_team: [{"number": int, "x": float, "y": float, "theta": float, "upright": bool}, ...]
    opp_team: same format
    """
    msg = {
        "type": "frame",
        "frame_id": frame_id,
        "timestamp_ms": timestamp_ms,
        "ball": {
            "x": float(ball_pos[0]),
            "y": float(ball_pos[1]),
            "vx": float(ball_vel[0]),
            "vy": float(ball_vel[1]),
        },
        "own_team": own_team,
        "opp_team": opp_team,
        "game_state": game_state,
        "own_score": own_score,
        "opp_score": opp_score,
    }
    return json.dumps(msg, separators=(",", ":"))


# ---------------------------------------------------------------------------
# BRIDGE SUBPROCESS MANAGER
# ---------------------------------------------------------------------------

class BHumanBridge:
    """
    Manages the BHuman C++ bridge subprocess.

    Lifecycle:
        bridge = BHumanBridge(config)
        bridge.start()          # launches subprocess, waits for handshake
        result = bridge.step(frame_data)  # sends frame, receives decision
        bridge.reset()          # resets BHuman state for new episode
        bridge.shutdown()       # graceful shutdown
    """

    def __init__(self, config: Optional[BridgeConfig] = None):
        self._config = config or BridgeConfig()
        self._process: Optional[subprocess.Popen] = None
        self._frame_counter = 0
        self._started = False
        self._transport_mode = self._config.preferred_transport
        self._env: Optional[Dict[str, str]] = None
        self._stdout_queue: "queue.Queue[Optional[bytes]]" = queue.Queue()
        self._stderr_lines = deque(maxlen=400)
        self._stderr_lock = threading.Lock()
        self._reader_threads: List[threading.Thread] = []
        self._reader_stop = threading.Event()
        self._reconnect_count = 0
        self._last_reset_ms = 0.0

    @property
    def is_running(self) -> bool:
        if not self._started:
            return False
        if self._transport_mode == "oneshot_file_fallback":
            return True
        return (
            self._process is not None
            and self._process.poll() is None
        )

    def start(self) -> None:
        """Launch the bridge subprocess and perform handshake."""
        exe = self._config.bridge_executable
        if not exe.exists():
            raise BridgeNotFoundError(
                f"Bridge executable not found at {exe}. "
                f"Build it with: cd bridge && cmake --build . --config Release"
            )

        env = os.environ.copy()
        self._env = env
        self._frame_counter = 0

        if self._config.preferred_transport == "oneshot_file_fallback":
            logger.info("Starting RLBridge directly in one-shot fallback mode: %s", exe)
            self._started = True
            self._transport_mode = "oneshot_file_fallback"
            return

        logger.info("Starting BHuman bridge with persistent pipe transport: %s", exe)
        startup_kwargs: Dict[str, Any] = {
            "args": [str(exe)],
            "stdin": subprocess.PIPE,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "env": env,
            "bufsize": 0,
        }
        if os.name == "nt" and hasattr(subprocess, "CREATE_NO_WINDOW"):
            startup_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

        self._process = subprocess.Popen(**startup_kwargs)
        self._start_reader_threads()

        # Send init command
        init_msg = {
            "type": "init",
            "scenario": self._config.scenario,
            "location": self._config.location,
            "team_number": self._config.team_number,
            "player_number": self._config.player_number,
            "num_players": self._config.num_players,
        }
        self._send(init_msg)

        # Wait for ready response
        response = self._recv(timeout=self._config.startup_timeout)
        if response is None:
            poll_code = self._process.poll() if self._process is not None else None
            stderr_out = self._read_stderr()
            if self._config.allow_oneshot_fallback:
                logger.warning(
                    "Bridge pipe handshake failed (poll=%s). "
                    "Falling back to one-shot file transport. stderr=%s",
                    poll_code, stderr_out,
                )
                self._shutdown_process_only()
                self._frame_counter = 0
                self._started = True
                self._transport_mode = "oneshot_file_fallback"
                return
            raise BridgeStartupError(
                f"Bridge did not respond to init. stderr: {stderr_out}"
            )

        if response.get("type") != "ready":
            if self._config.allow_oneshot_fallback:
                logger.warning(
                    "Bridge pipe handshake returned unexpected response. "
                    "Falling back to one-shot file transport. response=%s",
                    response,
                )
                self._shutdown_process_only()
                self._frame_counter = 0
                self._started = True
                self._transport_mode = "oneshot_file_fallback"
                return
            raise BridgeStartupError(
                f"Unexpected init response: {response}"
            )

        self._frame_counter = 0
        self._started = True
        self._transport_mode = "pipe_process"
        logger.info("BHuman bridge ready (version=%s)", response.get("version"))

    def step(
        self,
        ball_pos: np.ndarray,
        ball_vel: np.ndarray,
        own_team: List[Dict[str, Any]],
        opp_team: List[Dict[str, Any]],
        game_state: str = "playing",
        own_score: int = 0,
        opp_score: int = 0,
        timestamp_ms: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Send one frame to the bridge, receive decision.

        Returns the raw decision dict from the C++ bridge.
        """
        if not self.is_running:
            raise BridgeStartupError("Bridge is not running. Call start() first.")

        if self._transport_mode == "oneshot_file_fallback":
            return self._step_oneshot_file_transport(
                ball_pos=ball_pos,
                ball_vel=ball_vel,
                own_team=own_team,
                opp_team=opp_team,
                game_state=game_state,
                own_score=own_score,
                opp_score=opp_score,
                timestamp_ms=timestamp_ms,
            )

        self._frame_counter += 1
        if timestamp_ms is None:
            timestamp_ms = self._frame_counter * 33  # ~30Hz

        frame_json = frame_to_json(
            frame_id=self._frame_counter,
            ball_pos=ball_pos,
            ball_vel=ball_vel,
            own_team=own_team,
            opp_team=opp_team,
            game_state=game_state,
            own_score=own_score,
            opp_score=opp_score,
            timestamp_ms=timestamp_ms,
        )

        try:
            self._send_raw(frame_json)
            response = self._recv(timeout=self._config.frame_timeout)
        except BridgeStartupError:
            if self._config.allow_oneshot_fallback:
                logger.warning(
                    "Bridge pipe transport write failed. "
                    "Switching to one-shot file transport."
                )
                self._switch_to_oneshot_fallback()
                self._frame_counter -= 1
                return self._step_oneshot_file_transport(
                    ball_pos=ball_pos,
                    ball_vel=ball_vel,
                    own_team=own_team,
                    opp_team=opp_team,
                    game_state=game_state,
                    own_score=own_score,
                    opp_score=opp_score,
                    timestamp_ms=timestamp_ms,
                )
            raise

        if response is None:
            stderr_out = self._read_stderr()
            if self._config.allow_oneshot_fallback:
                logger.warning(
                    "Bridge pipe transport timed out on frame %s. "
                    "Switching to one-shot file transport. stderr=%s",
                    self._frame_counter, stderr_out,
                )
                self._switch_to_oneshot_fallback()
                self._frame_counter -= 1
                return self._step_oneshot_file_transport(
                    ball_pos=ball_pos,
                    ball_vel=ball_vel,
                    own_team=own_team,
                    opp_team=opp_team,
                    game_state=game_state,
                    own_score=own_score,
                    opp_score=opp_score,
                    timestamp_ms=timestamp_ms,
                )
            raise BridgeTimeoutError(
                f"Bridge did not respond to frame {self._frame_counter}. "
                f"stderr: {stderr_out}"
            )

        if response.get("type") == "error":
            raise BridgeProtocolError(
                f"Bridge error on frame {self._frame_counter}: "
                f"{response.get('message', 'unknown')}"
            )

        if response.get("type") != "decision":
            raise BridgeProtocolError(
                f"Expected 'decision', got '{response.get('type')}': {response}"
            )

        response.setdefault("debug_transport_mode", "pipe_process")
        response.setdefault("debug_transport_reconnect_count", self._reconnect_count)
        response.setdefault("debug_last_reset_ms", self._last_reset_ms)
        return response

    def reset(self) -> None:
        """Reset BHuman state for a new episode."""
        if not self.is_running:
            return

        self._last_reset_ms = time.perf_counter() * 1000.0
        if self._transport_mode == "oneshot_file_fallback":
            self._frame_counter = 0
            logger.debug("Bridge reset handled locally in oneshot transport")
            return

        self._send({"type": "reset"})
        response = self._recv(timeout=self._config.frame_timeout)
        if response and response.get("type") == "ack":
            self._frame_counter = 0
            logger.debug("Bridge reset acknowledged")
        else:
            logger.warning("Bridge reset: unexpected response %s", response)

    def shutdown(self) -> None:
        """Gracefully shut down the bridge subprocess."""
        if self._transport_mode == "oneshot_file_fallback":
            self._process = None
            self._started = False
            self._transport_mode = "pipe_process"
            self._reset_reader_state()
            logger.info("Bridge one-shot transport terminated")
            return

        if self._process is None:
            return

        try:
            if self._process.poll() is None:
                try:
                    self._send({"type": "shutdown"})
                    self._process.wait(timeout=5.0)
                except BridgeStartupError:
                    logger.warning("Bridge shutdown write failed; killing subprocess")
                    self._process.kill()
                    self._process.wait(timeout=2.0)
        except (subprocess.TimeoutExpired, BrokenPipeError, OSError):
            logger.warning("Bridge did not exit gracefully, killing")
            self._process.kill()
            self._process.wait(timeout=2.0)
        finally:
            self._process = None
            self._started = False
            self._transport_mode = "pipe_process"
            self._reset_reader_state()
            logger.info("Bridge subprocess terminated")

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass

    # --- Internal I/O ---

    def _send(self, msg: dict) -> None:
        self._send_raw(json.dumps(msg, separators=(",", ":")))

    def _send_raw(self, line: str) -> None:
        if self._process is None or self._process.stdin is None:
            raise BridgeStartupError("Bridge stdin not available")
        try:
            self._process.stdin.write((line + "\n").encode("utf-8"))
            self._process.stdin.flush()
        except (BrokenPipeError, OSError) as e:
            stderr_out = self._read_stderr()
            raise BridgeStartupError(
                f"Bridge stdin write failed: {e}. stderr: {stderr_out}"
            ) from e

    def _recv(self, timeout: float = 1.0) -> Optional[dict]:
        """Read one JSON line from bridge stdout with timeout."""
        if self._process is None:
            return None

        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            remaining = max(0.0, deadline - time.perf_counter())
            try:
                line = self._stdout_queue.get(timeout=remaining)
            except queue.Empty:
                return None

            if line is None:
                if self._process.poll() is not None:
                    return None
                continue
            text = line.decode("utf-8", errors="replace").strip()
            if not text:
                continue
            if not text.startswith("{"):
                logger.debug("Skipping non-JSON bridge stdout line: %s", text)
                continue

            try:
                return json.loads(text)
            except json.JSONDecodeError as e:
                logger.debug("Skipping invalid JSON bridge stdout line: %s", text)
                continue

        return None

    def _start_reader_threads(self) -> None:
        if self._process is None:
            return
        self._reset_reader_state()
        self._reader_stop.clear()

        if self._process.stdout is not None:
            stdout_thread = threading.Thread(
                target=self._reader_loop,
                args=(self._process.stdout, self._stdout_queue),
                name="RLBridgeStdoutReader",
                daemon=True,
            )
            stdout_thread.start()
            self._reader_threads.append(stdout_thread)

        if self._process.stderr is not None:
            stderr_thread = threading.Thread(
                target=self._stderr_reader_loop,
                args=(self._process.stderr,),
                name="RLBridgeStderrReader",
                daemon=True,
            )
            stderr_thread.start()
            self._reader_threads.append(stderr_thread)

    def _reader_loop(self, pipe, out_queue: "queue.Queue[Optional[bytes]]") -> None:
        try:
            while not self._reader_stop.is_set():
                line = pipe.readline()
                if not line:
                    break
                out_queue.put(line)
        except Exception as exc:
            logger.warning("RLBridge stdout reader terminated with error: %s", exc)
        finally:
            out_queue.put(None)

    def _stderr_reader_loop(self, pipe) -> None:
        try:
            while not self._reader_stop.is_set():
                line = pipe.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="replace")
                with self._stderr_lock:
                    self._stderr_lines.append(text)
        except Exception as exc:
            logger.warning("RLBridge stderr reader terminated with error: %s", exc)

    def _reset_reader_state(self) -> None:
        self._reader_stop.set()
        for thread in self._reader_threads:
            if thread.is_alive():
                thread.join(timeout=0.1)
        self._reader_threads = []
        self._stdout_queue = queue.Queue()
        with self._stderr_lock:
            self._stderr_lines.clear()
        self._reader_stop = threading.Event()

    def _recv_unix_line(self, timeout: float) -> Optional[bytes]:
        """Read one raw stdout line on Unix-like systems."""
        if self._process is None or self._process.stdout is None:
            return None

        import select as sel
        ready, _, _ = sel.select([self._process.stdout], [], [], timeout)
        if not ready:
            return None
        return self._process.stdout.readline()

    def _recv_windows_line(self, timeout: float) -> Optional[bytes]:
        """Windows-compatible raw line read using a reader thread."""
        import threading
        import queue

        result_queue: queue.Queue = queue.Queue()

        def reader():
            try:
                line = self._process.stdout.readline()
                result_queue.put(line)
            except Exception as e:
                result_queue.put(e)

        thread = threading.Thread(target=reader, daemon=True)
        thread.start()
        thread.join(timeout=timeout)

        if result_queue.empty():
            return None

        line = result_queue.get_nowait()
        if isinstance(line, Exception):
            raise BridgeProtocolError(f"Read error: {line}")
        if not line:
            return None
        return line

    def _read_stderr(self) -> str:
        """Read available stderr output (non-blocking)."""
        with self._stderr_lock:
            return "".join(self._stderr_lines)

    def _shutdown_process_only(self) -> None:
        """Best-effort cleanup of the pipe-based subprocess after handshake failure."""
        if self._process is None:
            return
        try:
            if self._process.poll() is None:
                self._process.kill()
                self._process.wait(timeout=2.0)
        except Exception:
            pass
        finally:
            self._process = None
            self._reset_reader_state()

    def _switch_to_oneshot_fallback(self) -> None:
        """Switch transport mode after a Windows pipe transport failure."""
        self._shutdown_process_only()
        self._reconnect_count += 1
        self._transport_mode = "oneshot_file_fallback"
        self._started = True

    def _step_oneshot_file_transport(
        self,
        ball_pos: np.ndarray,
        ball_vel: np.ndarray,
        own_team: List[Dict[str, Any]],
        opp_team: List[Dict[str, Any]],
        game_state: str = "playing",
        own_score: int = 0,
        opp_score: int = 0,
        timestamp_ms: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Windows stable transport:
        run RLBridge.exe in one-shot mode with file-backed stdin/stdout/stderr.

        This avoids the crash path observed with anonymous stdin pipes on Windows.
        The bridge is also more stable when stdout/stderr are merged into a single
        redirected stream, mirroring the working manual smoke launcher.
        """
        self._frame_counter += 1
        if timestamp_ms is None:
            timestamp_ms = self._frame_counter * 33

        frame_json = frame_to_json(
            frame_id=self._frame_counter,
            ball_pos=ball_pos,
            ball_vel=ball_vel,
            own_team=own_team,
            opp_team=opp_team,
            game_state=game_state,
            own_score=own_score,
            opp_score=opp_score,
            timestamp_ms=timestamp_ms,
        )

        init_msg = json.dumps({
            "type": "init",
            "scenario": self._config.scenario,
            "location": self._config.location,
            "team_number": self._config.team_number,
            "player_number": self._config.player_number,
            "num_players": self._config.num_players,
        }, separators=(",", ":"))

        bridge_dir = _oneshot_runtime_dir()
        input_fd, input_path_str = tempfile.mkstemp(
            prefix=f"{ONESHOT_ARTIFACT_PREFIX}in_", suffix=".jsonl", dir=bridge_dir
        )
        output_fd, output_path_str = tempfile.mkstemp(
            prefix=f"{ONESHOT_ARTIFACT_PREFIX}out_", suffix=".txt", dir=bridge_dir
        )
        script_fd, script_path_str = tempfile.mkstemp(
            prefix=ONESHOT_ARTIFACT_PREFIX, suffix=".cmd", dir=bridge_dir
        )
        os.close(input_fd)
        os.close(output_fd)
        os.close(script_fd)

        input_path = Path(input_path_str)
        output_path = Path(output_path_str)
        script_path = Path(script_path_str)

        try:
            input_path.write_text(
                init_msg + "\n" + frame_json + "\n",
                encoding="utf-8",
            )

            script_path.write_text(
                "@echo off\n"
                f"\"{self._config.bridge_executable}\" "
                f"< \"{input_path}\" "
                f"> \"{output_path}\" "
                "2>&1\n",
                encoding="utf-8",
            )
            completed = subprocess.run(
                ["cmd.exe", "/c", str(script_path)],
                env=self._env,
                timeout=max(self._config.startup_timeout, self._config.frame_timeout) + 5.0,
                check=False,
            )

            merged_text = output_path.read_text(encoding="utf-8", errors="replace")

            parsed_objects: List[Dict[str, Any]] = []
            seen_serialized: set[str] = set()

            # Fast path: plain JSONL when stdout is clean.
            for line in merged_text.splitlines():
                line = line.strip()
                if not line.startswith("{"):
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    serialized = json.dumps(obj, sort_keys=True, separators=(",", ":"))
                    if serialized not in seen_serialized:
                        parsed_objects.append(obj)
                        seen_serialized.add(serialized)

            # Always run the mixed-output extractor too. A common RLBridge
            # failure mode is: `ready` arrives as a clean JSON line, but the
            # `decision` payload is glued to bridge logs on the same line.
            for obj in _extract_json_objects_from_mixed_output(merged_text):
                serialized = json.dumps(obj, sort_keys=True, separators=(",", ":"))
                if serialized not in seen_serialized:
                    parsed_objects.append(obj)
                    seen_serialized.add(serialized)

            ready_response = None
            decision_response = None
            for obj in parsed_objects:
                if obj.get("type") == "ready":
                    ready_response = obj
                elif obj.get("type") == "decision":
                    decision_response = obj
                    break
                elif obj.get("type") == "error":
                    raise BridgeProtocolError(
                        f"Bridge error in oneshot transport: {obj.get('message', 'unknown')}"
                    )

            if completed.returncode != 0:
                raise BridgeStartupError(
                    f"Bridge oneshot transport failed with return code {completed.returncode}. "
                    f"output: {merged_text}"
                )
            if ready_response is None:
                raise BridgeStartupError(
                    f"Bridge oneshot transport missing ready response. output: {merged_text}"
                )
            if decision_response is None:
                raise BridgeTimeoutError(
                    f"Bridge oneshot transport missing decision response. output: {merged_text}"
                )

            decision_response.setdefault("debug_transport_mode", "oneshot_file_fallback")
            decision_response.setdefault("debug_transport_process_returncode", completed.returncode)
            decision_response.setdefault("debug_transport_reconnect_count", self._reconnect_count)
            decision_response.setdefault("debug_last_reset_ms", self._last_reset_ms)
            return decision_response
        finally:
            for path in (input_path, output_path, script_path):
                try:
                    path.unlink(missing_ok=True)
                except Exception:
                    pass


# ---------------------------------------------------------------------------
# MOCK BRIDGE — for testing without compiled C++ binary
# ---------------------------------------------------------------------------

class MockBHumanBridge:
    """
    Mock bridge that returns contract-valid decisions without C++ subprocess.
    Used for development and testing of the Python integration layer.
    """

    def __init__(self):
        self._frame_counter = 0
        self._started = False

    @property
    def is_running(self) -> bool:
        return self._started

    def start(self) -> None:
        self._started = True
        self._frame_counter = 0
        logger.info("MockBHumanBridge started")

    def step(
        self,
        ball_pos: np.ndarray,
        ball_vel: np.ndarray,
        own_team: List[Dict[str, Any]],
        opp_team: List[Dict[str, Any]],
        game_state: str = "playing",
        own_score: int = 0,
        opp_score: int = 0,
        timestamp_ms: Optional[int] = None,
    ) -> Dict[str, Any]:
        self._frame_counter += 1

        # Find nearest own player to ball
        active_number = 1
        min_dist = float("inf")
        for p in own_team:
            dx = p["x"] - ball_pos[0]
            dy = p["y"] - ball_pos[1]
            dist = (dx * dx + dy * dy) ** 0.5
            if dist < min_dist and p.get("upright", True):
                min_dist = dist
                active_number = p["number"]

        return {
            "type": "decision",
            "frame_id": self._frame_counter,
            "player_number": active_number,
            "skill_request": "dribble",
            "motion_request": "dribble",
            "motion_speed": [250.0, 0.0, 0.0],
            "kick_type": "none",
            "kick_length_mm": 0.0,
            "last_kick_type": "unavailable",
            "last_kick_timestamp_ms": 0,
            "strategy_status": "s4v4_scaled",
            "current_tactic": "t121_4v4_scaled",
            "behavior_status": "playing",
            "activation_graph": [
                {
                    "option": "PlayBall",
                    "depth": 0,
                    "state": "execute",
                    "option_time": self._frame_counter * 33,
                    "state_time": self._frame_counter * 33,
                    "parameters": [],
                },
                {
                    "option": "HandleBall",
                    "depth": 1,
                    "state": "dribble",
                    "option_time": self._frame_counter * 33,
                    "state_time": self._frame_counter * 33,
                    "parameters": ["mock=true"],
                },
            ],
            "active_agent_number": active_number,
            "decision_time_us": 100,
        }

    def reset(self) -> None:
        self._frame_counter = 0

    def shutdown(self) -> None:
        self._started = False
