from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from config import BHUMAN_DIR, SIMROBOT_BIN

# Number of physical cores to leave unbound for Python / torch / CUDA.
# With the observed topology (core N ↔ CPUs N and N+16 on a 16-core/32-thread
# machine) this reserves physical cores 0–(N-1) and their HT siblings.
_DEFAULT_PYTHON_RESERVED_CORES = 4


def _compute_lp_num_threads(n_slots: int, *, python_reserved_cores: int = _DEFAULT_PYTHON_RESERVED_CORES) -> int:
    """Compute per-slot LP_NUM_THREADS to avoid llvmpipe render-thread oversubscription.

    llvmpipe defaults to os.cpu_count() render threads per instance.  With N
    slots that creates N × cpu_count threads fighting for cpu_count CPUs — 6×
    oversubscription at N=6 on a 32-core machine.  Dividing evenly keeps total
    render threads ≈ available CPUs.
    """
    total_logical = os.cpu_count() or 4
    available = max(total_logical - python_reserved_cores * 2, total_logical // 2)
    return max(2, available // n_slots)


def _build_cpu_affinity_map(
    player_numbers: tuple[int, ...],
    *,
    python_reserved_cores: int = _DEFAULT_PYTHON_RESERVED_CORES,
) -> dict[int, str] | None:
    """Return {player_number: taskset_cpu_list} or None if pinning is not possible.

    NOTE: On machines where llvmpipe spawns many render threads, pinning to a
    small core set often hurts rather than helps.  This function is kept for
    experimentation but cpu_affinity defaults to False in VisibleParallelRuntime.
    """
    try:
        result = subprocess.run(["taskset", "--version"], capture_output=True)
        if result.returncode != 0:
            return None
    except FileNotFoundError:
        return None

    total_logical = os.cpu_count() or 0
    if total_logical < 2:
        return None

    # Detect HT siblings: on this machine core K ↔ CPUs K and K+half.
    half = total_logical // 2
    available_physical = list(range(python_reserved_cores, half))
    n_slots = len(player_numbers)
    if len(available_physical) < n_slots:
        return None

    cores_per_slot = len(available_physical) // n_slots
    affinity: dict[int, str] = {}
    for slot_idx, player in enumerate(player_numbers):
        start = available_physical[slot_idx * cores_per_slot]
        end = available_physical[slot_idx * cores_per_slot + cores_per_slot - 1]
        cpus = list(range(start, end + 1)) + list(range(start + half, end + half + 1))
        affinity[player] = ",".join(str(c) for c in sorted(cpus))
    return affinity


@dataclass
class VisibleParallelRuntime:
    player_numbers: tuple[int, ...]
    shm_name: str
    display_base: int
    scene_name: str = "RL1v0_3D_visible"
    screen: str = "1280x960x24"
    simrobot_bin: Path = SIMROBOT_BIN
    xvfb_bin: Path | None = None
    temp_root: Path | None = None
    simrobot_env: dict[str, str] = field(default_factory=dict)
    # cpu_affinity: taskset-based pinning.  Disabled by default — on machines
    # where llvmpipe spawns cpu_count() render threads, pinning to a small core
    # set causes worse context switching than letting the OS scheduler roam.
    # lp_num_threads is the effective per-slot fix (limits render threads).
    cpu_affinity: bool = False
    python_reserved_cores: int = _DEFAULT_PYTHON_RESERVED_CORES
    # lp_num_threads=0 → auto-compute from cpu_count / n_slots.
    lp_num_threads: int = 0

    def __post_init__(self) -> None:
        if self.xvfb_bin is None:
            self.xvfb_bin = self._find_xvfb()
        if self.temp_root is None:
            configured_root = os.getenv("RL_VISIBLE_RUNTIME_ROOT")
            if configured_root:
                self.temp_root = Path(configured_root)
                if self.temp_root.exists():
                    shutil.rmtree(self.temp_root, ignore_errors=True)
                self.temp_root.mkdir(parents=True, exist_ok=True)
            else:
                self.temp_root = Path(tempfile.mkdtemp(prefix="pybh_visible_parallel_"))
        self._xvfb_processes: list[subprocess.Popen[bytes]] = []
        self._simrobot_processes: list[subprocess.Popen[bytes]] = []
        self._cpu_affinity_map: dict[int, str] | None = None
        if self.cpu_affinity:
            self._cpu_affinity_map = _build_cpu_affinity_map(
                self.player_numbers,
                python_reserved_cores=self.python_reserved_cores,
            )
            if self._cpu_affinity_map:
                print(
                    f"[cpu_affinity] taskset pinning {len(self.player_numbers)} slots "
                    f"(reserved_cores=0-{self.python_reserved_cores - 1}): "
                    + "  ".join(
                        f"player{p}→{cpus}"
                        for p, cpus in self._cpu_affinity_map.items()
                    ),
                    flush=True,
                )
        _env_lp = int(os.getenv("RL_LP_NUM_THREADS", "0"))
        effective_lp = (
            self.lp_num_threads
            if self.lp_num_threads > 0
            else _env_lp
            if _env_lp > 0
            else _compute_lp_num_threads(
                len(self.player_numbers),
                python_reserved_cores=self.python_reserved_cores,
            )
        )
        self._lp_num_threads: int = effective_lp
        print(
            f"[lp_num_threads] LP_NUM_THREADS={effective_lp} per slot "
            f"({len(self.player_numbers)} slots × {effective_lp} = "
            f"{len(self.player_numbers) * effective_lp} total render threads "
            f"on {os.cpu_count() or '?'} logical CPUs)",
            flush=True,
        )

    @staticmethod
    def _find_xvfb() -> Path:
        candidates = [
            Path("/usr/bin/Xvfb"),
            Path("/tmp/xvfb-local/usr/bin/Xvfb"),
        ]
        for candidate in candidates:
            if candidate.exists() and os.access(candidate, os.X_OK):
                return candidate
        resolved = shutil.which("Xvfb")
        if resolved:
            return Path(resolved)
        raise FileNotFoundError(
            "Xvfb was not found. Install xvfb or provide /tmp/xvfb-local/usr/bin/Xvfb."
        )

    def start(self) -> None:
        if self._xvfb_processes or self._simrobot_processes:
            return
        self._prepare_roots()
        self._launch_xvfb()
        self._launch_simrobots()

    def close(self) -> None:
        for process in reversed(self._simrobot_processes):
            self._terminate(process)
        for process in reversed(self._xvfb_processes):
            self._terminate(process)
        self._simrobot_processes.clear()
        self._xvfb_processes.clear()
        if (
            self.temp_root
            and self.temp_root.exists()
            and os.getenv("RL_VISIBLE_KEEP_RUNTIME_ROOT", "0") not in {"1", "true", "True", "yes", "on"}
        ):
            shutil.rmtree(self.temp_root, ignore_errors=True)

    def __enter__(self) -> "VisibleParallelRuntime":
        self.start()
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        self.close()

    def _prepare_roots(self) -> None:
        assert self.temp_root is not None
        config_src = BHUMAN_DIR / "Config"
        for player_number in self.player_numbers:
            root = self._slot_root(player_number)
            root.mkdir(parents=True, exist_ok=True)
            config_dst = root / "Config"
            if config_dst.exists():
                shutil.rmtree(config_dst)
            shutil.copytree(config_src, config_dst)
            self._rewrite_settings(config_dst / "settings.cfg", player_number)
            scene_path = config_dst / "Scenes" / f"{self.scene_name}.ros2"
            self._rewrite_scene_robot_name(scene_path, player_number)
            self._ensure_stage_obstacle_bodies(scene_path, player_number)

    def _taskset_prefix(self, player_number: int) -> list[str]:
        if self._cpu_affinity_map and player_number in self._cpu_affinity_map:
            return ["taskset", "-c", self._cpu_affinity_map[player_number]]
        return []

    def _launch_xvfb(self) -> None:
        assert self.xvfb_bin is not None
        for player_number in self.player_numbers:
            display = self.display_base + player_number
            log_path = self._slot_root(player_number) / "xvfb.log"
            log_file = log_path.open("wb")
            process = subprocess.Popen(
                self._taskset_prefix(player_number) + [
                    str(self.xvfb_bin),
                    f":{display}",
                    "-screen",
                    "0",
                    self.screen,
                    "-ac",
                    "-nolisten",
                    "tcp",
                ],
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
            self._xvfb_processes.append(process)
        time.sleep(1.0)

    def _launch_simrobots(self) -> None:
        for player_number in self.player_numbers:
            display = self.display_base + player_number
            root = self._slot_root(player_number)
            scene_path = root / "Config" / "Scenes" / f"{self.scene_name}.ros2"
            log_file = (root / "simrobot.log").open("wb")
            env = os.environ.copy()
            env.update(
                {
                    "DISPLAY": f":{display}",
                    "PYBH_RL_SHM_NAME": self.shm_name,
                    "LIBGL_ALWAYS_SOFTWARE": "1",
                    "GALLIUM_DRIVER": "llvmpipe",
                    "LP_NUM_THREADS": str(self._lp_num_threads),
                    **self.simrobot_env,
                }
            )
            process = subprocess.Popen(
                self._taskset_prefix(player_number) + [
                    str(self.simrobot_bin),
                    str(scene_path.relative_to(root)),
                ],
                cwd=root,
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
            self._simrobot_processes.append(process)
        time.sleep(16.0)

    def _slot_root(self, player_number: int) -> Path:
        assert self.temp_root is not None
        return self.temp_root / f"player_{player_number}"

    @staticmethod
    def _rewrite_settings(path: Path, player_number: int) -> None:
        text = path.read_text(encoding="utf-8")
        replacements = {
            "teamNumber": "24",
            "playerNumber": str(player_number),
            "magicNumber": "24",
        }
        for key, value in replacements.items():
            text = _replace_cfg_assignment(text, key, value)
        path.write_text(text, encoding="utf-8")

    @staticmethod
    def _rewrite_scene_robot_name(path: Path, player_number: int) -> None:
        text = path.read_text(encoding="utf-8")
        if 'name="robot2"' in text or 'name="robot3"' in text or 'name="robot4"' in text:
            return
        text = text.replace('name="robot1"', f'name="robot{player_number}"', 1)
        path.write_text(text, encoding="utf-8")

    @staticmethod
    def _ensure_stage_obstacle_bodies(path: Path, player_number: int) -> None:
        """Add dummy bodies needed by stage 2+ world resets to per-slot 1v0 scenes.

        Each visible training slot is an independent SimRobot process whose controlled
        robot number matches the slot/player number.  Stage 2+ reset specs also place
        support teammates 2/3 and opponent obstacles 1/2.  The fast visible 1v0 scene
        only contains the controlled robot, so promotion would otherwise fail with
        "could not find teammate/opponent robot".  We add only non-controlled dummy
        bodies so the per-slot scene remains small and avoids duplicate robot names.
        """
        text = path.read_text(encoding="utf-8")
        if 'name="robot21"' in text:
            return

        bodies: list[str] = []
        for number, y in ((1, 1.1), (2, -1.1), (3, -2.6), (4, 2.6)):
            if int(player_number) == number:
                continue
            bodies.append(
                f'''      <Body ref="NaoDummy" name="robot{number}">
        <Set name="NaoColor" value="black"/>
        <Translation x="1.6" y="{y}" z="320mm"/>
        <Rotation z="180degree"/>
      </Body>'''
            )
        bodies.extend(
            [
                '''      <Body ref="NaoDummy" name="robot21">
        <Set name="NaoColor" value="red"/>
        <Translation x="-1.6" y="1.1" z="320mm"/>
        <Rotation z="0degree"/>
      </Body>''',
                '''      <Body ref="NaoDummy" name="robot22">
        <Set name="NaoColor" value="red"/>
        <Translation x="-1.6" y="2.6" z="320mm"/>
        <Rotation z="0degree"/>
      </Body>''',
                '''      <Body ref="NaoDummy" name="robot23">
        <Set name="NaoColor" value="red"/>
        <Translation x="-2.6" y="0.0" z="320mm"/>
        <Rotation z="0degree"/>
      </Body>''',
            ]
        )
        extras = "\n".join(bodies)
        if '<Compound name="extras"/>' in text:
            text = text.replace('<Compound name="extras"/>', f'<Compound name="extras">\n{extras}\n    </Compound>')
        else:
            text = text.replace('<Compound name="balls">', f'<Compound name="extras">\n{extras}\n    </Compound>\n\n    <Compound name="balls">')
        path.write_text(text, encoding="utf-8")

    @staticmethod
    def _terminate(process: subprocess.Popen[bytes]) -> None:
        if process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            process.kill()
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                print(
                    f"[visible-runtime] process did not exit after SIGKILL: pid={process.pid}",
                    flush=True,
                )


def _replace_cfg_assignment(text: str, key: str, value: str) -> str:
    lines = []
    replaced = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(f"{key} ") or stripped.startswith(f"{key}="):
            lines.append(f"{key} = {value};")
            replaced = True
        else:
            lines.append(line)
    if not replaced:
        lines.append(f"{key} = {value};")
    return "\n".join(lines) + "\n"
