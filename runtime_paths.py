from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent

_VALID_TRANSPORTS = {"pipe_process", "oneshot_file_fallback"}
_VALID_BRIDGE_MODES = {"persistent_2d_cognition", "embedded_2d_motion"}


def _platform_build_dir() -> str:
    if os.name == "nt":
        return "Windows"
    if sys.platform == "darwin":
        return "macOS"
    return "Linux"


def _exe_suffix() -> str:
    return ".exe" if os.name == "nt" else ""


@lru_cache(maxsize=1)
def _repo_dotenv() -> dict[str, str]:
    values: dict[str, str] = {}
    dotenv_path = REPO_ROOT / ".env"
    if not dotenv_path.exists():
        return values

    for raw_line in dotenv_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key:
            values[key] = value
    return values


def _setting(name: str) -> str:
    env_value = os.environ.get(name, "").strip()
    if env_value:
        return env_value
    return _repo_dotenv().get(name, "").strip()


def preferred_bridge_mode() -> str:
    requested = _setting("BHUMAN_BRIDGE_MODE")
    if requested in _VALID_BRIDGE_MODES:
        return requested
    return "persistent_2d_cognition"


def _resolve_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def resolve_cfg_root() -> str | None:
    candidates = [
        _setting("BHUMAN_CFG_ROOT"),
        "../SabanaHerons2026/Config",
        "../SabanaHerons2026-ScaledDimensions/Config",
        "../SabanaHerons2026-ScaledDimensions/SabanaHerons2026-ScaledDimensions/Config",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        resolved = _resolve_path(candidate)
        if resolved.is_dir():
            return str(resolved)
    return None


def resolve_bhuman_root() -> Path:
    cfg_root = resolve_cfg_root()
    if cfg_root is not None:
        return Path(cfg_root).resolve().parent

    candidates = [
        _setting("BHUMAN_ROOT"),
        "../SabanaHerons2026",
        "../SabanaHerons2026-ScaledDimensions",
        "../SabanaHerons2026-ScaledDimensions/SabanaHerons2026-ScaledDimensions",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        resolved = _resolve_path(candidate)
        if (resolved / "Src").is_dir() and (resolved / "Config").is_dir():
            return resolved
    return (REPO_ROOT / ".." / "SabanaHerons2026").resolve()


def preferred_bridge_transport() -> str:
    requested = _setting("BHUMAN_BRIDGE_TRANSPORT")
    if requested in _VALID_TRANSPORTS:
        return requested
    if os.name == "nt":
        bhuman_root = resolve_bhuman_root()
        built_bridge_candidates = (
            bhuman_root / "Build" / "Windows" / "RLBridge" / "Release" / "RLBridge.exe",
            bhuman_root / "Build" / "Windows" / "RLBridge" / "Release" / "bhuman_bridge.exe",
            bhuman_root / "Build" / "Windows" / "RLBridge" / "Develop" / "RLBridge.exe",
            bhuman_root / "Build" / "Windows" / "RLBridge" / "Develop" / "bhuman_bridge.exe",
        )
        if any(candidate.exists() for candidate in built_bridge_candidates):
            return "pipe_process"
        return "oneshot_file_fallback"
    # Linux/macOS only support the persistent pipe transport. The one-shot
    # fallback is a Windows-specific workaround that relies on cmd.exe.
    return "pipe_process"


def stable_bridge_transports() -> tuple[str, ...]:
    preferred = preferred_bridge_transport()
    if os.name == "nt":
        ordered = [preferred, "oneshot_file_fallback", "pipe_process"]
    else:
        # Non-Windows platforms do not support the cmd.exe-based one-shot
        # fallback, so we advertise only the persistent pipe transport.
        ordered = [preferred, "pipe_process"]

    stable: list[str] = []
    for mode in ordered:
        if mode in _VALID_TRANSPORTS and mode not in stable:
            stable.append(mode)
    return tuple(stable)


def resolve_bridge_executable() -> Path:
    override = _setting("BHUMAN_BRIDGE_EXE")
    if override:
        override_path = _resolve_path(override)
        if override_path.exists():
            return override_path

    bhuman_root = resolve_bhuman_root()
    platform_dir = _platform_build_dir()
    suffix = _exe_suffix()

    candidates = [
        bhuman_root / "Build" / platform_dir / "RLBridge" / "Release" / f"RLBridge{suffix}",
        bhuman_root / "Build" / platform_dir / "RLBridge" / "Release" / f"bhuman_bridge{suffix}",
        bhuman_root / "Build" / platform_dir / "RLBridge" / "Develop" / f"RLBridge{suffix}",
        bhuman_root / "Build" / platform_dir / "RLBridge" / "Develop" / f"bhuman_bridge{suffix}",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    local_candidates = [
        REPO_ROOT / "bridge" / f"bhuman_bridge{suffix}",
        REPO_ROOT / "bridge" / "build" / "Release" / f"bhuman_bridge{suffix}",
        REPO_ROOT / "bridge" / "build" / "Release" / f"RLBridge{suffix}",
    ]
    allow_local = _setting("BHUMAN_ALLOW_LOCAL_BRIDGE_EXE").lower() in {"1", "true", "yes"}
    if allow_local:
        for candidate in local_candidates:
            if candidate.exists():
                return candidate

    return candidates[0]
