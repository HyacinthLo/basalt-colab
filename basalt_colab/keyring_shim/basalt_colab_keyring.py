"""Headless-safe keyring backend shim for google-colab-cli >= 0.7.6 on Linux.

Loaded exclusively inside the `colab` subprocess via PYTHON_KEYRING_BACKEND when
basalt-colab invokes Colab CLI on Linux. Delegates to any working native OS
keyring (e.g., GNOME Keyring / KWallet) when available; otherwise falls back to
an owner-only (0600) atomic JSON file so headless Linux, WSL2, Docker, Remote
SSH, and Termux environments can persist OAuth tokens across subprocesses.
"""
from __future__ import annotations

import importlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

try:
    _kr_backend: Any = importlib.import_module("keyring.backend")
    _kr_errors: Any = importlib.import_module("keyring.errors")
    _KeyringBase: Any = _kr_backend.KeyringBackend
    _PasswordDeleteError: type[Exception] = _kr_errors.PasswordDeleteError
except ImportError:
    _kr_backend = None
    _KeyringBase = object

    class _FallbackDeleteError(Exception):
        pass

    _PasswordDeleteError = _FallbackDeleteError


_SKIP_MODULES = frozenset({
    __name__,
    "keyring.backends.fail",
    "keyring.backends.null",
    "keyring.backends.chainer",
})


def _native_backends() -> list[Any]:
    """Return viable native OS keyring backends sorted by descending priority."""
    if _kr_backend is None or not os.environ.get("DBUS_SESSION_BUS_ADDRESS"):
        return []
    candidates: list[tuple[float, Any]] = []
    try:
        all_kr = _kr_backend.get_all_keyring()
    except Exception:
        return []
    for kr in all_kr:
        cls = type(kr)
        if cls.__module__ in _SKIP_MODULES or cls.__name__ == "FallbackFileKeyring":
            continue
        try:
            prio = float(kr.priority)
        except Exception:
            continue
        if prio > 0:
            candidates.append((prio, kr))
    candidates.sort(key=lambda item: item[0], reverse=True)
    return [kr for _, kr in candidates]


def _store_path() -> Path:
    raw = os.environ.get("BASALT_COLAB_KEYRING_FILE", "").strip()
    if raw:
        return Path(raw).expanduser()
    xdg = os.environ.get("XDG_STATE_HOME", "").strip()
    base = Path(xdg).expanduser() if xdg else (Path.home() / ".local" / "state")
    return base / "basalt-colab" / "colab_keyring.json"


def _read_store(path: Path) -> dict[str, str]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(k): str(v) for k, v in data.items() if isinstance(k, str) and isinstance(v, str)}


def _write_store(path: Path, data: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if os.name != "nt":
        try:
            os.chmod(path.parent, 0o700)
        except OSError:
            pass
    payload = (json.dumps(data, sort_keys=True) + "\n").encode("utf-8")
    fd, tmp_name = tempfile.mkstemp(prefix=".colab_keyring.", dir=str(path.parent))
    try:
        if os.name != "nt":
            try:
                os.fchmod(fd, 0o600)
            except OSError:
                pass
        with os.fdopen(fd, "wb") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _entry_key(service: str, username: str) -> str:
    return f"{service}\0{username}"


class FallbackFileKeyring(_KeyringBase):
    """Uses native OS keyring when working; falls back to 0600 file store on headless Linux."""

    priority = 1

    def get_password(self, service: str, username: str) -> str | None:
        for native in _native_backends():
            try:
                return native.get_password(service, username)
            except Exception:
                continue
        return _read_store(_store_path()).get(_entry_key(service, username))

    def set_password(self, service: str, username: str, password: str) -> None:
        for native in _native_backends():
            try:
                native.set_password(service, username, password)
                return
            except Exception:
                continue
        path = _store_path()
        data = _read_store(path)
        data[_entry_key(service, username)] = password
        _write_store(path, data)

    def delete_password(self, service: str, username: str) -> None:
        for native in _native_backends():
            try:
                native.delete_password(service, username)
                return
            except _PasswordDeleteError:
                raise
            except Exception:
                continue
        path = _store_path()
        data = _read_store(path)
        key = _entry_key(service, username)
        if key not in data:
            raise _PasswordDeleteError(f"Password not found for {service}:{username}")
        del data[key]
        _write_store(path, data)
