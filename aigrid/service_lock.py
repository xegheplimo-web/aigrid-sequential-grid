"""Exclusive pidfile ownership for the service CLI."""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)
_active: set[Path] = set()
_guard = threading.Lock()


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except PermissionError:
        return True
    except (OSError, OverflowError):
        return False
    if sys.platform.startswith("linux"):
        try:
            return bool(Path(f"/proc/{pid}/cmdline").read_bytes())
        except PermissionError:
            return True
        except OSError:
            return False
    return True


class ServiceLock:
    def __init__(self, path: str | Path):
        self.path = Path(path).resolve()
        self.pid = os.getpid()
        self.owned = False
        self.eventlog = None

    def _pid(self) -> int | None:
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))["pid"]
            return value if type(value) is int and value > 0 else None
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def emit(self, kind: str) -> None:
        payload = {"path": self.path.name}
        logger.info("%s %s", kind, payload)
        sink = getattr(self.eventlog, "log", None)
        if callable(sink):
            sink(kind, "", payload)

    def acquire(self) -> int:
        with _guard:
            if self.path in _active:
                logger.warning("SERVICE_LOCKED %s", self.path.name)
                return 3
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                for attempt in range(2):
                    try:
                        fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                    except FileExistsError:
                        pid = self._pid()
                        if attempt or (pid is not None and pid != self.pid and _alive(pid)):
                            logger.warning("SERVICE_LOCKED %s", self.path.name)
                            return 3
                        self.path.unlink(missing_ok=True)
                        continue
                    self.owned = True
                    _active.add(self.path)
                    with os.fdopen(fd, "w", encoding="utf-8") as stream:
                        json.dump(
                            {
                                "pid": self.pid,
                                "ts_ms": int(time.time() * 1000),
                                "argv": " ".join(sys.argv)[:240],
                            },
                            stream,
                        )
                    return 0
            except OSError as exc:
                logger.error("SERVICE_LOCK_ERROR state directory unwritable: %s", exc)
                if self.owned:
                    self.owned = False
                    _active.discard(self.path)
                    if self._pid() == self.pid:
                        try:
                            self.path.unlink(missing_ok=True)
                        except OSError:
                            logger.warning("SERVICE_LOCK_RELEASE_ERROR %s", self.path.name)
                return 2
        return 3

    def release(self) -> None:
        with _guard:
            if not self.owned:
                return
            self.owned = False
            _active.discard(self.path)
            if self._pid() == self.pid:
                try:
                    self.path.unlink()
                except OSError as exc:
                    logger.warning("SERVICE_LOCK_RELEASE_ERROR: %s", exc)
                else:
                    self.emit("SERVICE_LOCK_RELEASED")
