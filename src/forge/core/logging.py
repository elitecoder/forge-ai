# Copyright 2026. Unified activity logging and status file writing.

import os
import tempfile
import threading
from datetime import datetime, timezone


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


def log_activity(log_path: str, source: str, message: str) -> None:
    if not log_path:
        return
    ts = utc_timestamp()
    line = f"[{ts}] {source}  {message}\n" if source else f"[{ts}] {message}\n"
    try:
        with open(log_path, "a") as f:
            f.write(line)
    except OSError:
        pass


def atomic_write_file(path: str, content: str) -> None:
    parent = os.path.dirname(path)
    fd, tmp = tempfile.mkstemp(dir=parent, suffix=".tmp")
    try:
        os.write(fd, content.encode("utf-8"))
        os.close(fd)
        os.rename(tmp, path)
    except BaseException:
        try:
            os.close(fd)
        except OSError:
            pass
        if os.path.exists(tmp):
            os.unlink(tmp)


class StatusWriter:
    """Periodic status file writer.

    Calls `write_fn()` at a fixed interval on a background thread.
    """

    def __init__(self, write_fn, interval: float = 15.0):
        self._write_fn = write_fn
        self._interval = interval
        self._stop_event: threading.Event | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._stop_event = threading.Event()
        stop = self._stop_event
        write_fn = self._write_fn
        interval = self._interval

        def _updater():
            while not stop.is_set():
                stop.wait(interval)
                if not stop.is_set():
                    try:
                        write_fn()
                    except Exception:
                        pass

        self._thread = threading.Thread(target=_updater, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._stop_event:
            self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        self._stop_event = None
        self._thread = None

    def write_now(self) -> None:
        try:
            self._write_fn()
        except Exception:
            pass
