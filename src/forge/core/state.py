# Copyright 2026. Generic locked state manager with atomic writes.

import fcntl
import json
import os
import tempfile
from pathlib import Path
from typing import Callable, Generic, TypeVar

T = TypeVar("T")


class LockedStateManager(Generic[T]):
    """File-backed state with fcntl locking and atomic writes."""

    def __init__(
        self,
        state_file: Path,
        serialize: Callable[[T], dict],
        deserialize: Callable[[dict], T],
    ):
        self.state_file = state_file
        self._lock_file = state_file.with_suffix(".json.lock")
        self._serialize = serialize
        self._deserialize = deserialize

    def exists(self) -> bool:
        return self.state_file.is_file()

    def load(self) -> T:
        self._lock_file.parent.mkdir(parents=True, exist_ok=True)
        lock_fd = os.open(str(self._lock_file), os.O_RDWR | os.O_CREAT)
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_SH)
            data = self.state_file.read_text(encoding="utf-8")
            return self._deserialize(json.loads(data))
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)

    def save(self, state: T) -> None:
        self._atomic_write(self._serialize(state))

    def update(self, mutator: Callable[[T], None]) -> T:
        self._lock_file.parent.mkdir(parents=True, exist_ok=True)
        lock_fd = os.open(str(self._lock_file), os.O_RDWR | os.O_CREAT)
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            data = self.state_file.read_text(encoding="utf-8")
            state = self._deserialize(json.loads(data))
            mutator(state)
            self._atomic_write(self._serialize(state))
            return state
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)

    def _atomic_write(self, d: dict) -> None:
        parent = self.state_file.parent
        parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(dir=str(parent), suffix=".tmp")
        closed = False
        try:
            os.write(fd, json.dumps(d, indent=2).encode("utf-8"))
            os.write(fd, b"\n")
            os.close(fd)
            closed = True
            os.rename(tmp_path, str(self.state_file))
        except BaseException:
            if not closed:
                try:
                    os.close(fd)
                except OSError:
                    pass
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise
