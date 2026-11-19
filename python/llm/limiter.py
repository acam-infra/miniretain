"""Bounded concurrency limiters (in-process and cross-process file-slot).

Prevents multi-job amplification (Go runtime.max_workers * llm.concurrency)
against a shared upstream LLM gateway such as geminiweb2api Go.
"""
from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import threading
import time

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX fallback
    fcntl = None


class ConcurrencyLimiter:
    """Thread-safe bounded concurrency budget with peak & wait metrics."""

    def __init__(self, limit: int) -> None:
        self.limit = max(1, int(limit))
        self._sem = threading.BoundedSemaphore(self.limit)
        self._lock = threading.Lock()
        self._inflight = 0
        self.peak_inflight = 0
        self.total_acquisitions = 0
        self.total_wait_s = 0.0

    @contextmanager
    def acquire(self):
        t0 = time.perf_counter()
        self._sem.acquire()
        wait_s = time.perf_counter() - t0
        with self._lock:
            self._inflight += 1
            if self._inflight > self.peak_inflight:
                self.peak_inflight = self._inflight
            self.total_acquisitions += 1
            self.total_wait_s += wait_s
        try:
            yield
        finally:
            with self._lock:
                self._inflight -= 1
            self._sem.release()

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "limit": self.limit,
                "inflight": self._inflight,
                "peak_inflight": self.peak_inflight,
                "total_acquisitions": self.total_acquisitions,
                "wait_s": round(self.total_wait_s, 3),
            }


class FileSlotConcurrencyLimiter(ConcurrencyLimiter):
    """Cross-process global concurrency limiter backed by POSIX flock slot files.

    Each in-flight LLM call acquires both a local semaphore permit (bounded by
    local_limit) and an exclusive non-blocking flock on one of `global_limit`
    slot files under `slot_dir`. Because the OS kernel automatically releases
    flock descriptors on process exit or SIGKILL, crashed workers never leak slots.
    """

    def __init__(
        self,
        local_limit: int,
        global_limit: int,
        slot_dir: str | Path,
        poll_interval_s: float = 0.015,
    ) -> None:
        self.global_limit = max(1, int(global_limit))
        effective_local = min(max(1, int(local_limit)), self.global_limit)
        super().__init__(effective_local)
        self.slot_dir = Path(slot_dir)
        self.slot_dir.mkdir(parents=True, exist_ok=True)
        self.poll_interval_s = max(0.005, float(poll_interval_s))
        self._slot_paths = [
            self.slot_dir / f"slot_{i}.lock" for i in range(self.global_limit)
        ]
        self._rr = 0
        self._rr_lock = threading.Lock()

    def _acquire_slot_fd(self) -> int | None:
        if fcntl is None:
            return None
        while True:
            with self._rr_lock:
                start = self._rr
                self._rr = (self._rr + 1) % self.global_limit
            for offset in range(self.global_limit):
                idx = (start + offset) % self.global_limit
                path = self._slot_paths[idx]
                fd = os.open(str(path), os.O_CREAT | os.O_RDWR, 0o644)
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    return fd
                except BlockingIOError:
                    os.close(fd)
            time.sleep(self.poll_interval_s)

    def _release_slot_fd(self, fd: int | None) -> None:
        if fd is None:
            return
        try:
            if fcntl is not None:
                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)

    @contextmanager
    def acquire(self):
        t0 = time.perf_counter()
        self._sem.acquire()
        fd = None
        try:
            fd = self._acquire_slot_fd()
            wait_s = time.perf_counter() - t0
            with self._lock:
                self._inflight += 1
                if self._inflight > self.peak_inflight:
                    self.peak_inflight = self._inflight
                self.total_acquisitions += 1
                self.total_wait_s += wait_s
            try:
                yield
            finally:
                with self._lock:
                    self._inflight -= 1
        finally:
            self._release_slot_fd(fd)
            self._sem.release()

    def snapshot(self) -> dict:
        snap = super().snapshot()
        snap["global_limit"] = self.global_limit
        snap["slot_dir"] = str(self.slot_dir)
        return snap


def build_limiter_from_env(local_concurrency: int) -> ConcurrencyLimiter:
    """Create a cross-process limiter when MINIRETAIN_LLM_GLOBAL_CONCURRENCY is set."""
    raw_global = os.environ.get("MINIRETAIN_LLM_GLOBAL_CONCURRENCY", "").strip()
    slot_dir = os.environ.get("MINIRETAIN_LLM_SLOT_DIR", "").strip()
    if raw_global and slot_dir:
        try:
            gcap = int(raw_global)
        except ValueError:
            gcap = 0
        if gcap > 0:
            return FileSlotConcurrencyLimiter(
                local_limit=local_concurrency,
                global_limit=gcap,
                slot_dir=slot_dir,
            )
    return ConcurrencyLimiter(local_concurrency)
