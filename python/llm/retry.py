"""Error classification, jittered backoff, Retry-After parsing, and shared 429 cooldown."""
from __future__ import annotations

from dataclasses import dataclass
import random
import re
import socket
import threading
import time

_RETRY_AFTER_RE = re.compile(r"retry[-_ ]after\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)


@dataclass(frozen=True)
class RetryDecision:
    retryable: bool
    category: str  # rate_limit | transient_http | timeout | parse_error | fatal | other
    delay_s: float
    cooldown_s: float = 0.0


def classify_error(exc: Exception) -> tuple[str, float | None]:
    """Classify an LLM client exception and extract optional Retry-After seconds."""
    msg = str(exc)
    retry_after: float | None = None
    m = _RETRY_AFTER_RE.search(msg)
    if m:
        try:
            retry_after = float(m.group(1))
        except ValueError:
            retry_after = None

    if "HTTP 429" in msg or "429" in msg and "rate" in msg.lower():
        return "rate_limit", retry_after
    if any(code in msg for code in ("HTTP 401", "HTTP 403", "HTTP 400", "HTTP 404")):
        return "fatal", None
    if any(code in msg for code in ("HTTP 500", "HTTP 502", "HTTP 503", "HTTP 504")):
        return "transient_http", retry_after
    if (
        isinstance(exc, (TimeoutError, socket.timeout))
        or "timed out" in msg.lower()
        or "timeout" in msg.lower()
    ):
        return "timeout", None
    if "json" in msg.lower() or "length mismatch" in msg.lower():
        return "parse_error", None
    return "other", None


class RetryPolicy:
    """Jittered exponential backoff with shared 429 cooldown across sibling workers."""

    def __init__(
        self,
        max_retries: int = 2,
        rate_limit_base_s: float = 0.4,
        rate_limit_cap_s: float = 4.0,
        transient_base_s: float = 0.15,
        transient_cap_s: float = 2.0,
        other_base_s: float = 0.05,
        other_cap_s: float = 0.5,
        jitter: bool = True,
        rng: random.Random | None = None,
    ) -> None:
        self.max_retries = max(0, int(max_retries))
        self.rate_limit_base_s = rate_limit_base_s
        self.rate_limit_cap_s = rate_limit_cap_s
        self.transient_base_s = transient_base_s
        self.transient_cap_s = transient_cap_s
        self.other_base_s = other_base_s
        self.other_cap_s = other_cap_s
        self.jitter = jitter
        self._rng = rng or random.Random()
        self._cooldown_lock = threading.Lock()
        self._cooldown_until = 0.0

    def _apply_jitter(self, delay: float) -> float:
        if not self.jitter or delay <= 0:
            return round(delay, 4)
        # Equal jitter [0.5 * delay, delay] avoids thundering herd while keeping bounded wait
        return round(delay * (0.5 + 0.5 * self._rng.random()), 4)

    def decide(self, exc: Exception, attempt: int) -> RetryDecision:
        category, retry_after = classify_error(exc)
        if category == "fatal" or attempt >= self.max_retries:
            return RetryDecision(retryable=False, category=category, delay_s=0.0)

        if category == "rate_limit":
            base = min(self.rate_limit_base_s * (2**attempt), self.rate_limit_cap_s)
            if retry_after is not None and retry_after > 0:
                delay = min(retry_after, self.rate_limit_cap_s * 2)
                cooldown = min(retry_after, self.rate_limit_cap_s)
            else:
                delay = self._apply_jitter(base)
                cooldown = min(base * 0.6, self.rate_limit_cap_s)
            return RetryDecision(
                retryable=True, category=category, delay_s=delay, cooldown_s=cooldown
            )

        if category in ("transient_http", "timeout"):
            base = min(self.transient_base_s * (2**attempt), self.transient_cap_s)
            return RetryDecision(
                retryable=True, category=category, delay_s=self._apply_jitter(base)
            )

        base = min(self.other_base_s * (2**attempt), self.other_cap_s)
        return RetryDecision(
            retryable=True, category=category, delay_s=self._apply_jitter(base)
        )

    def note_cooldown(self, cooldown_s: float) -> None:
        if cooldown_s <= 0:
            return
        until = time.monotonic() + cooldown_s
        with self._cooldown_lock:
            if until > self._cooldown_until:
                self._cooldown_until = until

    def wait_if_cooling_down(self) -> float:
        with self._cooldown_lock:
            rem = self._cooldown_until - time.monotonic()
        if rem > 0:
            time.sleep(rem)
            return rem
        return 0.0
