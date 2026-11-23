"""Universal Multi-API-Key, Multi-Endpoint, and IP/Proxy Pool Scheduler for Miniretain.

Supports:
- Multi-key rotation (from multi-line key files, comma-separated env vars, or YAML lists)
- IP / Proxy pool rotation ("direct" or HTTP/HTTPS/SOCKS proxies from YAML or proxy_file)
- Multi-endpoint / multi-model hybrid failover (e.g. primary cc gateway + backup endpoints/models)
- Thread-safe least-inflight + round-robin slot selection with automatic 429 / timeout cooldown
- Hybrid OCR provider failover (e.g. mineru <-> paddle-cloud -> paddle-http)
"""
from __future__ import annotations

import itertools
import os
import threading
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def load_lines_from_file_or_str(
    *,
    explicit_list: list[str] | None = None,
    explicit_value: str = "",
    file_path: str = "",
    env_var: str = "",
) -> list[str]:
    out: list[str] = []
    if explicit_list:
        for item in explicit_list:
            s = str(item or "").strip()
            if s and not s.startswith("#"):
                out.append(s)
    if explicit_value:
        for part in str(explicit_value).replace("\n", ",").split(","):
            s = part.strip()
            if s and not s.startswith("#"):
                out.append(s)
    if env_var and os.environ.get(env_var, "").strip():
        for part in os.environ[env_var].replace("\n", ",").split(","):
            s = part.strip()
            if s and not s.startswith("#"):
                out.append(s)
    if file_path:
        p = Path(file_path).expanduser()
        if p.is_file():
            for line in p.read_text(encoding="utf-8").splitlines():
                s = line.strip()
                if s and not s.startswith("#"):
                    out.append(s)
    # Deduplicate while preserving order
    seen: set[str] = set()
    deduped: list[str] = []
    for item in out:
        if item not in seen:
            seen.add(item)
            deduped.append(item)
    return deduped


def mask_key(key: str) -> str:
    if not key:
        return "none"
    if len(key) <= 8:
        return "***"
    return f"{key[:4]}...{key[-4:]}"


def build_opener_for_proxy(proxy_spec: str) -> urllib.request.OpenerDirector:
    """Build a urllib OpenerDirector for 'direct' (no proxy) or an explicit proxy URL."""
    spec = (proxy_spec or "direct").strip()
    if not spec or spec.lower() in ("direct", "none", "noproxy"):
        return urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return urllib.request.build_opener(
        urllib.request.ProxyHandler({"http": spec, "https": spec})
    )


@dataclass
class PoolSlot:
    slot_id: str
    base_url: str
    api_key: str
    model: str
    proxy: str = "direct"
    max_concurrency: int = 8
    inflight: int = 0
    total_requests: int = 0
    succeeded_requests: int = 0
    rate_limit_429: int = 0
    errors: int = 0
    cooldown_until: float = 0.0
    last_error: str = ""
    total_latency_s: float = 0.0

    def is_available(self, now: float) -> bool:
        return now >= self.cooldown_until and self.inflight < max(1, self.max_concurrency)

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "slot_id": self.slot_id,
            "base_url": self.base_url,
            "key_masked": mask_key(self.api_key),
            "model": self.model,
            "proxy": self.proxy or "direct",
            "total_requests": self.total_requests,
            "succeeded_requests": self.succeeded_requests,
            "rate_limit_429": self.rate_limit_429,
            "errors": self.errors,
            "avg_latency_s": round(self.total_latency_s / self.succeeded_requests, 3)
            if self.succeeded_requests > 0
            else 0.0,
            "in_cooldown": time.monotonic() < self.cooldown_until,
        }


class EndpointKeyProxyPool:
    """Thread-safe pool combining multiple endpoints, API keys, models, and egress proxies/IPs."""

    def __init__(self, slots: list[PoolSlot]) -> None:
        if not slots:
            raise ValueError("EndpointKeyProxyPool requires at least one slot")
        self.slots = slots
        self._lock = threading.Lock()
        self._rr_counter = itertools.count()
        self.rotations_count = 0

    @classmethod
    def from_config(
        cls,
        *,
        default_base_url: str,
        default_model: str,
        api_keys: list[str],
        proxies: list[str] | None = None,
        fallback_models: list[str] | None = None,
        endpoints: list[dict[str, Any]] | None = None,
        per_slot_concurrency: int = 8,
    ) -> "EndpointKeyProxyPool":
        proxy_list = [p.strip() for p in (proxies or ["direct"]) if str(p).strip()] or ["direct"]
        keys_list = [k.strip() for k in api_keys if str(k).strip()] or [""]
        models_list = [default_model] + [
            m.strip() for m in (fallback_models or []) if str(m).strip() and str(m).strip() != default_model
        ]

        slots: list[PoolSlot] = []
        if endpoints and isinstance(endpoints, list):
            for ep_idx, ep in enumerate(endpoints):
                if not isinstance(ep, dict):
                    continue
                burl = str(ep.get("base_url") or default_base_url).strip().rstrip("/")
                ep_keys = load_lines_from_file_or_str(
                    explicit_list=ep.get("api_keys") if isinstance(ep.get("api_keys"), list) else None,
                    explicit_value=str(ep.get("api_key") or ""),
                    file_path=str(ep.get("api_key_file") or ""),
                    env_var=str(ep.get("api_key_env") or ""),
                ) or keys_list
                ep_proxies = load_lines_from_file_or_str(
                    explicit_list=ep.get("proxies") if isinstance(ep.get("proxies"), list) else None,
                    explicit_value=str(ep.get("proxy") or ""),
                    file_path=str(ep.get("proxy_file") or ""),
                ) or proxy_list
                ep_models = (
                    [str(m).strip() for m in ep.get("models", []) if str(m).strip()]
                    if isinstance(ep.get("models"), list)
                    else [str(ep.get("model") or default_model).strip()]
                )
                ep_conc = int(ep.get("max_concurrency") or per_slot_concurrency)
                for m_idx, mod in enumerate(ep_models):
                    for k_idx, key in enumerate(ep_keys):
                        for p_idx, prx in enumerate(ep_proxies):
                            slots.append(
                                PoolSlot(
                                    slot_id=f"ep{ep_idx}-m{m_idx}-k{k_idx}-ip{p_idx}",
                                    base_url=burl,
                                    api_key=key,
                                    model=mod,
                                    proxy=prx,
                                    max_concurrency=ep_conc,
                                )
                            )

        if not slots:
            for m_idx, mod in enumerate(models_list):
                for k_idx, key in enumerate(keys_list):
                    for p_idx, prx in enumerate(proxy_list):
                        slots.append(
                            PoolSlot(
                                slot_id=f"m{m_idx}-k{k_idx}-ip{p_idx}",
                                base_url=default_base_url.rstrip("/"),
                                api_key=key,
                                model=mod,
                                proxy=prx,
                                max_concurrency=per_slot_concurrency,
                            )
                        )
        return cls(slots)

    def acquire(self, exclude_slot_ids: set[str] | None = None) -> PoolSlot:
        """Select the best healthy slot (least in-flight ratio + round-robin tie-breaker)."""
        with self._lock:
            now = time.monotonic()
            excluded = exclude_slot_ids or set()
            candidates = [s for s in self.slots if s.slot_id not in excluded and now >= s.cooldown_until]
            if not candidates:
                # If all excluded or in cooldown, pick the one whose cooldown expires earliest
                pool_subset = [s for s in self.slots if s.slot_id not in excluded] or self.slots
                best = min(pool_subset, key=lambda s: (max(0.0, s.cooldown_until - now), s.inflight))
            else:
                rr = next(self._rr_counter)
                best = min(
                    candidates,
                    key=lambda s: (
                        s.inflight / max(1, s.max_concurrency),
                        s.rate_limit_429 + s.errors,
                        (self.slots.index(s) + rr) % len(self.slots),
                    ),
                )
            best.inflight += 1
            best.total_requests += 1
            return best

    def release_success(self, slot: PoolSlot, latency_s: float) -> None:
        with self._lock:
            slot.inflight = max(0, slot.inflight - 1)
            slot.succeeded_requests += 1
            slot.total_latency_s += max(0.0, latency_s)

    def release_failure(
        self,
        slot: PoolSlot,
        *,
        is_rate_limit: bool = False,
        cooldown_s: float = 8.0,
        error_msg: str = "",
    ) -> None:
        with self._lock:
            slot.inflight = max(0, slot.inflight - 1)
            if is_rate_limit:
                slot.rate_limit_429 += 1
            else:
                slot.errors += 1
            slot.last_error = error_msg[:200]
            if cooldown_s > 0:
                slot.cooldown_until = max(slot.cooldown_until, time.monotonic() + cooldown_s)
            self.rotations_count += 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "total_slots": len(self.slots),
                "rotations_count": self.rotations_count,
                "slots": [s.to_public_dict() for s in self.slots],
            }
