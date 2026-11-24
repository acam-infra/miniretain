"""OpenAI-compatible LLM client with Multi-Key, Multi-Endpoint, and IP/Proxy Pool rotation."""
from __future__ import annotations

import ipaddress
import json
import os
import re
import threading
import time
from typing import Any
import urllib.error
import urllib.parse
import urllib.request

from common.pool import (
    EndpointKeyProxyPool,
    PoolSlot,
    build_opener_for_proxy,
    load_lines_from_file_or_str,
)

_CJK_CHAR_RE = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")
_LATIN_WORD_RE = re.compile(r"[A-Za-z]{3,}")
_EXEMPT_ECHO_ROLES = frozenset({"reference", "formula", "figure_label", "header_footer", "author_affiliation", "table"})
_PROTECTED_FRAGMENT_RE = re.compile(
    r"⟪F\d+⟫"
    r"|https?://\S+"
    r"|\b[\w.+-]+@[\w-]+\.[\w.-]+\b"
    r"|arXiv:\S+"
    r"|§\s*\d+(?:\.\d+)*"
    r"|[∗†‡§]"
    r"|\[\d+(?:\s*[-–—,]\s*\d+)*\]",
    re.IGNORECASE,
)


def _strip_protected_fragments(text: str) -> str:
    cleaned = _PROTECTED_FRAGMENT_RE.sub(" ", text or "")
    return " ".join(cleaned.split())


def _is_verbatim_untranslated_echo(item: dict, translation: str, target_lang: str) -> bool:
    tl = (target_lang or "").lower()
    if not tl.startswith(("zh", "ja", "ko")):
        return False
    if item.get("layout_role") in _EXEMPT_ECHO_ROLES:
        return False
    src = (item.get("text") or "").strip()
    if len(src) < 40 or _CJK_CHAR_RE.search(translation):
        return False
    if re.search(r"[a-z0-9\-]+/[a-z0-9\-]+", src) and " " not in src.strip():
        return False
    if "←" in src or "USENIX Association" in src or "Symposium on Operating Systems" in src:
        return False
    if re.search(r"\b[a-z][a-z0-9]*_[a-z0-9_]+\b|\b[A-Za-z_]\w*\.[A-Za-z_]\w*", src) and re.search(r"\b(?:if|then|else|return|for|while)\b", src):
        return False
    if len(src) <= 220 and re.search(r"(?:^|\s)(?:from\s+\S+\s+)?import\s+\S+", src):
        return False
    if len(src) <= 160 and re.search(r"\b[a-z][a-z0-9]*_[a-z0-9_]+\b|\b[A-Za-z_]\w*\.[A-Za-z_]\w*\s*\(", src):
        return False
    if len(src) <= 240 and re.search(r"\b[a-z_]\w*\s*:\s*[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+|\"\"\"|\x27\x27\x27", src):
        return False
    if len(src) <= 200 and len(re.findall(r"\b[A-Z]\.", src)) >= 3:
        return False
    toks = src.split()
    if len(toks) >= 6 and sum(1 for t in toks if any(c.isdigit() for c in t)) / len(toks) >= 0.30:
        return False

    rem_src = _strip_protected_fragments(src)
    if len(rem_src) < 40 or len(_LATIN_WORD_RE.findall(rem_src)) < 6:
        return False
    rem_trans = _strip_protected_fragments(translation)
    return translation.strip() == src or rem_trans == rem_src


def _is_tailnet_or_local(url: str) -> bool:
    try:
        host = urllib.parse.urlparse(url).hostname or ""
    except Exception:
        return False
    if host in {"localhost", "127.0.0.1", "::1"}:
        return True
    try:
        ip = ipaddress.ip_address(host)
        return ip.is_loopback or ip.is_private or ip in ipaddress.ip_network("100.64.0.0/10")
    except ValueError:
        return False


def _extract_json_object(content: str) -> dict:
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    _latex_bfnrt_re = re.compile(
        r"\\(?="
        r"(?:text|textbf|textit|textrm|textsf|texttt|theta|vartheta|tau|times|to|top|tilde|triangle|tfrac|tag|tan|tanh|tr)"
        r"|(?:beta|bar|bf|boldsymbol|bm|big|Big|bigg|Bigg|binom|bot|bullet|boxed|begin|bmod|bowtie|bigcirc|bigcup|bigcap)"
        r"|(?:frac|forall|flat|frown|footnotesize|fbox)"
        r"|(?:nu|nabla|neq|ne|neg|ni|notin|not|nleq|ngeq|nsubseteq|norm|newline|nonumber|noindent|natural|nmid|nwarrow|nearrow)"
        r"|(?:rho|varrho|rightarrow|Rightarrow|rightleftharpoons|rangle|rceil|rfloor|rm|mathrm|mathcal|mathbf|mathbb|mathsf|mathtt|mathit|operatorname|ref|right|rule)"
        r"(?![A-Za-z])"
        r")"
    )

    def _repair_json_latex(s: str) -> str:
        out: list[str] = []
        i, n, in_str = 0, len(s), False
        while i < n:
            ch = s[i]
            if not in_str:
                out.append(ch)
                if ch == '"':
                    in_str = True
                i += 1
                continue
            if ch == '"':
                out.append(ch)
                in_str = False
                i += 1
                continue
            if ch != "\\":
                out.append(ch)
                i += 1
                continue
            if i + 1 >= n:
                out.append("\\\\")
                i += 1
                continue
            nxt = s[i + 1]
            if nxt == "\\":
                out.append("\\\\")
                i += 2
            elif nxt in ('"', "/"):
                out.append("\\" + nxt)
                i += 2
            elif nxt == "u":
                if i + 5 < n and all(c in "0123456789abcdefABCDEF" for c in s[i + 2 : i + 6]):
                    out.append(s[i : i + 6])
                    i += 6
                else:
                    out.append("\\\\")
                    i += 1
            elif nxt in ("b", "f", "n", "r", "t"):
                if _latex_bfnrt_re.match(s, i):
                    out.append("\\\\")
                    i += 1
                else:
                    out.append("\\" + nxt)
                    i += 2
            else:
                out.append("\\\\")
                i += 1
        return "".join(out)

    def _loads_lenient(s: str) -> Any:
        return json.loads(_repair_json_latex(s), strict=False)

    try:
        obj = _loads_lenient(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            obj = _loads_lenient(text[start : end + 1])
        else:
            raise
    if not isinstance(obj, dict):
        raise ValueError(f"expected JSON object from LLM, got {type(obj).__name__}")
    return obj


class OpenAICompatLLMClient:
    """OpenAI-compatible LLM client with multi-API-key, multi-endpoint, and IP/proxy pool rotation."""

    name = "openai-compat"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self._pool: EndpointKeyProxyPool | None = None
        self._pool_sig: tuple[Any, ...] | None = None
        self._pool_lock = threading.Lock()

    def _get_pool(self, requested_model: str) -> EndpointKeyProxyPool:
        with self._pool_lock:
            base = str(
                self.config.get("base_url")
                or os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:18083/v1")
            ).strip().rstrip("/")
            keys = load_lines_from_file_or_str(
                explicit_list=self.config.get("api_keys") if isinstance(self.config.get("api_keys"), list) else None,
                explicit_value=str(self.config.get("api_key") or ""),
                file_path=str(self.config.get("api_key_file") or os.environ.get("OPENAI_API_KEY_FILE", "")),
                env_var="OPENAI_API_KEY",
            )
            proxies = load_lines_from_file_or_str(
                explicit_list=self.config.get("proxies") if isinstance(self.config.get("proxies"), list) else None,
                explicit_value=str(self.config.get("proxy_pool") or ""),
                file_path=str(self.config.get("proxy_file") or os.environ.get("OPENAI_PROXY_FILE", "")),
                env_var="OPENAI_PROXY_POOL",
            )
            fallback_models = load_lines_from_file_or_str(
                explicit_list=self.config.get("fallback_models") if isinstance(self.config.get("fallback_models"), list) else None,
                env_var="OPENAI_FALLBACK_MODELS",
            )
            sig = (base, tuple(keys), tuple(proxies), tuple(fallback_models), requested_model)
            if self._pool is not None and self._pool_sig == sig:
                return self._pool
            endpoints: list[dict[str, Any]] | None = None
            if isinstance(self.config.get("endpoints"), list):
                endpoints = self.config["endpoints"]
            elif os.environ.get("OPENAI_POOL_JSON", "").strip():
                try:
                    endpoints = json.loads(os.environ["OPENAI_POOL_JSON"])
                except Exception:
                    endpoints = None
            elif os.environ.get("OPENAI_POOL_FILE", "").strip():
                try:
                    with open(os.environ["OPENAI_POOL_FILE"], encoding="utf-8") as fh:
                        endpoints = json.loads(fh.read())
                except Exception:
                    endpoints = None

            self._pool = EndpointKeyProxyPool.from_config(
                default_base_url=base,
                default_model=requested_model,
                api_keys=keys,
                proxies=proxies,
                fallback_models=fallback_models,
                endpoints=endpoints,
                per_slot_concurrency=int(self.config.get("concurrency") or 8),
            )
            self._pool_sig = sig
            return self._pool

    def pool_snapshot(self) -> dict[str, Any]:
        if self._pool is None:
            return {}
        return self._pool.snapshot()

    def translate(self, items, target_lang, model, timeout_ms):
        pool = self._get_pool(model)
        doc_context = ""
        if items and isinstance(items[0], dict):
            doc_context = (items[0].get("_doc_context") or "").strip()
        lines = "\n".join(f"{i['block_id']}\t{i['text']}" for i in items)
        ctx_section = f"\nDocument terminology & style context:\n{doc_context}\n" if doc_context else ""
        prompt = (
            f"Translate each numbered academic paper block to {target_lang}. "
            f"Rules: (1) Keep placeholders like ⟪F0⟫, inline/display math ($...$, $$...$$), citations like [1], URLs, numbers/units, and standard technical "
            f"acronyms (e.g., LLM, KV Cache, MoE, FFN, Prefill, Decode, LEO, ISL, TFLOPS, Gbps, DRAM, HBM) "
            f"verbatim and accurate (never drop subscripts/superscripts or emit raw LaTeX commands outside $...$); "
            f"(2) Preserve any inline Markdown bold **...** and italic *...* markers around the corresponding "
            f"translated terms or paragraph lead-in headings (e.g., '**Batching.**' -> '**批处理。**', '*continuous batching*' -> '*连续批处理*'); "
            f"(3) Use natural, concise academic {target_lang}; "
            f"(4) Reply with ONLY a valid JSON object mapping every exact block_id -> translation (do not omit any block_id)."
            f"{ctx_section}\n{lines}"
        )

        max_slot_attempts = min(max(1, len(pool.slots)), 3)
        tried_slots: set[str] = set()
        last_exc: Exception | None = None
        data: dict[str, Any] | None = None

        for _attempt_idx in range(max_slot_attempts):
            slot = pool.acquire(exclude_slot_ids=tried_slots)
            tried_slots.add(slot.slot_id)
            t0 = time.perf_counter()
            use_model = slot.model or model
            body = json.dumps({
                "model": use_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
            }).encode("utf-8")
            headers = {"Content-Type": "application/json"}
            if slot.api_key:
                headers["Authorization"] = f"Bearer {slot.api_key}"
            req = urllib.request.Request(
                f"{slot.base_url}/chat/completions",
                data=body,
                headers=headers,
            )
            if slot.proxy and slot.proxy.lower() not in ("direct", "none", "noproxy"):
                opener = build_opener_for_proxy(slot.proxy)
            elif _is_tailnet_or_local(slot.base_url):
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            else:
                opener = urllib.request.build_opener()

            try:
                with opener.open(req, timeout=timeout_ms / 1000) as resp:
                    data = json.loads(resp.read())
                pool.release_success(slot, time.perf_counter() - t0)
                break
            except urllib.error.HTTPError as exc:
                detail = exc.read()[:240].decode("utf-8", errors="replace")
                retry_after = exc.headers.get("Retry-After") if exc.headers else None
                ra_note = f" retry_after={retry_after}" if retry_after else ""
                is_rl = exc.code == 429
                cooldown = 15.0 if is_rl else (60.0 if exc.code in (401, 403) else 5.0)
                if retry_after:
                    try:
                        cooldown = max(cooldown, float(retry_after))
                    except ValueError:
                        pass
                err_str = f"LLM HTTP {exc.code}{ra_note} (slot={slot.slot_id}): {detail}"
                pool.release_failure(slot, is_rate_limit=is_rl, cooldown_s=cooldown, error_msg=err_str)
                last_exc = RuntimeError(err_str)
                if exc.code in (401, 403, 400) and len(pool.slots) <= 1:
                    raise last_exc from exc
            except Exception as exc:  # noqa: BLE001
                err_str = f"LLM error (slot={slot.slot_id}): {exc}"
                pool.release_failure(slot, is_rate_limit=False, cooldown_s=5.0, error_msg=err_str)
                last_exc = exc

        if data is None:
            if last_exc is not None:
                raise last_exc
            raise RuntimeError("LLM pool exhausted without response")

        content = data["choices"][0]["message"]["content"]
        mapping = _extract_json_object(content)
        out = []
        for item in items:
            bid = item["block_id"]
            if bid not in mapping:
                out.append({"block_id": bid, "status": "missing", "translation": None, "error": f"missing key {bid}"})
                continue
            value = mapping[bid]
            if not isinstance(value, str) or not value.strip():
                out.append({"block_id": bid, "status": "empty", "translation": None, "error": f"empty translation for {bid}"})
                continue
            cleaned = value.strip()
            src_raw = re.sub(r"\*{1,2}", "", (item.get("text") or "").strip())
            cleaned_nomark = re.sub(r"\*{1,2}", "", cleaned)
            effective_len = len(re.sub(r"\$\$?[^$\n]+\$\$?|\\\([^)]+\\\)|\\\[[^\]]+\\\]", "X", cleaned_nomark))
            if len(items) > 1 and len(src_raw) >= 15 and "⟪F" not in src_raw and effective_len > max(32, int(len(src_raw) * 1.25)):
                out.append({"block_id": bid, "status": "empty", "translation": None, "error": f"suspiciously long translation for {bid}"})
                continue
            expected_toks = set(re.findall(r"⟪F\d+⟫", src_raw))
            actual_toks = set(re.findall(r"⟪F\d+⟫", cleaned))
            if actual_toks - expected_toks:
                out.append({"block_id": bid, "status": "empty", "translation": None, "error": f"foreign formula tokens {sorted(actual_toks - expected_toks)} for {bid}"})
                continue
            if _is_verbatim_untranslated_echo(item, cleaned, target_lang):
                out.append({"block_id": bid, "status": "empty", "translation": None, "error": f"untranslated verbatim echo for {bid}"})
                continue
            out.append({"block_id": bid, "status": "ok", "translation": cleaned, "error": None})
        return out
