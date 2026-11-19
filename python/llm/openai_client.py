"""OpenAI-compatible LLM client (works with gemini-web2api, magpie, and any /v1/chat/completions)."""
from __future__ import annotations

import ipaddress
import json
import os
import urllib.error
import urllib.parse
import urllib.request


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
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            obj = json.loads(text[start : end + 1])
        else:
            raise
    if not isinstance(obj, dict):
        raise ValueError(f"expected JSON object from LLM, got {type(obj).__name__}")
    return obj


class OpenAICompatLLMClient:
    """Environment: OPENAI_BASE_URL (default http://127.0.0.1:18083/v1),
    OPENAI_API_KEY / OPENAI_API_KEY_FILE. One LLM call per batch; items keep their order."""

    name = "openai-compat"

    def _resolve_key(self) -> str:
        key = os.environ.get("OPENAI_API_KEY", "").strip()
        if key:
            return key
        key_file = os.environ.get("OPENAI_API_KEY_FILE", "").strip()
        if key_file:
            with open(key_file, encoding="utf-8") as fh:
                return fh.read().strip()
        return ""

    def translate(self, items, target_lang, model, timeout_ms):
        base = os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:18083/v1").rstrip("/")
        key = self._resolve_key()
        lines = "\n".join(f"{i['block_id']}\t{i['text']}" for i in items)
        prompt = (
            f"Translate each numbered academic paper block to {target_lang}. "
            f"Rules: (1) Keep placeholders like ⟪F0⟫, citations like [1], URLs, and standard technical "
            f"acronyms (e.g., LLM, KV Cache, MoE, FFN, Prefill, Decode, LEO, ISL, TFLOPS, Gbps, DRAM, HBM) "
            f"accurate; (2) Use natural, concise academic {target_lang}; "
            f"(3) Reply with ONLY a valid JSON object mapping exact block_id -> translation.\n\n{lines}"
        )
        body = json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
        }).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        req = urllib.request.Request(
            f"{base}/chat/completions",
            data=body,
            headers=headers,
        )
        opener = (
            urllib.request.build_opener(urllib.request.ProxyHandler({}))
            if _is_tailnet_or_local(base)
            else urllib.request.build_opener()
        )
        try:
            with opener.open(req, timeout=timeout_ms / 1000) as resp:
                data = json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read()[:240].decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM HTTP {exc.code}: {detail}") from exc
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
            out.append({"block_id": bid, "status": "ok", "translation": value.strip(), "error": None})
        return out
