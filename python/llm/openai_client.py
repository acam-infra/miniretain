"""OpenAI-compatible LLM client (works with gemini-web2api and any /v1/chat/completions)."""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request


class OpenAICompatLLMClient:
    """Environment: OPENAI_BASE_URL (default http://127.0.0.1:8083/v1),
    OPENAI_API_KEY. One LLM call per batch; items keep their order."""

    name = "openai-compat"

    def translate(self, items, target_lang, model, timeout_ms):
        base = os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:18083/v1").rstrip("/")
        key_file = os.environ.get("OPENAI_API_KEY_FILE", "")
        key = os.environ.get("OPENAI_API_KEY", "")
        if key_file and not key:
            key = open(key_file).read().strip()
        lines = "\n".join(f"{i['block_id']}\t{i['text']}" for i in items)
        prompt = (
            f"Translate each line to {target_lang}. Keep inline formula tokens "
            f"and placeholders unchanged. Reply with ONLY a JSON object mapping "
            f"block_id -> translation, same keys as input.\n\n{lines}"
        )
        body = json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
        }).encode()
        req = urllib.request.Request(
            f"{base}/chat/completions",
            data=body,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout_ms / 1000) as resp:
                data = json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"LLM HTTP {exc.code}: {exc.read()[:200]}") from exc
        content = data["choices"][0]["message"]["content"]
        content = content.strip()
        if content.startswith("```"):
            content = content.strip("`").lstrip("json").strip()
        mapping = json.loads(content)
        out = []
        for item in items:
            value = mapping.get(item["block_id"])
            out.append(value if isinstance(value, str) else f"<missing translation for {item['block_id']}>")
        return out
