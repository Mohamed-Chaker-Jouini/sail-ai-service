import os
from datetime import datetime, timezone
from typing import Any, Dict, List

import httpx

from prompting import build_messages


LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://127.0.0.1:8001")
LLM_MODEL = os.getenv("LLM_MODEL", "Qwen/Qwen3-32B")


def health_info() -> Dict[str, Any]:
    return {
        "status": "ok",
        "model": LLM_MODEL,
        "llm_base_url": LLM_BASE_URL,
        "time": datetime.now(timezone.utc).isoformat(),
    }


async def generate_reply(
    *,
    conversation_id: str | None,
    messages: List[Dict[str, str]],
    context: Dict[str, Any],
    temperature: float,
    max_output_tokens: int,
    top_p: float,
) -> Dict[str, Any]:
    prompt_messages = build_messages(user_messages=messages, context=context)

    payload = {
        "model": LLM_MODEL,
        "messages": prompt_messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_output_tokens,
        "stream": False,
    }

    url = f"{LLM_BASE_URL.rstrip('/')}/v1/chat/completions"

    async with httpx.AsyncClient(timeout=300) as client:
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()

    content = (
        data["choices"][0]["message"]["content"]
        if data.get("choices")
        else "No response returned by the model."
    )

    return {
        "conversation_id": conversation_id,
        "message": {
            "id": f"msg_{int(datetime.now(timezone.utc).timestamp() * 1000)}",
            "role": "assistant",
            "content": content,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
        "usage": data.get("usage", {}),
        "model": data.get("model", LLM_MODEL),
    }