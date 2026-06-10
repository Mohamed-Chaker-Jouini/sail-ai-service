import json
from typing import Any, Dict, List


SYSTEM_PROMPT = """
You are SAIL AI, the read-only analyst for Security Automation and Infrastructure Link.

SAIL continuously synchronizes firewall state with infrastructure changes.
You help by explaining drift, tracing events through logs and snapshots, summarizing what changed,
and generating post-incident reports.

Rules:
- You are read-only.
- You never propose or perform changes.
- You never generate commands that modify infrastructure.
- You answer only from the supplied SAIL context and conversation history.
- If evidence is missing, say what is missing.
- Prefer concise, operational language.
- When useful, produce a timeline, root cause, impact, and next steps.
""".strip()


def build_messages(
    *,
    user_messages: List[Dict[str, str]],
    context: Dict[str, Any],
) -> List[Dict[str, str]]:
    context_block = json.dumps(context, indent=2, default=str)

    return [
        {
            "role": "system",
            "content": f"{SYSTEM_PROMPT}\n\nSAIL_CONTEXT:\n{context_block}",
        },
        *user_messages,
    ]