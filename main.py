from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Literal, Optional

from llm import generate_reply, health_info

app = FastAPI(title="SAIL AI Service", docs_url="/docs")

Role = Literal["system", "user", "assistant"]


class ChatMessage(BaseModel):
    role: Role
    content: str


class ChatRequest(BaseModel):
    conversation_id: Optional[str] = None
    messages: List[ChatMessage]
    stream: bool = False
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_output_tokens: int = Field(default=600, ge=1, le=8192)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    context: Dict[str, Any] = {}


@app.get("/api/ai/health")
def health():
    return health_info()


@app.post("/api/ai/chat")
async def chat(req: ChatRequest):
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages is required")

    return await generate_reply(
        conversation_id=req.conversation_id,
        messages=[m.model_dump() for m in req.messages],
        context=req.context,
        temperature=req.temperature,
        max_output_tokens=req.max_output_tokens,
        top_p=req.top_p,
    )