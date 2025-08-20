#!/usr/bin/env python

import typing
import os
import json
from fastapi import FastAPI, Header, APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from openai import AsyncOpenAI

app = FastAPI()
router = APIRouter()

class ChatArgs(BaseModel):
    model: str
    messages: typing.List[typing.Dict[str, str]]

@router.post("/chat/completions")
async def groq_api(args: ChatArgs, authorization: str = Header(None)):
    # 1️⃣ 优先从 Header 取 key
    api_key = None
    if authorization:
        api_key = authorization.split(" ")[1]
    else:
        api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        return {"error": "API key not provided"}

    client = AsyncOpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=api_key
    )

    # 🚀 使用流式生成（包装成 OpenAI SSE 格式）
    async def event_generator():
        async with client.chat.completions.stream(
            model=args.model,
            messages=args.messages,
        ) as stream:
            async for event in stream:
                if event.type == "token":
                    data = {
                        "id": "chatcmpl-stream",
                        "object": "chat.completion.chunk",
                        "choices": [
                            {"delta": {"content": event.token}, "index": 0, "finish_reason": None}
                        ]
                    }
                    yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
            # 结束标记
            yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

# 注册路由
app.include_router(router, prefix="/v1")
