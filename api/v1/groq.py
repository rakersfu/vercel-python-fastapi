#!/usr/bin/env python

import os
import json
from fastapi import FastAPI, Header, APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from openai import AsyncOpenAI
import typing

app = FastAPI()
router = APIRouter()

class ChatArgs(BaseModel):
    model: str
    messages: typing.List[typing.Dict[str, str]]

@router.post("/chat/completions")
async def groq_api(args: ChatArgs, authorization: str = Header(None)):
    api_key = authorization.split(" ")[1] if authorization else os.getenv("GROQ_API_KEY")
    if not api_key:
        return {"error": "API key not provided"}

    client = AsyncOpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=api_key
    )

    async def event_generator():
        async with client.chat.completions.stream(
            model=args.model,
            messages=args.messages,
        ) as stream:
            async for event in stream:
                # ⚠️ 直接转发全部事件
                data = {"type": event.type, "data": event.dict()}
                yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

            yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

app.include_router(router, prefix="/v1")
