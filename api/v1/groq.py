#!/usr/bin/env python

import typing
from fastapi import FastAPI, Header, APIRouter
from pydantic import BaseModel
from openai import AsyncOpenAI
import os

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
        # 2️⃣ 如果 Header 没有，就尝试从环境变量取
        api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        return {"error": "API key not provided"}

    client = AsyncOpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=api_key
    )

    response = await client.chat.completions.create(
        model=args.model,
        messages=args.messages,
    )
    return response

# ⚠️ 一定要 include_router，否则访问会报 Not Found
app.include_router(router, prefix="/v1")
