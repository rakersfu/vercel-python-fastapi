#!/usr/bin/env python
import typing
import pydantic
from fastapi import Header
from fastapi.routing import APIRouter
from openai import AsyncClient

router = APIRouter()

class ChatArgs(pydantic.BaseModel):
    model: str
    messages: typing.List[typing.Dict[str, str]]

@router.post("/chat/completions")
async def groq_api(args: ChatArgs, authorization: str = Header(...)):
    # 提取 API Key
    api_key = authorization.split(" ")[1]

    # 初始化 Groq 的客户端
    client = AsyncClient(
        base_url="https://api.groq.com/openai/v1",
        api_key=api_key
    )

    # 调用 Groq API
    response = await client.chat.completions.create(
        model=args.model,
        messages=args.messages,
    )

    # 提取主要内容返回
    return {
        "id": response.id,
        "model": response.model,
        "choices": response.choices,
        "usage": response.usage
    }
