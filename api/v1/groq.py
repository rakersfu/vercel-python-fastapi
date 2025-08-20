#!/usr/bin/env python

import typing
from fastapi import Header, APIRouter
from pydantic import BaseModel
from openai import AsyncOpenAI

router = APIRouter()

class ChatArgs(BaseModel):
    model: str
    messages: typing.List[typing.Dict[str, str]]

@router.post("/chat/completions")
async def groq_api(args: ChatArgs, authorization: str = Header(...)):
    api_key = authorization.split(" ")[1]  # 取出 Bearer Token
    client = AsyncOpenAI(base_url="https://api.groq.com/openai", api_key=api_key)

    response = await client.chat.completions.create(
        model=args.model,
        messages=args.messages,
    )
    return response
