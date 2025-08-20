#!/usr/bin/env python
import typing
from fastapi import FastAPI, Header, APIRouter
from pydantic import BaseModel
from openai import AsyncOpenAI

app = FastAPI()
router = APIRouter()

class ChatArgs(BaseModel):
    model: str
    messages: typing.List[typing.Dict[str, str]]

@router.post("/chat/completions")
async def groq_api(args: ChatArgs, authorization: str = Header(...)):
    api_key = authorization.split(" ")[1]
    client = AsyncOpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=api_key
    )
    return await client.chat.completions.create(
        model=args.model,
        messages=args.messages,
        max_tokens=1000  # 👈 建议加上，避免输出过短
    )

# 必须挂载 router
app.include_router(router, prefix="/v1")
