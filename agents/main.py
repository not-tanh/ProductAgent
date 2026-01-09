import json
import os
import sys
import traceback
import uuid
from contextlib import asynccontextmanager
from typing import Optional

import redis.asyncio as redis
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    messages_to_dict,
    messages_from_dict,
)
from langfuse import get_client, propagate_attributes
from langfuse.langchain import CallbackHandler
from pydantic import BaseModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv()
from agents import simple_agent

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", "86400"))
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "10"))
LOCK_TTL_SECONDS = int(os.getenv("SESSION_LOCK_TTL_SECONDS", "10"))

SESSION_KEY_PREFIX = os.getenv("SESSION_KEY_PREFIX", "chat:session:")
LOCK_KEY_PREFIX = os.getenv("LOCK_KEY_PREFIX", "chat:lock:")


@asynccontextmanager
async def lifespan(app_: FastAPI):
    app_.state.langfuse = get_client()
    app_.state.redis = redis.from_url(REDIS_URL, decode_responses=True)
    try:
        await app_.state.redis.ping()
        yield
    finally:
        await app_.state.redis.close()
        app_.state.langfuse.shutdown()


app = FastAPI(lifespan=lifespan)


def _session_key(session_id: str) -> str:
    return f"{SESSION_KEY_PREFIX}{session_id}"


def _lock_key(session_id: str) -> str:
    return f"{LOCK_KEY_PREFIX}{session_id}"


async def load_history(session_id: str) -> list[BaseMessage]:
    redis_client = app.state.redis

    assert redis_client is not None
    raw = await redis_client.get(_session_key(session_id))
    if not raw:
        return []
    try:
        return messages_from_dict(json.loads(raw))
    except Exception:
        traceback.print_exc()
        # Corrupt session state: treat as empty
        return []


async def save_history(session_id: str, messages: list[BaseMessage]) -> None:
    redis_client = app.state.redis

    assert redis_client is not None

    payload = json.dumps(messages_to_dict(messages), ensure_ascii=False)
    await redis_client.set(_session_key(session_id), payload, ex=SESSION_TTL_SECONDS)


def extract_last_ai_text(messages: list[BaseMessage]) -> str:
    # Find the last AI message (do not assume last element is AI)
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            return msg.content
    return ""


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    session_id: str
    reply: str


@app.post("/chat", response_model=ChatResponse)
async def chat(
    body: ChatRequest,
    x_session_id: Optional[str] = Header(default=None),
):
    redis_client = app.state.redis
    assert redis_client is not None

    session_id = x_session_id or str(uuid.uuid4())

    # Per-session lock to prevent concurrent requests overwriting history
    lock = redis_client.lock(
        _lock_key(session_id),
        timeout=LOCK_TTL_SECONDS,
        blocking_timeout=LOCK_TTL_SECONDS,
    )

    if not await lock.acquire():
        raise HTTPException(status_code=409, detail="Session is busy. Retry.")

    langfuse = app.state.langfuse

    try:
        history = await load_history(session_id)
        history.append(HumanMessage(content=body.message))

        with langfuse.start_as_current_observation(
            as_type="span",
            name="chat_request",
            input={"message": body.message},
        ) as root_span:
            with propagate_attributes(
                session_id=session_id,
                user_id=session_id,
                tags=["product-agent", "fastapi"],
                metadata={"endpoint": "/chat"},
            ):
                handler = CallbackHandler()

                # Trim to last N messages to control context and cost
                result = await simple_agent.ainvoke(
                    {"messages": history[-MAX_HISTORY_MESSAGES:]}, config={"callbacks": [handler]})

                reply = result["messages"][-1].content
                root_span.update(output={"reply": reply})

        new_history = result.get("messages")

        # print('Len:', len(new_history))
        # for h in new_history:
        #     print(json.dumps(h, indent=4))
        #     print('=' * 50)
        # chat_log_only = [m for m in new_history if isinstance(m, (HumanMessage, AIMessage))]

        # Persist full message trace (or filter to only Human/AI if preferred)
        await save_history(session_id, new_history)

        reply = extract_last_ai_text(new_history)
        return ChatResponse(session_id=session_id, reply=reply)

    finally:
        try:
            await lock.release()
        except Exception:
            traceback.print_exc()
            pass
