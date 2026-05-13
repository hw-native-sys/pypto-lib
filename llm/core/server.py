# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass

from .async_engine import AsyncLLMEngine, ServingConfig, TokenOutput
from .types import GenerateConfig

logger = logging.getLogger(__name__)

try:
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, StreamingResponse
    from pydantic import BaseModel, Field
except ImportError as e:
    raise ImportError(
        "Serving requires fastapi and pydantic. Install with: pip install fastapi uvicorn sse-starlette pydantic"
    ) from e


# --- Request/Response Models ---

class CompletionRequest(BaseModel):
    model: str = ""
    prompt: str = ""
    max_tokens: int = 256
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int | None = None
    stop: list[str] | None = None
    stream: bool = False


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = ""
    messages: list[ChatMessage]
    max_tokens: int = 256
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int | None = None
    stop: list[str] | None = None
    stream: bool = False


class CompletionChoice(BaseModel):
    index: int = 0
    text: str = ""
    finish_reason: str | None = None


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: list[CompletionChoice]


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatMessage | None = None
    delta: ChatMessage | None = None
    finish_reason: str | None = None


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]


# --- Server ---

class ServingServer:
    def __init__(self, async_engine: AsyncLLMEngine, model_id: str) -> None:
        self.engine = async_engine
        self.model_id = model_id
        self.app = FastAPI(title="PyPTO Serving")
        self._register_routes()

    def _register_routes(self) -> None:
        self.app.add_api_route("/health", self._health, methods=["GET"])
        self.app.add_api_route("/v1/models", self._list_models, methods=["GET"])
        self.app.add_api_route("/v1/completions", self._completions, methods=["POST"], response_model=None)
        self.app.add_api_route("/v1/chat/completions", self._chat_completions, methods=["POST"], response_model=None)

    async def _health(self) -> JSONResponse:
        return JSONResponse({"status": "ok"})

    async def _list_models(self) -> JSONResponse:
        return JSONResponse({
            "object": "list",
            "data": [{"id": self.model_id, "object": "model", "owned_by": "pypto"}],
        })

    async def _completions(self, request: CompletionRequest) -> StreamingResponse | JSONResponse:
        request_id = f"cmpl-{uuid.uuid4().hex[:8]}"
        config = GenerateConfig(
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            stop=tuple(request.stop) if request.stop else (),
            stream=request.stream,
        )

        if request.stream:
            return StreamingResponse(
                self._stream_completion(request_id, request.prompt, config, request.model or self.model_id),
                media_type="text/event-stream",
            )

        full_text = ""
        finish_reason = ""
        async for output in self.engine.add_request(request_id, request.prompt, config):
            if output.text:
                full_text = output.text
            if output.finished:
                finish_reason = self._map_finish_reason(output.finish_reason)

        response = CompletionResponse(
            id=request_id,
            created=int(time.time()),
            model=request.model or self.model_id,
            choices=[CompletionChoice(text=full_text, finish_reason=finish_reason)],
        )
        return JSONResponse(response.model_dump())

    async def _chat_completions(self, request: ChatCompletionRequest) -> StreamingResponse | JSONResponse:
        request_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        prompt = self._apply_chat_template(request.messages)
        config = GenerateConfig(
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            stop=tuple(request.stop) if request.stop else (),
            stream=request.stream,
        )

        if request.stream:
            return StreamingResponse(
                self._stream_chat_completion(request_id, prompt, config, request.model or self.model_id),
                media_type="text/event-stream",
            )

        full_text = ""
        finish_reason = ""
        async for output in self.engine.add_request(request_id, prompt, config):
            if output.text:
                full_text = output.text
            if output.finished:
                finish_reason = self._map_finish_reason(output.finish_reason)

        response = ChatCompletionResponse(
            id=request_id,
            object="chat.completion",
            created=int(time.time()),
            model=request.model or self.model_id,
            choices=[ChatCompletionChoice(
                message=ChatMessage(role="assistant", content=full_text),
                finish_reason=finish_reason,
            )],
        )
        return JSONResponse(response.model_dump())

    async def _stream_completion(
        self, request_id: str, prompt: str, config: GenerateConfig, model: str
    ):
        prev_text = ""
        async for output in self.engine.add_request(request_id, prompt, config):
            delta = output.text[len(prev_text):] if output.text else ""
            prev_text = output.text or prev_text
            finish_reason = self._map_finish_reason(output.finish_reason) if output.finished else None

            chunk = CompletionResponse(
                id=request_id,
                created=int(time.time()),
                model=model,
                choices=[CompletionChoice(text=delta, finish_reason=finish_reason)],
            )
            yield f"data: {json.dumps(chunk.model_dump())}\n\n"

            if output.finished:
                yield "data: [DONE]\n\n"
                break

    async def _stream_chat_completion(
        self, request_id: str, prompt: str, config: GenerateConfig, model: str
    ):
        prev_text = ""
        async for output in self.engine.add_request(request_id, prompt, config):
            delta = output.text[len(prev_text):] if output.text else ""
            prev_text = output.text or prev_text
            finish_reason = self._map_finish_reason(output.finish_reason) if output.finished else None

            chunk = ChatCompletionResponse(
                id=request_id,
                object="chat.completion.chunk",
                created=int(time.time()),
                model=model,
                choices=[ChatCompletionChoice(
                    delta=ChatMessage(role="assistant", content=delta),
                    finish_reason=finish_reason,
                )],
            )
            yield f"data: {json.dumps(chunk.model_dump())}\n\n"

            if output.finished:
                yield "data: [DONE]\n\n"
                break

    def _apply_chat_template(self, messages: list[ChatMessage]) -> str:
        """Simple chat template — can be replaced with tokenizer's chat_template."""
        parts = []
        for msg in messages:
            if msg.role == "system":
                parts.append(f"<|system|>\n{msg.content}")
            elif msg.role == "user":
                parts.append(f"<|user|>\n{msg.content}")
            elif msg.role == "assistant":
                parts.append(f"<|assistant|>\n{msg.content}")
        parts.append("<|assistant|>\n")
        return "\n".join(parts)

    @staticmethod
    def _map_finish_reason(reason: str) -> str:
        mapping = {
            "FINISHED_EOS": "stop",
            "FINISHED_LENGTH": "length",
            "FINISHED_STOP": "stop",
            "FINISHED_ABORTED": "stop",
        }
        return mapping.get(reason, "stop")


def create_serving_app(async_engine: AsyncLLMEngine, model_id: str) -> FastAPI:
    server = ServingServer(async_engine, model_id)
    return server.app
