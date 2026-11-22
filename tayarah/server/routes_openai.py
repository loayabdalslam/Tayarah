import json
import time
import uuid
from typing import List, Union
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse

from tayarah.server.schemas import (
    OpenAIChatCompletionRequest,
    OpenAICompletionRequest,
    OpenAIModelList,
    OpenAIModelItem
)
from tayarah.engine_manager import EngineManager

router = APIRouter(prefix="/v1", tags=["OpenAI / vLLM Compatible API"])

def format_openai_chat_prompt(messages) -> str:
    lines = []
    for m in messages:
        role = m.role.capitalize()
        content = m.content
        lines.append(f"{role}: {content}")
    lines.append("Assistant:")
    return "\n".join(lines)

@router.get("/models")
async def list_models():
    manager = EngineManager()
    model_list = manager.list_available_models()
    created_ts = int(time.time())
    
    data = []
    for m in model_list:
        data.append(OpenAIModelItem(
            id=m["id"],
            created=created_ts,
            owned_by="tayarah",
            root=m.get("family", "tayarah")
        ))
    return OpenAIModelList(data=data)

@router.get("/models/{model_id}")
async def get_model(model_id: str):
    manager = EngineManager()
    model_list = manager.list_available_models()
    for m in model_list:
        if m["id"] == model_id:
            return OpenAIModelItem(
                id=m["id"],
                created=int(time.time()),
                owned_by="tayarah",
                root=m.get("family", "tayarah")
            )
    raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")

@router.post("/chat/completions")
async def chat_completions(req: OpenAIChatCompletionRequest):
    manager = EngineManager()
    prompt = format_openai_chat_prompt(req.messages)
    req_id = f"chatcmpl-tayarah-{uuid.uuid4().hex[:12]}"
    created_ts = int(time.time())
    model_name = req.model

    if req.stream:
        def sse_chat_generator():
            # Initial role packet
            initial_packet = {
                "id": req_id,
                "object": "chat.completion.chunk",
                "created": created_ts,
                "model": model_name,
                "choices": [{
                    "index": 0,
                    "delta": {"role": "assistant"},
                    "finish_reason": None
                }]
            }
            yield f"data: {json.dumps(initial_packet)}\n\n"

            for chunk, is_done, meta in manager.stream_generate(
                prompt=prompt,
                max_tokens=req.max_tokens or 256,
                temperature=req.temperature or 0.4,
                top_p=req.top_p or 0.9,
                model_name=model_name
            ):
                if not is_done and chunk:
                    chunk_packet = {
                        "id": req_id,
                        "object": "chat.completion.chunk",
                        "created": created_ts,
                        "model": model_name,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": chunk},
                            "finish_reason": None
                        }]
                    }
                    yield f"data: {json.dumps(chunk_packet)}\n\n"
                elif is_done:
                    stop_packet = {
                        "id": req_id,
                        "object": "chat.completion.chunk",
                        "created": created_ts,
                        "model": model_name,
                        "choices": [{
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop"
                        }],
                        "usage": {
                            "prompt_tokens": 15,
                            "completion_tokens": meta.get("eval_count", 0),
                            "total_tokens": 15 + meta.get("eval_count", 0)
                        }
                    }
                    yield f"data: {json.dumps(stop_packet)}\n\n"
                    yield "data: [DONE]\n\n"

        return StreamingResponse(sse_chat_generator(), media_type="text/event-stream")
    else:
        result = manager.complete_generate(
            prompt=prompt,
            max_tokens=req.max_tokens or 256,
            temperature=req.temperature or 0.4,
            top_p=req.top_p or 0.9,
            model_name=model_name
        )
        meta = result["meta"]
        tok_count = meta.get("eval_count", 0)
        return {
            "id": req_id,
            "object": "chat.completion",
            "created": created_ts,
            "model": model_name,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": result["text"]
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 15,
                "completion_tokens": tok_count,
                "total_tokens": 15 + tok_count
            }
        }

@router.post("/completions")
async def completions(req: OpenAICompletionRequest):
    manager = EngineManager()
    prompt = req.prompt if isinstance(req.prompt, str) else "\n".join(req.prompt)
    req_id = f"cmpl-tayarah-{uuid.uuid4().hex[:12]}"
    created_ts = int(time.time())
    model_name = req.model

    if req.stream:
        def sse_completion_generator():
            for chunk, is_done, meta in manager.stream_generate(
                prompt=prompt,
                max_tokens=req.max_tokens or 256,
                temperature=req.temperature or 0.4,
                top_p=req.top_p or 0.9,
                model_name=model_name
            ):
                if not is_done and chunk:
                    chunk_packet = {
                        "id": req_id,
                        "object": "text_completion",
                        "created": created_ts,
                        "model": model_name,
                        "choices": [{
                            "text": chunk,
                            "index": 0,
                            "logprobs": None,
                            "finish_reason": None
                        }]
                    }
                    yield f"data: {json.dumps(chunk_packet)}\n\n"
                elif is_done:
                    stop_packet = {
                        "id": req_id,
                        "object": "text_completion",
                        "created": created_ts,
                        "model": model_name,
                        "choices": [{
                            "text": "",
                            "index": 0,
                            "logprobs": None,
                            "finish_reason": "stop"
                        }]
                    }
                    yield f"data: {json.dumps(stop_packet)}\n\n"
                    yield "data: [DONE]\n\n"

        return StreamingResponse(sse_completion_generator(), media_type="text/event-stream")
    else:
        result = manager.complete_generate(
            prompt=prompt,
            max_tokens=req.max_tokens or 256,
            temperature=req.temperature or 0.4,
            top_p=req.top_p or 0.9,
            model_name=model_name
        )
        meta = result["meta"]
        tok_count = meta.get("eval_count", 0)
        return {
            "id": req_id,
            "object": "text_completion",
            "created": created_ts,
            "model": model_name,
            "choices": [{
                "text": result["text"],
                "index": 0,
                "logprobs": None,
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": tok_count,
                "total_tokens": 10 + tok_count
            }
        }
