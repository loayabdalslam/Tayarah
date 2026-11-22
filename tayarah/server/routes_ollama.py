import json
import time
from datetime import datetime, timezone
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse, JSONResponse

from tayarah.server.schemas import (
    OllamaGenerateRequest,
    OllamaChatRequest,
)
from tayarah.engine_manager import EngineManager

router = APIRouter(prefix="/api", tags=["Ollama API"])

def get_iso_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()

def build_prompt_from_messages(messages) -> str:
    parts = []
    for m in messages:
        role = m.role.capitalize() if hasattr(m, "role") else m.get("role", "User").capitalize()
        content = m.content if hasattr(m, "content") else m.get("content", "")
        parts.append(f"{role}: {content}")
    parts.append("Assistant:")
    return "\n".join(parts)

@router.get("/version")
async def ollama_version():
    return {"version": "0.5.4-tayarah-mach"}

@router.get("/tags")
async def ollama_tags():
    manager = EngineManager()
    models = manager.list_available_models()
    now_str = get_iso_timestamp()
    
    tag_list = []
    for m in models:
        tag_list.append({
            "name": m["id"],
            "model": f"{m['id']}:latest",
            "modified_at": now_str,
            "size": 4970000000 if "cpu" in m["id"] else 980000000,
            "digest": f"sha256:tayarah_{m['id']}_mach_hash",
            "details": {
                "parent_model": "",
                "format": "symbolic",
                "family": m["family"],
                "families": [m["family"], "the1"],
                "parameter_size": "7.5B" if "cpu" in m["id"] else "0.5B",
                "quantization_level": "Q4_K_M" if "cpu" in m["id"] else "FP16"
            }
        })
    return {"models": tag_list}

@router.post("/show")
async def ollama_show(request: Request):
    body = await request.json()
    model_name = body.get("name") or body.get("model", "tayarah")
    return {
        "modelfile": f"# Tayarah Mach-Speed Speculative Model: {model_name}\nFROM {model_name}\nPARAMETER temperature 0.7",
        "parameters": "temperature 0.7\ntop_p 0.9",
        "template": "{{ .System }}\nUser: {{ .Prompt }}\nAssistant:",
        "details": {
            "family": "tayarah",
            "format": "symbolic",
            "parameter_size": "Adaptive",
            "quantization_level": "FP16/SIMD"
        }
    }

@router.get("/ps")
async def ollama_ps():
    manager = EngineManager()
    telemetry = manager.get_telemetry()
    return {
        "models": [
            {
                "name": "tayarah:latest",
                "model": "tayarah:latest",
                "size": 980000000,
                "digest": "sha256:tayarah_mach_active",
                "expires_at": "2099-12-31T23:59:59Z",
                "size_vram": telemetry.get("gpu_vram", "0 GB")
            }
        ]
    }

@router.post("/generate")
async def ollama_generate(req: OllamaGenerateRequest):
    manager = EngineManager()
    max_tokens = 256
    temp = 0.7
    top_p = 0.9

    if req.options:
        max_tokens = req.options.get("num_predict", max_tokens)
        temp = req.options.get("temperature", temp)
        top_p = req.options.get("top_p", top_p)

    full_prompt = req.prompt
    if req.system:
        full_prompt = f"System: {req.system}\n{full_prompt}"

    if req.stream:
        def stream_generator():
            for chunk, is_done, meta in manager.stream_generate(
                prompt=full_prompt,
                max_tokens=max_tokens,
                temperature=temp,
                top_p=top_p,
                model_name=req.model
            ):
                if not is_done:
                    packet = {
                        "model": req.model,
                        "created_at": get_iso_timestamp(),
                        "response": chunk,
                        "done": False
                    }
                    yield json.dumps(packet) + "\n"
                else:
                    packet = {
                        "model": req.model,
                        "created_at": get_iso_timestamp(),
                        "response": "",
                        "done": True,
                        "total_duration": meta.get("total_duration_ns", 1000000),
                        "load_duration": 1000000,
                        "prompt_eval_count": 10,
                        "prompt_eval_duration": 500000,
                        "eval_count": meta.get("eval_count", 0),
                        "eval_duration": meta.get("eval_duration_ns", 1000000)
                    }
                    yield json.dumps(packet) + "\n"

        return StreamingResponse(stream_generator(), media_type="application/x-ndjson")
    else:
        result = manager.complete_generate(
            prompt=full_prompt,
            max_tokens=max_tokens,
            temperature=temp,
            top_p=top_p,
            model_name=req.model
        )
        meta = result["meta"]
        return {
            "model": req.model,
            "created_at": get_iso_timestamp(),
            "response": result["text"],
            "done": True,
            "total_duration": meta.get("total_duration_ns", 1000000),
            "load_duration": 1000000,
            "prompt_eval_count": 10,
            "prompt_eval_duration": 500000,
            "eval_count": meta.get("eval_count", 0),
            "eval_duration": meta.get("eval_duration_ns", 1000000)
        }

@router.post("/chat")
async def ollama_chat(req: OllamaChatRequest):
    manager = EngineManager()
    max_tokens = 256
    temp = 0.7
    top_p = 0.9

    if req.options:
        max_tokens = req.options.get("num_predict", max_tokens)
        temp = req.options.get("temperature", temp)
        top_p = req.options.get("top_p", top_p)

    full_prompt = build_prompt_from_messages(req.messages)

    if req.stream:
        def chat_stream_generator():
            for chunk, is_done, meta in manager.stream_generate(
                prompt=full_prompt,
                max_tokens=max_tokens,
                temperature=temp,
                top_p=top_p,
                model_name=req.model
            ):
                if not is_done:
                    packet = {
                        "model": req.model,
                        "created_at": get_iso_timestamp(),
                        "message": {
                            "role": "assistant",
                            "content": chunk
                        },
                        "done": False
                    }
                    yield json.dumps(packet) + "\n"
                else:
                    packet = {
                        "model": req.model,
                        "created_at": get_iso_timestamp(),
                        "message": {
                            "role": "assistant",
                            "content": ""
                        },
                        "done": True,
                        "total_duration": meta.get("total_duration_ns", 1000000),
                        "load_duration": 1000000,
                        "prompt_eval_count": 10,
                        "prompt_eval_duration": 500000,
                        "eval_count": meta.get("eval_count", 0),
                        "eval_duration": meta.get("eval_duration_ns", 1000000)
                    }
                    yield json.dumps(packet) + "\n"

        return StreamingResponse(chat_stream_generator(), media_type="application/x-ndjson")
    else:
        result = manager.complete_generate(
            prompt=full_prompt,
            max_tokens=max_tokens,
            temperature=temp,
            top_p=top_p,
            model_name=req.model
        )
        meta = result["meta"]
        return {
            "model": req.model,
            "created_at": get_iso_timestamp(),
            "message": {
                "role": "assistant",
                "content": result["text"]
            },
            "done": True,
            "total_duration": meta.get("total_duration_ns", 1000000),
            "eval_count": meta.get("eval_count", 0),
            "eval_duration": meta.get("eval_duration_ns", 1000000)
        }
