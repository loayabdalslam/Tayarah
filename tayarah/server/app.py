import os
import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from tayarah.server.routes_ollama import router as ollama_router
from tayarah.server.routes_openai import router as openai_router
from tayarah.engine_manager import EngineManager

def create_app() -> FastAPI:
    app = FastAPI(
        title="Tayarah Mach-Speed LLM Runtime (طيارة)",
        description="High-Throughput Neuro-Symbolic Streaming Server compatible with Ollama and vLLM / OpenAI APIs.",
        version="0.5.4"
    )

    # Enable CORS for universal integration (Open WebUI, Cursor, Continue, etc.)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include API Routers
    app.include_router(ollama_router)
    app.include_router(openai_router)

    static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

    @app.get("/", response_class=HTMLResponse)
    async def get_index():
        index_file = os.path.join(static_dir, "index.html")
        if os.path.exists(index_file):
            return FileResponse(index_file, media_type="text/html")
        return HTMLResponse("<h1>✈️ Tayarah Mach-Speed Streaming Runtime Online</h1>")

    @app.get("/health")
    async def health_check():
        manager = EngineManager()
        return {
            "status": "healthy",
            "runtime": "Tayarah Mach-Speed Neuro-Symbolic",
            "active_mode": manager.active_mode,
            "drafter": "racore/The1",
            "speculative_extrapolation": "Active"
        }

    @app.get("/status")
    @app.get("/api/telemetry")
    async def get_status():
        manager = EngineManager()
        return manager.get_telemetry()

    return app

app = create_app()
