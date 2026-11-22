"""
Tayarah (طيارة) — Mach-Speed Neuro-Symbolic LLM Runtime.
Bypasses the LLM memory bandwidth bottleneck via racore/The1 Inductive World Model
and Single-Pass Vectorized Parallel Verification.
"""

__all__ = [
    "TayarahSymbolicDrafter",
    "TayarahGPUEngine",
    "TayarahUniversalAdapter",
    "TayarahRuntime",
    "SymbLLMRuntime",
    "EngineManager",
    "TayarahDualTerminal",
    "GemmaOllamaTerminal",
    "create_app"
]

def __getattr__(name: str):
    if name == "TayarahSymbolicDrafter":
        from tayarah.drafter import TayarahSymbolicDrafter
        return TayarahSymbolicDrafter
    elif name == "TayarahGPUEngine":
        from tayarah.gpu_engine import TayarahGPUEngine
        return TayarahGPUEngine
    elif name == "TayarahUniversalAdapter":
        from tayarah.adapter import TayarahUniversalAdapter
        return TayarahUniversalAdapter
    elif name in ("TayarahRuntime", "SymbLLMRuntime"):
        from tayarah.runtime import TayarahRuntime
        return TayarahRuntime
    elif name == "EngineManager":
        from tayarah.engine_manager import EngineManager
        return EngineManager
    elif name == "TayarahDualTerminal":
        from tayarah.terminal import TayarahDualTerminal
        return TayarahDualTerminal
    elif name == "GemmaOllamaTerminal":
        from tayarah.gemma_terminal import GemmaOllamaTerminal
        return GemmaOllamaTerminal
    elif name == "create_app":
        from tayarah.server.app import create_app
        return create_app
    raise AttributeError(f"module 'tayarah' has no attribute '{name}'")
