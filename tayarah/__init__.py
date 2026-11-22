"""
Tayarah (طيارة) — Mach-Speed Neuro-Symbolic LLM Runtime.
Bypasses the LLM memory bandwidth bottleneck via racore/The1 Inductive World Model
and Single-Pass Vectorized Parallel Verification.
"""

from tayarah.drafter import TayarahSymbolicDrafter
from tayarah.gpu_engine import TayarahGPUEngine
from tayarah.adapter import TayarahUniversalAdapter
from tayarah.runtime import TayarahRuntime, SymbLLMRuntime
from tayarah.engine_manager import EngineManager
from tayarah.terminal import TayarahDualTerminal
from tayarah.server.app import create_app

__all__ = [
    "TayarahSymbolicDrafter",
    "TayarahGPUEngine",
    "TayarahUniversalAdapter",
    "TayarahRuntime",
    "SymbLLMRuntime",
    "EngineManager",
    "TayarahDualTerminal",
    "create_app"
]
