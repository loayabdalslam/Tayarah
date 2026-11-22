"""
Tayarah Speculative Runtime Entrypoint (Backward Compatibility Bridge)
"""
import os
import sys

root_dir = os.path.dirname(os.path.abspath(__file__))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from tayarah.runtime import TayarahRuntime, SymbLLMRuntime

__all__ = ["TayarahRuntime", "SymbLLMRuntime"]
