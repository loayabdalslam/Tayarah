"""
Tayarah Symbolic Drafter Entrypoint (Backward Compatibility Bridge)
"""
import os
import sys

root_dir = os.path.dirname(os.path.abspath(__file__))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from tayarah.drafter import TayarahSymbolicDrafter

__all__ = ["TayarahSymbolicDrafter"]
