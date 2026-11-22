"""
Tayarah Universal Adapter Entrypoint (Backward Compatibility Bridge)
"""
import os
import sys

root_dir = os.path.dirname(os.path.abspath(__file__))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from tayarah.adapter import TayarahUniversalAdapter

if __name__ == "__main__":
    adapter = TayarahUniversalAdapter("Qwen/Qwen2.5-Coder-0.5B-Instruct")
    print("Metadata:", adapter.metadata)
