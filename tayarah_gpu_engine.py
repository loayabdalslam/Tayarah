"""
Tayarah GPU Engine Entrypoint (Backward Compatibility Bridge)
"""
import os
import sys

root_dir = os.path.dirname(os.path.abspath(__file__))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from tayarah.gpu_engine import TayarahGPUEngine

if __name__ == "__main__":
    print("Testing Tayarah GPU Native Engine...")
    eng = TayarahGPUEngine()
    bench = eng.benchmark_tensor_speed(100)
    print(f"Tensor Throughput: {bench['throughput_tps']} tokens/sec ({bench['tensor_time_ms']} ms)")
