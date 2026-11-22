import struct
import os

gguf_path = os.path.expanduser("~/.lmstudio/models/lmstudio-community/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q4_K_M.gguf")
if os.path.exists(gguf_path):
    print("File exists, size:", os.path.getsize(gguf_path) / (1024**3), "GB")
    with open(gguf_path, "rb") as f:
        magic = f.read(4)
        version, n_tensors, n_kv = struct.unpack("<III", f.read(12))
        print(f"Magic: {magic}, Version: {version}, Tensors: {n_tensors}, Key-Values: {n_kv}")
else:
    print("File not found")
