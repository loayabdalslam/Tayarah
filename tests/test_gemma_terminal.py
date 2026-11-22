import os
import sys
import time

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from tayarah.gemma_terminal import GemmaOllamaTerminal

def test_gemma():
    print("Testing Gemma 4 E4B (7.5B) Ollama Turbo Terminal...")
    terminal = GemmaOllamaTerminal(port=8085, n_gpu_layers=0, threads=8)
    
    print("1. Booting / ensuring Ollama backend on port 8085...")
    terminal.ensure_server()
    assert terminal.is_server_running(), "Server failed to start"
    print("✅ Backend is online and healthy!")

    print("\n2. Testing streaming generation...")
    terminal.stream_chat("Say 'Gemma 4 is ready' in 4 words.")
    
    print("\n3. Testing live hardware benchmark...")
    terminal.run_benchmark(tokens_to_test=20)

    print("\n4. Terminating backend cleanly...")
    if terminal.proc:
        terminal.proc.terminate()
        terminal.proc.wait(timeout=5)
    print("✅ All Gemma 4 tests passed with flying colors!")

if __name__ == "__main__":
    test_gemma()
