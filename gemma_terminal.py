"""
Gemma 4 E4B (7.5B Parameters) Q4_K_M GGUF Dedicated Terminal Entrypoint
Powered by Ollama Native Backend Engine (llama-server) + racore/The1
"""
import os
import sys

# Ensure root is in sys.path
root_dir = os.path.dirname(os.path.abspath(__file__))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from tayarah.gemma_terminal import GemmaOllamaTerminal, main

if __name__ == "__main__":
    main()
