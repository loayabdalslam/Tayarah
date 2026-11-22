import os
import sys
import argparse
import uvicorn

# Ensure OpenMP compatibility for Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Add current directory to path
base_dir = os.path.dirname(os.path.abspath(__file__))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from tayarah.engine_manager import EngineManager

def main():
    parser = argparse.ArgumentParser(description="Tayarah Mach-Speed Streaming Server (طيارة)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host interface to bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=11435, help="Port to listen on (default: 11435)")
    parser.add_argument("--mode", type=str, default="gpu", choices=["gpu", "cpu"], help="Default acceleration mode (default: gpu)")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    args = parser.parse_args()

    # Pre-initialize engine manager with chosen mode
    manager = EngineManager(default_mode=args.mode)

    banner = f"""
========================================================================
  ✈️  TAYARAH MACH-SPEED STREAMING SERVER (طيارة) ONLINE
========================================================================
  • Dedicated Server URL : http://127.0.0.1:{args.port}
  • Live Web Playground  : http://127.0.0.1:{args.port}/
  • Ollama Compatible    : http://127.0.0.1:{args.port}/api
  • vLLM / OpenAI API    : http://127.0.0.1:{args.port}/v1
  • Active Engine        : {manager.active_mode.upper()} Mode (racore/The1 Active)
  • Streaming Endpoints  :
      - Ollama Generate  : POST http://127.0.0.1:{args.port}/api/generate
      - Ollama Chat      : POST http://127.0.0.1:{args.port}/api/chat
      - OpenAI Chat      : POST http://127.0.0.1:{args.port}/v1/chat/completions
========================================================================
"""
    print(banner)

    uvicorn.run(
        "tayarah.server.app:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level="info"
    )

if __name__ == "__main__":
    main()
