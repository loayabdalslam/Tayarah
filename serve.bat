@echo off
title Tayarah Mach-Speed Streaming Server (Ollama / vLLM)
color 0B
echo ============================================================
echo   Starting Tayarah Streaming Server (طيارة)...
echo   Compatible with Ollama and vLLM / OpenAI APIs
echo ============================================================
echo.
set KMP_DUPLICATE_LIB_OK=TRUE
python run_server.py --port 11435 --mode gpu
pause
