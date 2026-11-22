@echo off
title Gemma 4 E4B (7.5B) Turbo Terminal - Ollama Backend
color 0B
echo ============================================================
echo   Starting Gemma 4 E4B (7.5B Q4_K_M GGUF) Turbo Terminal...
echo   Backend: Ollama Native llama-server (Intel AVX2 SIMD)
echo ============================================================
echo.
set KMP_DUPLICATE_LIB_OK=TRUE
python gemma_terminal.py
pause
