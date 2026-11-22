@echo off
title Tayarah Benchmark Video Generator
cd /d "%~dp0"
echo ========================================================
echo   Generating Tayarah vs Ollama Benchmark Video (1080p)
echo ========================================================
python generate_comparison_video.py
echo.
echo Video generation complete! File saved as: tayarah_vs_ollama_benchmark.mp4
pause
