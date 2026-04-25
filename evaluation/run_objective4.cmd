@echo off
setlocal

cd /d "%~dp0.."

if not exist ".venv\Scripts\python.exe" (
  echo [error] Missing .venv\Scripts\python.exe
  echo Please create or restore the project virtual environment first.
  exit /b 1
)

echo Checking Ollama at http://localhost:11434 ...
curl -s http://localhost:11434/api/tags >nul
if errorlevel 1 (
  echo [error] Cannot reach Ollama.
  echo Start Ollama first, then run this command again:
  echo   ollama serve
  exit /b 1
)

echo Running Objective 4 generated-answer evaluation ...
".venv\Scripts\python.exe" evaluation\run_evaluation.py --with-generation

echo.
echo Objective 4 outputs:
echo   data\evaluation\summary.json
echo   data\evaluation\evaluation_results.csv
echo   data\evaluation\topk_retrieval_samples.csv

endlocal
