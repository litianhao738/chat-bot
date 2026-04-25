@echo off
setlocal

cd /d "%~dp0.."

if not exist ".venv\Scripts\python.exe" (
  echo [error] Missing .venv\Scripts\python.exe
  echo Please create or restore the project virtual environment first.
  exit /b 1
)

echo Summarizing Objective 5 follow-up prediction analytics ...
".venv\Scripts\python.exe" evaluation\summarize_followups.py

echo.
echo Objective 5 outputs:
echo   data\analytics\followup_events.jsonl
echo   data\analytics\followup_summary.json
echo   data\analytics\followup_journeys.csv

endlocal
