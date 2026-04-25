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
echo Generating Objective 5 figures ...
".venv\Scripts\python.exe" evaluation\generate_obj5_figures.py

echo.
echo Objective 5 outputs:
echo   data\analytics\followup_events.jsonl
echo   data\analytics\followup_summary.json
echo   data\analytics\followup_journeys.csv
echo   outputs\obj5\user_journey_sankey.png
echo   outputs\obj5\prediction_accuracy_bar.png
echo   outputs\obj5\followup_metrics_table.png
echo   outputs\obj5\ctr_by_topic.png
echo   outputs\obj5\clicked_rank_distribution.png
echo   outputs\obj5\manifest.json

endlocal
