@echo off
setlocal
set ROOT=%~dp0
cd /d "%ROOT%"

echo ===== yc-pro startup =====
echo Project: %ROOT%
echo.

rem Prefer the currently activated virtual environment.
if defined VIRTUAL_ENV (
    set "PYEXE=%VIRTUAL_ENV%\Scripts\python.exe"
    echo Using activated virtual environment:
    echo   %VIRTUAL_ENV%
) else if defined CONDA_PREFIX (
    set "PYEXE=%CONDA_PREFIX%\python.exe"
    echo Using activated Conda environment:
    echo   %CONDA_PREFIX%
) else if exist "%ROOT%.venv\Scripts\python.exe" (
    set "PYEXE=%ROOT%.venv\Scripts\python.exe"
    echo Using project virtual environment:
    echo   %ROOT%.venv
) else if exist "D:\python-envs\chatbot\Scripts\python.exe" (
    set "PYEXE=D:\python-envs\chatbot\Scripts\python.exe"
    echo Using known local virtual environment:
    echo   D:\python-envs\chatbot
) else (
    set "PYEXE=python"
    echo Using python from PATH.
)

echo.
echo [1/4] Checking Python...
"%PYEXE%" --version
if errorlevel 1 (
    echo ERROR: Python is not available. Please create or activate a virtual environment first.
    pause
    exit /b 1
)

echo.
echo [2/4] Checking Streamlit...
"%PYEXE%" -c "import streamlit; print('Streamlit OK')"
if errorlevel 1 (
    echo.
    echo ERROR: Streamlit or other startup dependencies are missing.
    echo Run:
    echo   "%PYEXE%" -m pip install -r requirements.txt
    pause
    exit /b 1
)

echo.
echo [3/4] Checking ChromaDB index...
if not exist "data\chroma_db" (
    echo ChromaDB index not found. Building now...
    "%PYEXE%" rag\build_index.py
    if errorlevel 1 (
        echo ERROR: index build failed. See output above.
        pause
        exit /b 1
    )
) else (
    echo ChromaDB index found.
)

echo.
echo [4/4] Launching Streamlit...
echo If the terminal stays open here, the app is running normally.
echo Open the Local URL shown below in your browser.
echo.
"%PYEXE%" -m streamlit run app.py

echo.
echo Streamlit has stopped.
pause
