@echo off
set PYEXE=E:\miniconda3\envs\EChatbot\python.exe
set ROOT=%~dp0

echo ===== yc-pro startup =====
cd /d "%ROOT%"

if not exist "data\chroma_db" (
    echo [1/2] ChromaDB index not found. Building now ^(first run only^)...
    %PYEXE% rag\build_index.py
    if errorlevel 1 (
        echo ERROR: index build failed. See output above.
        pause & exit /b 1
    )
)

echo [2/2] Launching Streamlit...
%PYEXE% -m streamlit run app.py
