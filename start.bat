@echo off
setlocal
cd /d "%~dp0"
title Private Document AI

if not exist "venv\Scripts\activate.bat" (
    echo The virtual environment was not found.
    echo Create it first with:  python -m venv venv
    echo Then install the packages:  pip install -r requirements.txt
    pause
    exit /b 1
)

call venv\Scripts\activate.bat

echo Checking that Ollama is running...
ollama list >nul 2>&1
if errorlevel 1 (
    echo Ollama is not running. Starting it...
    start "" /min ollama serve
    ping -n 6 127.0.0.1 >nul
)

echo.
echo Starting Private Document AI at http://127.0.0.1:8501
echo Close this window to stop the app.
echo.

rem Open the browser after the server has had a few seconds to start
start "" /min cmd /c "ping -n 6 127.0.0.1 >nul & start http://127.0.0.1:8501"

python -m streamlit run app.py
pause
