@echo off
REM CodeFlow AI - Windows Startup Script

echo.
echo  ╔═══════════════════════════════════════════╗
echo  ║          CodeFlow AI - Startup            ║
echo  ║   AI-Powered Code Visualization Platform  ║
echo  ╚═══════════════════════════════════════════╝
echo.

REM Check if virtual environment exists
if not exist "venv" (
    echo [*] Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo [*] Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies
echo [*] Installing dependencies...
pip install -r requirements.txt -q

echo.
echo [*] Starting CodeFlow AI server...
echo [*] Open http://localhost:8000 in your browser
echo [*] Press Ctrl+C to stop the server
echo.

cd backend
python main.py
