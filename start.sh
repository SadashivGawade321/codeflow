#!/bin/bash
# CodeFlow AI - Unix Startup Script

echo ""
echo "╔═══════════════════════════════════════════╗"
echo "║          CodeFlow AI - Startup            ║"
echo "║   AI-Powered Code Visualization Platform  ║"
echo "╚═══════════════════════════════════════════╝"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "[*] Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "[*] Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "[*] Installing dependencies..."
pip install -r requirements.txt -q

echo ""
echo "[*] Starting CodeFlow AI server..."
echo "[*] Open http://localhost:8000 in your browser"
echo "[*] Press Ctrl+C to stop the server"
echo ""

cd backend
python main.py
