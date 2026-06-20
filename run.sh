#!/bin/bash
export PATH="$HOME/.local/bin:$PATH"
cd "$(dirname "$0")"
echo "Starting Resume Screener..."
echo "Open http://localhost:8000 in your browser"
python3 app/main.py
