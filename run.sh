#!/bin/bash
export PATH="$HOME/.local/bin:$PATH"
cd "$(dirname "$0")"
echo "Starting Resume Screener..."

# Install all required dependencies
echo "Installing dependencies..."
pip3 install -r requirements.txt || echo "Warning: Some dependencies failed to install."

# Ensure spaCy model is downloaded (required for NLP-based resume analysis)
python3 -m spacy download en_core_web_sm --quiet 2>/dev/null || echo "Warning: Could not download spaCy model. NLP features will use fallback methods."

echo "Checking for ghost servers on port 8000..."
fuser -k 8000/tcp 2>/dev/null || true

echo "Open http://localhost:8000 in your browser"
python3 app/main.py