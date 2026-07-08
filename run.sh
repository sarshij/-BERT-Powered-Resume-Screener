#!/bin/bash
export PATH="$HOME/.local/bin:$PATH"
cd "$(dirname "$0")"
echo "Starting Resume Screener..."

# Ensure spaCy model is downloaded (required for NLP-based resume analysis)
python3 -m spacy download en_core_web_sm --quiet 2>/dev/null || echo "Warning: Could not download spaCy model. NLP features will use fallback methods."

echo "Open http://localhost:8000 in your browser"
python3 app/main.py
