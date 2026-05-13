#!/bin/bash
# Quick script to start the server

cd /Users/keshav/Desktop/Deep\ Learning\ Web\ App

echo "Starting Face Recognition Server..."
echo ""

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "   Please create it first: python3 -m venv venv"
    exit 1
fi

# Activate venv and start server
source venv/bin/activate

echo "✅ Virtual environment activated"
echo "🚀 Starting server on http://localhost:8000"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
