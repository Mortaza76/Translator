#!/bin/bash

# Startup script for the Urdu Translation UI
# This script starts both the backend and frontend servers

echo "🚀 Starting Urdu Speech Translation UI..."
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

# Check if Node.js is available
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js 18+ first."
    exit 1
fi

# Check if npm is available
if ! command -v npm &> /dev/null; then
    echo "❌ npm is not installed. Please install npm first."
    exit 1
fi

# Check if frontend dependencies are installed
if [ ! -d "frontend/node_modules" ]; then
    echo "📦 Installing frontend dependencies..."
    cd frontend
    npm install
    cd ..
    echo "✅ Frontend dependencies installed"
    echo ""
fi

# Start backend server in background
echo "🔧 Starting backend server on port 8000..."
python3 api_server.py &
BACKEND_PID=$!

# Wait a moment for backend to start
sleep 2

# Start frontend server
echo "🎨 Starting frontend server on port 3000..."
cd frontend
npm run dev &
FRONTEND_PID=$!
cd ..

echo ""
echo "✅ Both servers are starting..."
echo ""
echo "📍 Backend API: http://localhost:8000"
echo "📍 Frontend UI: http://localhost:3000"
echo ""
echo "Press Ctrl+C to stop both servers"

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Stopping servers..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    echo "✅ Servers stopped"
    exit 0
}

# Trap Ctrl+C
trap cleanup INT

# Wait for both processes
wait
