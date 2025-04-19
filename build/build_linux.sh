#!/bin/bash
# Build script for Linux

# Ensure we're in the project root directory
cd "$(dirname "$0")/.."

# Create build directory if it doesn't exist
mkdir -p dist

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Install dependencies if needed
pip install -r requirements.txt

# Build the standalone executable
pyinstaller strata.spec

echo "Build completed! Executable is in the dist folder."