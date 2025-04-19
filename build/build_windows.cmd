@echo off
REM Build script for Windows

REM Ensure we're in the project root directory
cd %~dp0\..

REM Create build directory if it doesn't exist
if not exist dist mkdir dist

REM Activate virtual environment if it exists
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
)

REM Install dependencies if needed
pip install -r requirements.txt

REM Build the standalone executable
pyinstaller strata.spec

echo Build completed! Executable is in the dist folder.