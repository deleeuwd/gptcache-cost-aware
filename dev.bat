@echo off
REM Simple batch script to build and run development Docker container

if "%1"=="build" (
    echo Building development Docker image...
    docker build -f Dockerfile.dev -t gptcache-dev .
    goto :EOF
)

if "%1"=="run" (
    echo Starting development container...
    docker stop gptcache-dev-container 2>nul
    docker rm gptcache-dev-container 2>nul
    docker run -it --name gptcache-dev-container -v "%CD%:/workspace" -w /workspace gptcache-dev
    goto :EOF
)

if "%1"=="help" (
    echo GPTCache Development Docker Script
    echo.
    echo Usage:
    echo   dev build    - Build the development Docker image
    echo   dev run      - Run the development container
    echo   dev          - Build and run (default)
    echo.
    echo The container will mount your current directory to /workspace
    goto :EOF
)

REM Default: Build and Run
echo Building and starting development environment...
docker build -f Dockerfile.dev -t gptcache-dev .
if errorlevel 1 (
    echo Build failed!
    exit /b 1
)

docker stop gptcache-dev-container 2>nul
docker rm gptcache-dev-container 2>nul
docker run -it --name gptcache-dev-container -v "%CD%:/workspace" -w /workspace gptcache-dev
