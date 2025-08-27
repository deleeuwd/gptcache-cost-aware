# Development Docker Setup

This directory contains a development Docker environment that mounts your local code for easy development.

## Development Docker Setup

This repo includes a development Docker setup that mounts your local code into a container for interactive development.

Key goals:
- Mount the project root into the container so edits on the host are visible immediately
- Install Python dependencies from `requirements.txt` and run `setup.py` in editable mode
- Provide short helper scripts for Windows and cross-platform manual commands

## Usage (Windows - cmd.exe)

From the repository root (cmd.exe):

```cmd
:: Build and run (default)
dev

:: Build only
dev build

:: Run only (after building)
dev run
```

Note: run these from the repo root where `dev.bat` lives.

## Usage (Windows - PowerShell)

From the repository root (PowerShell):

```powershell
# Build and run (default)
.\dev-docker.ps1

# Build only
.\dev-docker.ps1 -Build

# Run only (after building)
.\dev-docker.ps1 -Run
```

PowerShell won't execute a program from the current directory by name; prepend `./` or `.\` as shown.

If script execution is restricted on your system, run PowerShell with bypass for this script once:

```powershell
# Run the script ignoring execution policy for this invocation
powershell -ExecutionPolicy Bypass -File .\dev-docker.ps1
```

## Usage (Linux / macOS - bash)

From the repository root (bash):

```bash
# Build the image
docker build -f Dockerfile.dev -t gptcache-dev .

# Run the container (mount current directory into /workspace)
docker run -it --name gptcache-dev-container -v "$(pwd):/workspace" -w /workspace gptcache-dev
```

## Manual Docker commands (cross-platform)

Build:

```bash
docker build -f Dockerfile.dev -t gptcache-dev .
```

Run (Windows PowerShell - use ${PWD}.Path to get full path):

```powershell
docker run -it --name gptcache-dev-container -v "${PWD}.Path:/workspace" -w /workspace gptcache-dev
```

Run (Windows cmd.exe):

```cmd
docker run -it --name gptcache-dev-container -v "%CD%:/workspace" -w /workspace gptcache-dev
```

Run (bash):

```bash
docker run -it --name gptcache-dev-container -v "$(pwd):/workspace" -w /workspace gptcache-dev
```

## How it behaves

- The container starts with `/workspace` as the working directory (the project root)
- The image installs packages from `requirements.txt` and runs `python -m pip install -e .` during build
- You get an interactive bash shell inside the container; edit files on the host and run them inside the container

## Adding dependencies

1. Add the packages to `requirements.txt`
2. Rebuild the development image:

```powershell
# Windows (PowerShell or cmd)
.\dev-docker.ps1 -Build
# or
dev build

# Linux / bash
docker build -f Dockerfile.dev -t gptcache-dev .
```

## Optional: create a persistent `dev` shortcut on Windows PowerShell

If you want `dev` as a command in all new PowerShell sessions, add an alias to your PowerShell profile (one-time):

```powershell
'Set-Alias dev "C:\Users\annie\OneDrive\Documents\GitHub\gptcache-cost-aware\dev.bat"' >> $PROFILE
```

Open a new PowerShell window for the alias to take effect.

## Troubleshooting

- If Docker build fails, check the build output for the failing pip or apt command. Often re-running fixes transient network issues.
- If filesystem changes inside the container don't match the host, ensure you mounted the correct folder (`%CD%` / `$(pwd)` / `${PWD}.Path`).

---

This file was updated to include exact, copyable commands for Windows (cmd and PowerShell) and Linux/macOS (bash).
