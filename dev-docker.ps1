#!/usr/bin/env pwsh
# Build and run development Docker container

param(
    [switch]$Build,
    [switch]$Run,
    [switch]$Help
)

$ImageName = "gptcache-dev"
$ContainerName = "gptcache-dev-container"

function Show-Help {
    Write-Host "GPTCache Development Docker Script"
    Write-Host ""
    Write-Host "Usage:"
    Write-Host "  .\dev-docker.ps1 -Build    # Build the development Docker image"
    Write-Host "  .\dev-docker.ps1 -Run      # Run the development container"
    Write-Host "  .\dev-docker.ps1           # Build and run (default)"
    Write-Host ""
    Write-Host "The container will mount your current directory to /workspace"
    Write-Host "You can edit files locally and they'll be reflected in the container immediately."
}

function Invoke-Build {
    Write-Host "Building development Docker image..." -ForegroundColor Green
    docker build -f Dockerfile.dev -t $ImageName .
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Docker build failed!"
        exit 1
    }
    Write-Host "Build completed successfully!" -ForegroundColor Green
}

function Start-Container {
    Write-Host "Starting development container..." -ForegroundColor Green
    
    # Stop and remove existing container if it exists
    docker stop $ContainerName 2>$null
    docker rm $ContainerName 2>$null
    
    # Run new container with volume mount
    docker run -it --name $ContainerName `
        -v "${PWD}:/workspace" `
        -w /workspace `
        $ImageName
}

if ($Help) {
    Show-Help
    exit 0
}

if ($Build) {
    Invoke-Build
    exit 0
}

if ($Run) {
    Start-Container
    exit 0
}

# Default: Build and Run
Invoke-Build
Start-Container
