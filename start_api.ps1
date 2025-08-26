# start_api.ps1 — SpeechUp IA API Startup Script for Windows PowerShell
# 
# This script ensures the SpeechUp IA API starts with all dependencies properly configured:
# - Activates the Python virtual environment
# - Sets required environment variables for audio and ASR features
# - Dynamically adds ffmpeg to PATH from common installation locations
# - Verifies ffmpeg availability and version
# - Starts the FastAPI server with uvicorn
#
# Author: SpeechUp IA Team
# Usage: .\start_api.ps1

# =============================================================================
# 1. SCRIPT SETUP AND NAVIGATION
# =============================================================================

# Get the directory where this script is located and navigate to it
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

Write-Host "Starting SpeechUp IA API..." -ForegroundColor Green
Write-Host "Working directory: $(Get-Location)" -ForegroundColor Cyan

# =============================================================================
# 2. VIRTUAL ENVIRONMENT ACTIVATION
# =============================================================================

Write-Host "Activating Python virtual environment..." -ForegroundColor Yellow

# Check if virtual environment exists
if (-not (Test-Path ".\.venv\Scripts\Activate.ps1")) {
    Write-Error "Virtual environment not found at .\.venv\Scripts\Activate.ps1"
    Write-Host "Please run 'python -m venv .venv' to create the virtual environment" -ForegroundColor Red
    exit 1
}

# Activate the virtual environment
try {
    .\.venv\Scripts\Activate.ps1
    Write-Host "Virtual environment activated successfully" -ForegroundColor Green
} catch {
    Write-Error "Failed to activate virtual environment: $_"
    exit 1
}

# =============================================================================
# 3. ENVIRONMENT VARIABLES CONFIGURATION
# =============================================================================

Write-Host "Setting environment variables..." -ForegroundColor Yellow

# Core SpeechUp configuration
$env:SPEECHUP_USE_AUDIO = "1"
$env:SPEECHUP_USE_ASR = "1"
$env:SPEECHUP_ASR_MODEL = "base"
$env:WHISPER_DEVICE = "cpu"
$env:SPEECHUP_DEBUG_ASR = "1"
$env:SPEECHUP_ASR_MAX_WINDOW_SEC = "20"
$env:LOG_LEVEL = "INFO"
$env:PATH = "C:\Users\kenyc\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-7.1.1-full_build\bin;" + $env:PATH
$env:SPEECHUP_ASR_MODEL="base"
$env:WHISPER_DEVICE="cpu"




Write-Host "Environment variables configured" -ForegroundColor Green

# =============================================================================
# 4. FFMPEG PATH MANAGEMENT
# =============================================================================

Write-Host "Configuring ffmpeg PATH..." -ForegroundColor Yellow

# Define possible ffmpeg installation paths
$ffmpegPaths = @(
    "$Env:LOCALAPPDATA\Microsoft\WinGet\Links",  # winget installation
    "C:\ffmpeg\bin",                             # Manual installation
    "$Env:PROGRAMFILES\ffmpeg\bin",              # Program Files installation
    "$Env:LOCALAPPDATA\Programs\ffmpeg\bin"      # User Programs installation
)

# Function to check if a path exists and add it to PATH if not already there
function Add-ToPathIfNotExists {
    param([string]$path)
    
    if (Test-Path $path) {
        if ($env:Path -notlike "*$path*") {
            $env:Path = "$path;$env:Path"
            Write-Host "Added to PATH: $path" -ForegroundColor Green
            return $true
        } else {
            Write-Host "Already in PATH: $path" -ForegroundColor Blue
            return $false
        }
    } else {
        Write-Host "Path not found: $path" -ForegroundColor Red
        return $false
    }
}

# Add each ffmpeg path to PATH if it exists
$pathsAdded = 0
foreach ($path in $ffmpegPaths) {
    if (Add-ToPathIfNotExists $path) {
        $pathsAdded++
    }
}

if ($pathsAdded -eq 0) {
    Write-Warning "No new ffmpeg paths were added to PATH"
}

# =============================================================================
# 5. FFMPEG VERIFICATION
# =============================================================================

Write-Host "Verifying ffmpeg availability..." -ForegroundColor Yellow

# Try to find ffmpeg in the updated PATH
$ffmpegCommand = Get-Command ffmpeg -ErrorAction SilentlyContinue

if ($ffmpegCommand) {
    $ffmpegPath = $ffmpegCommand.Path
    Write-Host "ffmpeg found at: $ffmpegPath" -ForegroundColor Green
    
    # Get ffmpeg version
    try {
        $ffmpegVersion = & ffmpeg -version 2>$null | Select-Object -First 1
        if ($ffmpegVersion) {
            Write-Host "Version: $ffmpegVersion" -ForegroundColor Cyan
        }
    } catch {
        Write-Warning "Could not retrieve ffmpeg version"
    }
} else {
    Write-Error "ffmpeg not found in PATH after configuration"
    Write-Host "Troubleshooting steps:" -ForegroundColor Red
    Write-Host "   1. Ensure ffmpeg is installed (winget install Gyan.FFmpeg)" -ForegroundColor Red
    Write-Host "   2. Check if ffmpeg.exe exists in one of these locations:" -ForegroundColor Red
    foreach ($path in $ffmpegPaths) {
        if (Test-Path $path) {
            Write-Host "      - $path" -ForegroundColor Red
        }
    }
    Write-Host "   3. Restart PowerShell to refresh environment variables" -ForegroundColor Red
    exit 1
}

# =============================================================================
# 6. FINAL CONFIGURATION DISPLAY
# =============================================================================

Write-Host "Final Configuration:" -ForegroundColor Cyan
Write-Host "   SPEECHUP_USE_AUDIO: $($env:SPEECHUP_USE_AUDIO)" -ForegroundColor White
Write-Host "   SPEECHUP_USE_ASR: $($env:SPEECHUP_USE_ASR)" -ForegroundColor White
Write-Host "   SPEECHUP_ASR_MODEL: $($env:SPEECHUP_ASR_MODEL)" -ForegroundColor White
Write-Host "   WHISPER_DEVICE: $($env:WHISPER_DEVICE)" -ForegroundColor White
Write-Host "   SPEECHUP_DEBUG_ASR: $($env:SPEECHUP_DEBUG_ASR)" -ForegroundColor White
Write-Host "   SPEECHUP_ASR_MAX_WINDOW_SEC: $($env:SPEECHUP_ASR_MAX_WINDOW_SEC)" -ForegroundColor White
Write-Host "   LOG_LEVEL: $($env:LOG_LEVEL)" -ForegroundColor White
Write-Host "   ffmpeg: $ffmpegPath" -ForegroundColor White

# Display Python version
try {
    $pythonVersion = python --version 2>$null
    if ($pythonVersion) {
        Write-Host "   Python: $pythonVersion" -ForegroundColor White
    }
} catch {
    Write-Warning "⚠️  Could not retrieve Python version"
}

# =============================================================================
# 7. START THE API SERVER
# =============================================================================

Write-Host "Starting SpeechUp IA API server..." -ForegroundColor Green
Write-Host "Server will be available at: http://localhost:8000" -ForegroundColor Cyan
Write-Host "API documentation: http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host " Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host "=" * 60 -ForegroundColor DarkGray

# Start the FastAPI server
try {
    uvicorn api.main:app --reload
} catch {
    Write-Error "Failed to start API server: $_"
    Write-Host "Make sure all dependencies are installed: pip install -r requirements.txt" -ForegroundColor Red
    exit 1
}
