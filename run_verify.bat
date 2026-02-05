@echo off
echo ============================================
echo Self-Evolving Code Agent - Docker Setup
echo ============================================

echo.
echo [1/3] Checking Docker status...
docker version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Docker is not running.
    echo Please start Docker Desktop and try again.
    echo.
    echo To start Docker Desktop:
    echo   - Search "Docker Desktop" in Start Menu
    echo   - Wait for the icon in system tray to turn green
    pause
    exit /b 1
)

echo Docker is running.

echo.
echo [2/3] Starting Qdrant...
docker-compose up -d qdrant
if errorlevel 1 (
    echo ERROR: Failed to start Qdrant
    pause
    exit /b 1
)
echo Qdrant started successfully.

echo.
echo [3/3] Running verification...
python verify_full.py

echo.
echo ============================================
echo Verification complete!
echo ============================================
pause
