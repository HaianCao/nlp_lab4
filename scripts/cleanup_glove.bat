@echo off
echo ================================
echo Cleanup GloVe Model Script
echo ================================

echo Checking gensim installation...

REM Get gensim cache directory
echo import gensim.downloader as api > temp_check.py
echo print(api.base_dir) >> temp_check.py

for /f "delims=" %%i in ('python temp_check.py 2^>nul') do set GENSIM_DIR=%%i
del temp_check.py 2>nul

if "%GENSIM_DIR%"=="" (
    echo Error: Cannot find gensim installation
    echo Make sure gensim is installed: pip install gensim
    pause
    exit /b 1
)

echo Gensim cache directory: %GENSIM_DIR%

REM Check if directory exists
if not exist "%GENSIM_DIR%" (
    echo No gensim cache found - already clean!
    echo.
    echo Cleanup completed.
    exit /b 0
)

echo.
echo Current cached models:
dir /b "%GENSIM_DIR%" 2>nul
if errorlevel 1 (
    echo Cache directory is empty
)

echo.
echo WARNING: This will delete ALL cached gensim models!
set /p response="Do you want to continue? (y/N): "

if /i "%response%"=="y" (
    echo.
    echo Cleaning up gensim cache...
    
    REM Remove the entire gensim-data directory
    rmdir /s /q "%GENSIM_DIR%" 2>nul
    
    if not exist "%GENSIM_DIR%" (
        echo Successfully deleted gensim cache!
        echo Freed up disk space
        echo.
        echo Cache status:
        echo   - Cache directory: DELETED
    ) else (
        echo Error: Failed to delete cache
        echo Try running as Administrator or check permissions
        pause
        exit /b 1
    )
) else (
    echo Operation cancelled - no changes made
)

echo.
echo Cleanup script completed!