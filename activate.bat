@echo off
REM Activate the virtual environment for CNNProto

echo ========================================
echo Activating CNNProto Virtual Environment
echo ========================================
echo.

call venv\Scripts\activate.bat

echo.
echo Virtual environment activated!
echo Python version:
python --version
echo.
echo To deactivate, type: deactivate
echo.
