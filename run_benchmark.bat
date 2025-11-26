@echo off
setlocal enabledelayedexpansion

set SEEDS=42 1

echo ========================================================
echo STARTING AUTOMATED BENCHMARK
echo Seeds to run: %SEEDS%
echo ========================================================

for %%s in (%SEEDS%) do (
    echo.
    echo --------------------------------------------------------
    echo [RUNNING] All Datasets with SEED: %%s
    echo --------------------------------------------------------
    
    python main.py --all --seed %%s
    
    if !errorlevel! neq 0 (
        echo [ERROR] Failed to run with seed %%s
    ) else (
        echo [SUCCESS] Completed run with seed %%s
    )
)

echo.
echo ========================================================
echo ALL BENCHMARKS COMPLETED.
echo check 'results/' folder for summary files.
echo ========================================================
pause