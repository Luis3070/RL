@echo off
setlocal

set "ROOT=%~dp0.."
set "OUT=%ROOT%\bridge\env_bridge_smoke_output.txt"
set "PYTHON_EXE="

if defined CONDA_PREFIX if exist "%CONDA_PREFIX%\python.exe" set "PYTHON_EXE=%CONDA_PREFIX%\python.exe"
if not defined PYTHON_EXE if exist "%ROOT%\.venv\Scripts\python.exe" set "PYTHON_EXE=%ROOT%\.venv\Scripts\python.exe"
if not defined PYTHON_EXE set "PYTHON_EXE=python"

"%PYTHON_EXE%" "%ROOT%\smoke_test_environment_bridge.py" > "%OUT%" 2>&1
exit /b %ERRORLEVEL%
