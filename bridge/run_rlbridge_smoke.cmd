@echo off
setlocal

set "ROOT=%~dp0.."
set "BRIDGE_EXE=%BHUMAN_BRIDGE_EXE%"

if not defined BRIDGE_EXE if exist "%ROOT%\..\SabanaHerons2026\Build\Windows\RLBridge\Release\RLBridge.exe" set "BRIDGE_EXE=%ROOT%\..\SabanaHerons2026\Build\Windows\RLBridge\Release\RLBridge.exe"
if not defined BRIDGE_EXE if exist "%ROOT%\bridge\bhuman_bridge.exe" set "BRIDGE_EXE=%ROOT%\bridge\bhuman_bridge.exe"

if not defined BRIDGE_EXE (
  echo RLBridge executable not found. Set BHUMAN_BRIDGE_EXE or build RLBridge. > "%ROOT%\bridge\rlbridge_smoke_output.txt"
  exit /b 1
)

"%BRIDGE_EXE%" < "%ROOT%\bridge\rlbridge_smoke_input.jsonl" > "%ROOT%\bridge\rlbridge_smoke_output.txt" 2>&1
exit /b %ERRORLEVEL%
