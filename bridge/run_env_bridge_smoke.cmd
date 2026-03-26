@echo off
REM Simple launcher for the end-to-end environment <-> RLBridge smoke test.
REM Prefers the known-good RLRobocup conda env on this machine.

set "CONDA_PY=C:\Users\Alegu\miniconda3\envs\RLRobocup\python.exe"

if exist "%CONDA_PY%" (
  "%CONDA_PY%" smoke_test_environment_bridge.py > bridge\env_bridge_smoke_output.txt 2>&1
) else (
  py -3 smoke_test_environment_bridge.py > bridge\env_bridge_smoke_output.txt 2>&1
)
