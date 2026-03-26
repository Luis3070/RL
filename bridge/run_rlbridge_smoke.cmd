@echo off
"C:\Users\Alegu\Sabana2026\Build\Windows\RLBridge\Release\RLBridge.exe" < bridge\rlbridge_smoke_input.jsonl > bridge\rlbridge_smoke_output.txt 2>&1
exit /b %ERRORLEVEL%
