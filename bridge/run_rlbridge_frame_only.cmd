@echo off
"C:\Users\Alegu\Sabana2026\Build\Windows\RLBridge\Release\RLBridge.exe" < bridge\rlbridge_smoke_input_frame_only.jsonl > bridge\rlbridge_smoke_output_frame_only.txt 2>&1
exit /b %ERRORLEVEL%
