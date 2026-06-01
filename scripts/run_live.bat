@echo off
setlocal

set BIN=build\bin\Release\gesture_recognition.exe
if not exist %BIN% (
    echo Binary not found: %BIN%
    echo Run scripts\build.bat first.
    exit /b 1
)

%BIN% --model models/gesture_classifier.onnx %*
endlocal
