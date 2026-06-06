@echo off
setlocal enabledelayedexpansion

rem Find OpenCV the same way build.bat does, so we can put its DLLs on PATH.
set "OCV="
if defined OpenCV_DIR  set "OCV=%OpenCV_DIR%"
if not defined OCV if defined OPENCV_DIR set "OCV=%OPENCV_DIR%"
if not defined OCV (
  for %%D in ("C:\opencv\build" "C:\tools\opencv\build" "C:\dev\opencv\build" "%VCPKG_ROOT%\installed\x64-windows") do (
    if exist "%%~D\OpenCVConfig.cmake" set "OCV=%%~D"
  )
)

rem Add the OpenCV DLL folder to PATH for this run only (so the exe launches).
if defined OCV (
  for %%V in (vc16 vc17) do if exist "!OCV!\x64\%%V\bin" set "PATH=!PATH!;!OCV!\x64\%%V\bin"
  if exist "!OCV!\bin" set "PATH=!PATH!;!OCV!\bin"
)

set "EXE=build\bin\Release\gesture_demo.exe"
if not exist "%EXE%" (
  echo %EXE% not found. Build it first with build.bat
  exit /b 1
)

rem Run from the repo root so assets\emoji\*.png resolve.
"%EXE%"
endlocal
