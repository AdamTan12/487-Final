@echo off
setlocal enabledelayedexpansion

set "EXE=build\bin\Release\gesture_demo.exe"
if not exist "%EXE%" (
  echo %EXE% not found. Build it first with build.bat
  exit /b 1
)

rem If built against a system OpenCV, put its DLL folder on PATH for this run.
set "OCV="
if defined OpenCV_DIR  set "OCV=%OpenCV_DIR%"
if not defined OCV if defined OPENCV_DIR set "OCV=%OPENCV_DIR%"
if not defined OCV (
  for %%D in ("C:\opencv\build" "C:\tools\opencv\build" "C:\dev\opencv\build") do (
    if exist "%%~D\OpenCVConfig.cmake" set "OCV=%%~D"
  )
)
if defined OCV (
  for %%V in (vc16 vc17) do if exist "!OCV!\x64\%%V\bin" set "PATH=!PATH!;!OCV!\x64\%%V\bin"
  if exist "!OCV!\bin" set "PATH=!PATH!;!OCV!\bin"
)

rem If built via vcpkg, its DLLs are auto-copied next to the exe, but add the
rem install bin too just in case.
if exist "build\vcpkg_installed\x64-windows\bin" set "PATH=!PATH!;%CD%\build\vcpkg_installed\x64-windows\bin"

rem Run from the repo root so assets\emoji\*.png resolve.
"%EXE%"
endlocal
