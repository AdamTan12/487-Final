@echo off
setlocal enabledelayedexpansion

rem Let CMake's own detection win first (OpenCV_DIR env var, vcpkg toolchain).
rem Only fall back to guessing common install dirs if nothing is set.
set "OCV="
if defined OpenCV_DIR  set "OCV=%OpenCV_DIR%"
if not defined OCV if defined OPENCV_DIR set "OCV=%OPENCV_DIR%"
if not defined OCV (
  for %%D in ("C:\opencv\build" "C:\tools\opencv\build" "C:\dev\opencv\build" "%VCPKG_ROOT%\installed\x64-windows") do (
    if exist "%%~D\OpenCVConfig.cmake" set "OCV=%%~D"
  )
)

set "ARG="
if defined OCV (
  echo Using OpenCV at: !OCV!
  set "ARG=-D OpenCV_DIR=!OCV!"
)

if not exist build mkdir build
cd build

cmake .. -G "Visual Studio 17 2022" -A x64 %ARG%
if errorlevel 1 (
  echo.
  echo CMake configure failed. If OpenCV was not found, set OpenCV_DIR to the
  echo folder that contains OpenCVConfig.cmake and retry, e.g.:
  echo     set OpenCV_DIR=C:\opencv\build
  exit /b 1
)

cmake --build . --config Release
if errorlevel 1 exit /b 1

cd ..
echo.
echo Built: build\bin\Release\gesture_demo.exe
echo Run with: run.bat
endlocal
