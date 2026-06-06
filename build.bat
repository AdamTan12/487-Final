@echo off
setlocal enabledelayedexpansion

set "ROOT=%CD%"

rem ---------------------------------------------------------------------------
rem 1) Fast path: if OpenCV is already on this machine, use it (no compiling).
rem ---------------------------------------------------------------------------
set "OCV="
if defined OpenCV_DIR  set "OCV=%OpenCV_DIR%"
if not defined OCV if defined OPENCV_DIR set "OCV=%OPENCV_DIR%"
if not defined OCV (
  for %%D in ("C:\opencv\build" "C:\tools\opencv\build" "C:\dev\opencv\build") do (
    if exist "%%~D\OpenCVConfig.cmake" set "OCV=%%~D"
  )
)

if not exist build mkdir build

if defined OCV (
  echo Found existing OpenCV at: !OCV!
  cd build
  cmake .. -G "Visual Studio 17 2022" -A x64 -D OpenCV_DIR="!OCV!"
) else (
  rem -------------------------------------------------------------------------
  rem 2) No system OpenCV: bootstrap a local vcpkg and let it install OpenCV.
  rem    First run compiles OpenCV from source and is SLOW (grab a coffee);
  rem    every build after that is cached and fast.
  rem -------------------------------------------------------------------------
  echo No system OpenCV found. Bootstrapping vcpkg and installing OpenCV...
  if not exist vcpkg (
    git clone https://github.com/microsoft/vcpkg.git || (echo git clone failed & exit /b 1)
  )
  if not exist vcpkg\vcpkg.exe (
    call vcpkg\bootstrap-vcpkg.bat -disableMetrics || (echo vcpkg bootstrap failed & exit /b 1)
  )
  cd build
  cmake .. -G "Visual Studio 17 2022" -A x64 ^
    -D CMAKE_TOOLCHAIN_FILE="!ROOT!\vcpkg\scripts\buildsystems\vcpkg.cmake" ^
    -D VCPKG_TARGET_TRIPLET=x64-windows
)

if errorlevel 1 (
  echo.
  echo CMake configure failed. See the message above.
  exit /b 1
)

cmake --build . --config Release
if errorlevel 1 exit /b 1

cd ..
echo.
echo Built: build\bin\Release\gesture_demo.exe
echo Run with: run.bat
endlocal
