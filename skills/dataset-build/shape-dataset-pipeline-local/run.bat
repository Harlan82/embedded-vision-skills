@echo off
REM ============================================================
REM  Shape dataset pipeline (LOCAL, single entry)
REM  Usage:
REM    run.bat --src "./work/folder" --cls class_2 --zip "./work/batch-a.zip"
REM    run.bat --jobs jobs.json --zip "./work/shapes.zip" --gray 0.20
REM    run.bat --doctor
REM  Only the LAST LINE matters:  RESULT: OK  /  RESULT: FAIL:<reason>
REM ============================================================
setlocal
set HERE=%~dp0
set PY=python
where python >nul 2>nul
if errorlevel 1 set PY=py

if "%~1"=="" (
  echo Usage:
  echo   run.bat --src "./work/folder" --cls NAME --zip "./work/out.zip"
  echo   run.bat --jobs jobs.json --zip "./work/out.zip" --gray 0.20
  echo   run.bat --doctor
  exit /b 2
)

if "%~1"=="--doctor" (
  "%PY%" "%HERE%scripts\doctor.py" %2 %3 %4 %5 %6 %7 %8 %9
) else (
  "%PY%" "%HERE%scripts\pipeline.py" %*
)
endlocal
