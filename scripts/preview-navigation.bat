@echo off
setlocal
if not defined ISAACSIM_ROOT set "ISAACSIM_ROOT=C:\isaacsim\isaac-sim-standalone-6.2.0-windows-x86_64"
if not defined ISAACLAB_ROOT set "ISAACLAB_ROOT=%USERPROFILE%\Desktop\mentorpi\IsaacLab"
set "PYTHONEXE=%ISAACLAB_ROOT%\.venv\Scripts\python.exe"
if not exist "%PYTHONEXE%" (
    echo Isaac Lab Python not found: %PYTHONEXE%
    exit /b 1
)
pushd "%~dp0.."
call "%ISAACSIM_ROOT%\python.bat" -m mentorpi_nav.cli.preview --mode smoke --num-envs 1 --viz kit --kit_args=--/app/vulkan=false %*
set "NAV_EXIT_CODE=%ERRORLEVEL%"
popd
exit /b %NAV_EXIT_CODE%
