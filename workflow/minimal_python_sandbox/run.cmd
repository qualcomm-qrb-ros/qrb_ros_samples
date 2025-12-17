@echo off
setlocal enabledelayedexpansion

set "IMG=test_docker_exec:latest"
set "NAME=test_docker_exec"

REM mkdir -p _sandbox
if not exist "_sandbox" mkdir "_sandbox"
if errorlevel 1 exit /b %errorlevel%

REM chmod +x _sandbox/runner.sh
REM Windows 无 chmod；建议在 Dockerfile 里 chmod 或容器内用 sh 执行 runner.sh

REM Build image
docker build -t "%IMG%" .
if errorlevel 1 exit /b %errorlevel%

REM Run container, mounting host ./_sandbox to container /usr/local/sandbox
docker run --rm --name "%NAME%" -v "%cd%/_sandbox:/usr/local/sandbox" "%IMG%"
if errorlevel 1 exit /b %errorlevel%