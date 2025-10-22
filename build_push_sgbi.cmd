@echo off
setlocal ENABLEDELAYEDEXPANSION

REM ======= CONFIGURE APENAS UMA VEZ =======
set DOCKER_USER=dionidias
set IMAGE_NAME=sgbi-api

REM ======= DETERMINA VERSAO =======
set "VERSION=2.0.0"

REM 1) Se existir arquivo VERSION na raiz, usa ele
if exist "VERSION" (
  for /f "usebackq tokens=* delims=" %%v in ("VERSION") do set "VERSION=%%v"
)

REM 2) Se ainda vazio, tenta git describe --tags (se for repo git)
if "%VERSION%"=="" (
  for /f "delims=" %%v in ('git describe --tags --abbrev^=0 2^>NUL') do set "VERSION=%%v"
)

REM 3) Se ainda vazio, gera timestamp imune a locale (yyyyMMdd-HHmm)
if "%VERSION%"=="" (
  for /f "delims=" %%v in ('powershell -NoProfile -Command "$ts=(Get-Date).ToString(''yyyyMMdd-HHmm''); Write-Output $ts"') do set "VERSION=%%v"
)

REM Sanitiza espaços
for /f "tokens=* delims= " %%v in ("%VERSION%") do set "VERSION=%%v"

REM Valida
if "%VERSION%"=="" (
  echo ERRO: Nao foi possivel definir VERSION.
  exit /b 1
)

REM ======= NAO ALTERAR DAQUI PRA BAIXO =======
set IMAGE=%DOCKER_USER%/%IMAGE_NAME%

cd /d "%~dp0"

echo.
echo === docker login (pede so se precisar) ===
docker login
if errorlevel 1 (
  echo ERRO no docker login
  exit /b 1
)

echo.
echo === preparando buildx ===
docker buildx inspect >NUL 2>&1
if errorlevel 1 (
  docker buildx create --name multiarch-builder --use
)
docker buildx inspect --bootstrap >NUL

echo.
echo === BUILD/PUSH %IMAGE%:%VERSION% e :latest ===
docker buildx build ^
  --platform linux/amd64,linux/arm64 ^
  -t %IMAGE%:%VERSION% ^
  -t %IMAGE%:latest ^
  --push .

if errorlevel 1 (
  echo FALHA no build/push
  exit /b 1
)

echo.
echo OK: publicado %IMAGE%:%VERSION% e %IMAGE%:latest
exit /b 0
