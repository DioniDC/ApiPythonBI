@echo off
setlocal ENABLEDELAYEDEXPANSION

REM ===================== CONFIG =====================
set "DOCKER_USER=dionidias"
set "IMAGE_NAME=sgbi-api"
set "PLATFORMS=linux/amd64,linux/arm64"
set "BUILDER_NAME=multiarch-builder"
set "BUILD_CONTEXT=."
set "DEFAULT_VERSION=2.0.0"
set "LOGIN_INTERATIVO=1"
REM Para login automatico, antes de rodar:
REM   set DOCKERHUB_USERNAME=seu_usuario
REM   set DOCKERHUB_TOKEN=seu_token
REM   set LOGIN_INTERATIVO=0

REM ===================== VERSION =====================
set "VERSION="
if exist "VERSION" for /f "usebackq tokens=* delims=" %%v in ("VERSION") do set "VERSION=%%v"
if "%VERSION%"=="" for /f "delims=" %%v in ('git describe --tags --abbrev^=0 2^>NUL') do set "VERSION=%%v"
if "%VERSION%"=="" set "VERSION=%DEFAULT_VERSION%"
for /f "tokens=* delims= " %%v in ("%VERSION%") do set "VERSION=%%v"
if "%VERSION%"=="" (
  echo ERRO: falha ao determinar VERSION.
  exit /b 1
)

REM ===================== AJUSTES =====================
set "IMAGE=%DOCKER_USER%/%IMAGE_NAME%"
cd /d "%~dp0"

echo.
echo Publicando %IMAGE%:%VERSION% e %IMAGE%:latest
echo Contexto: %BUILD_CONTEXT%
echo Plataformas: %PLATFORMS%
echo.

REM -------- Docker Desktop / daemon --------
set "DOCKER_HOST="
if exist "C:\Program Files\Docker\Docker\Docker Desktop.exe" (
  start "" "C:\Program Files\Docker\Docker\Docker Desktop.exe"
)

echo Aguardando Docker daemon iniciar...
set /a __tries=0
:wait_docker
docker info >NUL 2>&1
if errorlevel 1 (
  set /a __tries+=1
  if %__tries% GEQ 90 (
    echo ERRO: Docker daemon nao iniciou em tempo habil.
    exit /b 1
  )
  timeout /t 1 >NUL
  goto :wait_docker
)
echo Docker OK.

REM -------- Contexto default --------
docker context use default >NUL 2>&1

REM -------- Login Docker Hub --------
echo.
echo docker login
if "%LOGIN_INTERATIVO%"=="0" (
  if not defined DOCKERHUB_USERNAME (
    echo ERRO: LOGIN_INTERATIVO=0 mas DOCKERHUB_USERNAME nao definido.
    exit /b 1
  )
  if not defined DOCKERHUB_TOKEN (
    echo ERRO: LOGIN_INTERATIVO=0 mas DOCKERHUB_TOKEN nao definido.
    exit /b 1
  )
  echo Fazendo login com variaveis de ambiente...
  echo %DOCKERHUB_TOKEN% | docker login -u "%DOCKERHUB_USERNAME%" --password-stdin
) else (
  docker login
)
if errorlevel 1 (
  echo ERRO no docker login.
  exit /b 1
)

REM -------- buildx builder --------
echo.
echo Preparando buildx
docker buildx inspect "%BUILDER_NAME%" >NUL 2>&1
if errorlevel 1 (
  docker buildx create --name "%BUILDER_NAME%" --use >NUL
) else (
  docker buildx use "%BUILDER_NAME%" >NUL
)
docker buildx inspect --bootstrap >NUL

REM -------- Build e Push --------
echo.
echo Build+Push %IMAGE%:%VERSION% e :latest
docker buildx build --platform %PLATFORMS% --provenance=false --pull -t %IMAGE%:%VERSION% -t %IMAGE%:latest "%BUILD_CONTEXT%" --push
if errorlevel 1 (
  echo FALHA no build/push.
  exit /b 1
)

echo.
echo OK: publicado %IMAGE%:%VERSION% e %IMAGE%:latest
exit /b 0
