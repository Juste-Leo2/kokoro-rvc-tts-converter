@echo off
setlocal enabledelayedexpansion

:: --- Essential Configuration ---
set VENV_DIR=.venv
set REQUIREMENTS_FILE=requirements.txt
set MAIN_SCRIPT=main.py
set PYTHON_VERSION=3.10
:: Python version UV should look for ^^^
:: Options to FORCE initial installation
set UV_INSTALL_OPTIONS=--no-cache --no-deps --index-strategy unsafe-best-match
:: PyTorch packages to manage for reinstallation
set TORCH_PACKAGES=torch torchaudio
set MODEL_DIR=modelTTS
set KOKORO_MODEL_URL=https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
set KOKORO_MODEL_FILENAME=kokoro-v1.0.onnx
set KOKORO_VOICES_URL=https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
set KOKORO_VOICES_FILENAME=voices-v1.0.bin

echo =============================================
echo == Simplified Launch Script ==
echo =============================================
echo(

:: --- 1. Tools Check ---
echo --- Checking Tools ---
where python >nul 2>nul
if "%ERRORLEVEL%" NEQ "0" ( echo Error: python not found. Please install Python. & goto :error_exit )
where uv >nul 2>nul
if "%ERRORLEVEL%" NEQ "0" (
echo Info: uv not found. Attempting PowerShell install...
where powershell >nul 2>nul
if "%ERRORLEVEL%" NEQ "0" ( echo Error: PowerShell not found. Please install uv manually. & goto :error_exit )
powershell -NoProfile -ExecutionPolicy Bypass -Command "& {irm https://astral.sh/uv/install.ps1 | iex}"
if "%ERRORLEVEL%" NEQ "0" ( echo Error: uv install failed. Install manually. & goto :error_exit )
echo WARNING: RERUN THE SCRIPT if uv is still not found.
where uv >nul 2>nul
if "%ERRORLEVEL%" NEQ "0" ( echo Error: uv still not found after install. RERUN THE SCRIPT. & goto :error_exit )
)
where curl >nul 2>nul
if "%ERRORLEVEL%" NEQ "0" ( echo Error: curl not found. Required for downloads. Please install it. & goto :error_exit )
echo Tools found (python, uv, curl).
echo(

:: --- 2. Virtual Environment ---
echo --- Virtual Environment (%VENV_DIR%) ---
set ACTIVATE_SCRIPT=%VENV_DIR%\Scripts\activate.bat
if not exist "%ACTIVATE_SCRIPT%" (
echo Creating env with Python %PYTHON_VERSION%...
uv venv -p %PYTHON_VERSION% "%VENV_DIR%"
if "%ERRORLEVEL%" NEQ "0" ( echo Error: Virtual env creation failed. Make sure Python %PYTHON_VERSION% is installed. & goto :error_exit )
) else ( echo Existing environment found. )
echo Activating env...
call "%ACTIVATE_SCRIPT%"
if "%ERRORLEVEL%" NEQ "0" ( echo Error activating env. & goto :error_exit )
echo Env activated.
echo(

:: --- 3. Initial Dependencies Installation ---
echo --- Installing Initial Dependencies (%REQUIREMENTS_FILE%) ---
if not exist "%REQUIREMENTS_FILE%" ( echo Error: %REQUIREMENTS_FILE% not found. & goto :error_exit )
echo Using forced install options: %UV_INSTALL_OPTIONS%
echo WARNING: --no-deps skips transitive dependencies.
uv pip install %UV_INSTALL_OPTIONS% -r "%REQUIREMENTS_FILE%"
if "%ERRORLEVEL%" NEQ "0" (
echo WARNING: Error during forced install. Continuing...
) else (
echo Initial dependencies installed (or forced attempt done)
)
echo(

:: --- 4. Downloading Models ---
echo --- Downloading Models (%MODEL_DIR%) ---
set TARGET_MODEL_PATH=%MODEL_DIR%\%KOKORO_MODEL_FILENAME%
set TARGET_VOICES_PATH=%MODEL_DIR%\%KOKORO_VOICES_FILENAME%
if not exist "%MODEL_DIR%" (
echo Creating folder %MODEL_DIR%...
mkdir "%MODEL_DIR%"
if "%ERRORLEVEL%" NEQ "0" ( echo Error creating folder %MODEL_DIR%. & goto :error_exit )
)
if not exist "%TARGET_MODEL_PATH%" (
echo Downloading %KOKORO_MODEL_FILENAME%...
curl -L -o "%TARGET_MODEL_PATH%" "%KOKORO_MODEL_URL%"
if "%ERRORLEVEL%" NEQ "0" ( echo Error downloading %KOKORO_MODEL_FILENAME%. & del "%TARGET_MODEL_PATH%" 2>nul & goto :error_exit )
) else ( echo Model %KOKORO_MODEL_FILENAME% already exists. )
if not exist "%TARGET_VOICES_PATH%" (
echo Downloading %KOKORO_VOICES_FILENAME%...
curl -L -o "%TARGET_VOICES_PATH%" "%KOKORO_VOICES_URL%"
if "%ERRORLEVEL%" NEQ "0" ( echo Error downloading %KOKORO_VOICES_FILENAME%. & del "%TARGET_VOICES_PATH%" 2>nul & goto :error_exit )
) else ( echo File %KOKORO_VOICES_FILENAME% already exists. )
echo Models ready.
echo(

:: --- 4.5 Optional PyTorch Configuration ---
echo --- Optional PyTorch Configuration ---
echo(
echo PyTorch install options:
echo   - CUDA Version: Enter number WITHOUT dot (e.g., 118, 121, 124)
echo   - CPU or Skip: type cpu or none (or leave blank)
echo(
set CUDA_CHOICE=
set /p CUDA_CHOICE="Enter your choice (e.g., 121, cpu, none): "

:: Simple input cleanup
for /f "tokens=* delims= " %%a in ("%CUDA_CHOICE%") do set CUDA_CHOICE=%%a

:: --- PyTorch Logic with GOTO ---
if /i "%CUDA_CHOICE%"=="none" (
echo Choice 'none' No specific reinstallation
goto :PyTorchConfigDone
)
if /i "%CUDA_CHOICE%"=="" (
echo Blank choice No specific reinstallation
goto :PyTorchConfigDone
)
if /i "%CUDA_CHOICE%"=="cpu" (
echo Choice 'cpu' No specific reinstallation
goto :PyTorchConfigDone
)

:: If we are here, it means CUDA reinstallation is requested
echo Choice '%CUDA_CHOICE%' Attempting CUDA reinstall
set CUDA_TAG=cu%CUDA_CHOICE%
set PYTORCH_INDEX_ARG=--index-url https://download.pytorch.org/whl/!CUDA_TAG!

echo(
echo --- SEPARATE PyTorch Reinstallation (!CUDA_TAG!) ---

echo Uninstalling torch (if present)...
uv pip uninstall -y torch > nul 2>&1
echo Uninstalling torchaudio (if present)...
uv pip uninstall -y torchaudio > nul 2>&1

echo(
echo Installing torch from index !CUDA_TAG!...
uv pip install !PYTORCH_INDEX_ARG! torch
if !ERRORLEVEL! NEQ 0 (
echo Error installing torch for !CUDA_TAG! (^!PYTORCH_INDEX_ARG!^)
echo Make sure '%CUDA_CHOICE%' is a valid CUDA tag and that it's available on the PyTorch index.
goto :error_exit
)

echo(
echo Installing torchaudio from index !CUDA_TAG!...
uv pip install !PYTORCH_INDEX_ARG! torchaudio
if !ERRORLEVEL! NEQ 0 (
echo Error installing torchaudio for !CUDA_TAG! (^!PYTORCH_INDEX_ARG!^)
echo Make sure it's compatible and available on the PyTorch index.
goto :error_exit
)

echo Specific torch and torchaudio reinstall completed.

:PyTorchConfigDone
echo(

:: --- 5. Main Script Launch ---
echo --- Launching %MAIN_SCRIPT% ---
if not exist "%MAIN_SCRIPT%" ( echo Error: %MAIN_SCRIPT% not found. & goto :error_exit )
echo Running: %VENV_DIR%\Scripts\python.exe %MAIN_SCRIPT%
echo --- Script Output Start ---
%VENV_DIR%\Scripts\python.exe "%MAIN_SCRIPT%"
set SCRIPT_EXIT_CODE=%ERRORLEVEL%
echo --- Script Output End (Code: %SCRIPT_EXIT_CODE%) ---
echo(
if "%SCRIPT_EXIT_CODE%" NEQ "0" ( echo Python Script Failed (Code: %SCRIPT_EXIT_CODE%). & goto :error_exit_script )
echo Completed successfully.
goto :success_exit

:error_exit
echo(
echo !!! SETUP FAILED !!!
endlocal
exit /b 1

:error_exit_script
echo(
echo !!! PYTHON SCRIPT FAILED !!!
endlocal
exit /b %SCRIPT_EXIT_CODE%

:success_exit
endlocal
exit /b 0
