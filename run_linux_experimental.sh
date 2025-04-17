#!/bin/bash

# --- Script Configuration ---
# Stop script on first error (optional but recommended)
# set -e

# Configuration Variables (Equivalent to 'set' in batch)
VENV_DIR=".venv"
REQUIREMENTS_FILE="requirements.txt"
MAIN_SCRIPT="main.py"
PYTHON_VERSION="3.10" # Python version UV should look for
# Options to FORCE initial installation (Note: Bash arrays or simple strings work)
# Using a simple string here for direct translation
UV_INSTALL_OPTIONS="--no-cache --no-deps --index-strategy unsafe-best-match"
# PyTorch packages - Bash arrays are better for lists
# TORCH_PACKAGES=("torch" "torchaudio") # Using array is more idiomatic in Bash
MODEL_DIR="modelTTS"
KOKORO_MODEL_URL="https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
KOKORO_MODEL_FILENAME="kokoro-v1.0.onnx"
KOKORO_VOICES_URL="https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"
KOKORO_VOICES_FILENAME="voices-v1.0.bin"

# --- Helper Functions for Error Handling ---
# Equivalent to :error_exit and :error_exit_script labels and exit /b
error_exit() {
  echo # Blank line
  echo "!!! SETUP FAILED !!!"
  # No 'endlocal' needed in bash
  exit 1
}

error_exit_script() {
  local exit_code=$1 # Get exit code passed to function
  echo # Blank line
  echo "!!! PYTHON SCRIPT FAILED (Code: $exit_code) !!!"
  exit "$exit_code"
}

# --- Main Script Logic ---

echo "============================================="
echo "== Simplified Launch Script (Linux)      =="
echo "============================================="
echo

# --- 1. Tools Check ---
echo "--- Checking Tools ---"

# Check for python (prefer python3 on Linux)
# Try python3 first, then python
PYTHON_CMD="python3"
if ! command -v $PYTHON_CMD &> /dev/null; then
    PYTHON_CMD="python"
    if ! command -v $PYTHON_CMD &> /dev/null; then
        echo "Error: python3 or python not found. Please install Python."
        error_exit
    fi
    echo "Info: Found 'python' instead of 'python3'. Will use 'python'."
fi
echo "Found Python command: $PYTHON_CMD"

# Check for uv
if ! command -v uv &> /dev/null; then
  echo "Info: uv not found. Attempting to install using curl..."
  # Check for curl first, as it's needed for the install
  if ! command -v curl &> /dev/null; then
    echo "Error: curl not found. Needed to install uv. Please install curl (e.g., sudo apt install curl)."
    error_exit
  fi
  # Install uv using the official shell script
  if curl -LsSf https://astral.sh/uv/install.sh | sh; then
    # Add uv to PATH for the current script execution
    # The installer script usually tells the user to update their shell profile,
    # but we need it *now* for this script session.
    export PATH="$HOME/.cargo/bin:$PATH"
    echo "Info: uv installed. Attempting to proceed."
    # Verify installation again
    if ! command -v uv &> /dev/null; then
        echo "Error: uv install seemed successful, but still not found in PATH."
        echo "Please ensure '$HOME/.cargo/bin' is in your PATH or restart your terminal and RERUN THE SCRIPT."
        error_exit
    fi
  else
    echo "Error: uv install failed. Please install uv manually from https://astral.sh/uv"
    error_exit
  fi
else
    echo "Found uv."
fi


# Check for curl (again, just in case install step was skipped but curl still needed)
if ! command -v curl &> /dev/null; then
  echo "Error: curl not found. Required for downloads. Please install it (e.g., sudo apt install curl)."
  error_exit
fi
echo "Tools found ($PYTHON_CMD, uv, curl)."
echo

# --- 2. Virtual Environment ---
echo "--- Virtual Environment ($VENV_DIR) ---"
ACTIVATE_SCRIPT="$VENV_DIR/bin/activate" # Linux activation script path

# Check if activation script exists (means venv likely exists)
if [ ! -f "$ACTIVATE_SCRIPT" ]; then
  echo "Creating env with Python $PYTHON_VERSION..."
  # Use uv to create the virtual environment
  # uv venv -p "$PYTHON_VERSION" "$VENV_DIR" # Use this if you want uv to find python 3.10
  uv venv --python "$PYTHON_CMD" "$VENV_DIR" # Use this to force using the found python command
  # Check the exit status of the last command ($?)
  if [ $? -ne 0 ]; then
    echo "Error: Virtual env creation failed."
    echo "Make sure Python $PYTHON_VERSION is installed and accessible,"
    echo "or that the detected '$PYTHON_CMD' is compatible."
    error_exit
  fi
else
  echo "Existing environment found."
fi

echo "Activating env..."
# Use 'source' (or '.') to activate the environment in the *current* shell
source "$ACTIVATE_SCRIPT"
if [ $? -ne 0 ]; then
  echo "Error activating virtual environment."
  error_exit
fi
echo "Env activated."
echo

# --- 3. Initial Dependencies Installation ---
echo "--- Installing Initial Dependencies ($REQUIREMENTS_FILE) ---"
if [ ! -f "$REQUIREMENTS_FILE" ]; then
  echo "Error: $REQUIREMENTS_FILE not found."
  # Deactivate venv before exiting if needed (optional, depends on desired state after failure)
  # deactivate &> /dev/null
  error_exit
fi

echo "Using forced install options: $UV_INSTALL_OPTIONS"
echo "WARNING: --no-deps skips transitive dependencies."
# Use the options string directly
uv pip install $UV_INSTALL_OPTIONS -r "$REQUIREMENTS_FILE"
if [ $? -ne 0 ]; then
  # Mimic the batch script's behavior of warning but continuing
  echo "WARNING: Error during forced install. Continuing as per original script logic..."
else
  echo "Initial dependencies installed (or forced attempt done)."
fi
echo

# --- 4. Downloading Models ---
echo "--- Downloading Models ($MODEL_DIR) ---"
TARGET_MODEL_PATH="$MODEL_DIR/$KOKORO_MODEL_FILENAME"
TARGET_VOICES_PATH="$MODEL_DIR/$KOKORO_VOICES_FILENAME"

# Check if model directory exists, create if not (-p creates parent dirs if needed and doesn't error if exists)
if [ ! -d "$MODEL_DIR" ]; then
  echo "Creating folder $MODEL_DIR..."
  mkdir -p "$MODEL_DIR"
  if [ $? -ne 0 ]; then
    echo "Error creating folder $MODEL_DIR."
    # deactivate &> /dev/null
    error_exit
  fi
fi

# Download ONNX model if it doesn't exist
if [ ! -f "$TARGET_MODEL_PATH" ]; then
  echo "Downloading $KOKORO_MODEL_FILENAME..."
  # curl options: -L follow redirects, -o output to file
  curl -L -o "$TARGET_MODEL_PATH" "$KOKORO_MODEL_URL"
  if [ $? -ne 0 ]; then
    echo "Error downloading $KOKORO_MODEL_FILENAME."
    # Attempt to remove partially downloaded file (-f ignores errors if file doesn't exist)
    rm -f "$TARGET_MODEL_PATH"
    # deactivate &> /dev/null
    error_exit
  fi
else
  echo "Model $KOKORO_MODEL_FILENAME already exists."
fi

# Download voices file if it doesn't exist
if [ ! -f "$TARGET_VOICES_PATH" ]; then
  echo "Downloading $KOKORO_VOICES_FILENAME..."
  curl -L -o "$TARGET_VOICES_PATH" "$KOKORO_VOICES_URL"
  if [ $? -ne 0 ]; then
    echo "Error downloading $KOKORO_VOICES_FILENAME."
    rm -f "$TARGET_VOICES_PATH"
    # deactivate &> /dev/null
    error_exit
  fi
else
  echo "File $KOKORO_VOICES_FILENAME already exists."
fi
echo "Models ready."
echo

# --- 4.5 Optional PyTorch Configuration ---
echo "--- Optional PyTorch Configuration ---"
echo
echo "PyTorch install options:"
echo "  - CUDA Version: Enter number WITHOUT dot (e.g., 118, 121, 124)"
echo "  - CPU Version: type 'cpu'"
echo "  - Skip/Default: type 'none' (or leave blank)"
echo

# Use 'read' command for user input
read -p "Enter your choice (e.g., 121, cpu, none): " CUDA_CHOICE

# Simple input cleanup (remove leading/trailing whitespace - optional)
# CUDA_CHOICE=$(echo "$CUDA_CHOICE" | xargs) # xargs trims whitespace

# Convert choice to lowercase for case-insensitive comparison
CUDA_CHOICE_LOWER=$(echo "$CUDA_CHOICE" | tr '[:upper:]' '[:lower:]')

# Use if/elif/else instead of goto
if [[ "$CUDA_CHOICE_LOWER" == "none" || -z "$CUDA_CHOICE_LOWER" || "$CUDA_CHOICE_LOWER" == "cpu" ]]; then
  echo "Choice '$CUDA_CHOICE'. No specific PyTorch CUDA reinstallation will be performed."
  # This branch does nothing regarding PyTorch re-installation, just proceeds.
else
  # Assume it's a CUDA version number if not none, blank, or cpu
  echo "Choice '$CUDA_CHOICE'. Attempting specific PyTorch CUDA reinstallation."
  CUDA_TAG="cu$CUDA_CHOICE" # Construct the tag (e.g., cu121)
  PYTORCH_INDEX_ARG="--index-url https://download.pytorch.org/whl/$CUDA_TAG"

  echo
  echo "--- SEPARATE PyTorch Reinstallation ($CUDA_TAG) ---"

  echo "Uninstalling torch (if present)..."
  # Use > /dev/null 2>&1 to silence output like in the batch script
  uv pip uninstall -y torch > /dev/null 2>&1
  echo "Uninstalling torchaudio (if present)..."
  uv pip uninstall -y torchaudio > /dev/null 2>&1

  echo
  echo "Installing torch from index $CUDA_TAG..."
  uv pip install "$PYTORCH_INDEX_ARG" torch
  if [ $? -ne 0 ]; then
    echo "Error installing torch for $CUDA_TAG ($PYTORCH_INDEX_ARG)"
    echo "Make sure '$CUDA_CHOICE' is a valid CUDA tag (e.g., 118, 121) and available on the PyTorch index."
    # deactivate &> /dev/null
    error_exit
  fi

  echo
  echo "Installing torchaudio from index $CUDA_TAG..."
  uv pip install "$PYTORCH_INDEX_ARG" torchaudio
  if [ $? -ne 0 ]; then
    echo "Error installing torchaudio for $CUDA_TAG ($PYTORCH_INDEX_ARG)"
    echo "Make sure it's compatible and available on the PyTorch index."
    # deactivate &> /dev/null
    error_exit
  fi

  echo "Specific torch and torchaudio reinstall completed for $CUDA_TAG."
fi
echo # Ensure a blank line after the PyTorch section

# --- 5. Main Script Launch ---
echo "--- Launching $MAIN_SCRIPT ---"
if [ ! -f "$MAIN_SCRIPT" ]; then
  echo "Error: $MAIN_SCRIPT not found."
  # deactivate &> /dev/null
  error_exit
fi

# Get the python executable from the virtual environment
PYTHON_EXEC="$VENV_DIR/bin/python"

echo "Running: $PYTHON_EXEC $MAIN_SCRIPT"
echo "--- Script Output Start ---"
# Execute the python script using the virtual environment's python
"$PYTHON_EXEC" "$MAIN_SCRIPT"
# Capture the exit code of the python script
SCRIPT_EXIT_CODE=$?
echo "--- Script Output End (Code: $SCRIPT_EXIT_CODE) ---"
echo

# Check the Python script's exit code
if [ "$SCRIPT_EXIT_CODE" -ne 0 ]; then
  # Use the dedicated error function for script failure
  # deactivate &> /dev/null
  error_exit_script "$SCRIPT_EXIT_CODE"
fi

echo "Completed successfully."

# Deactivate environment (optional, script exits anyway)
# deactivate &> /dev/null

# Exit with success code 0
exit 0