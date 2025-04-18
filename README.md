# Kokoro TTS and RVC Text-to-Speech Converter 🎙️🧠

[![Français](https://img.shields.io/badge/Langue-Français-blue.svg)](./READMEFR.md)
[![English](https://img.shields.io/badge/Language-English-green.svg)](./README.md)

## Description

This project is a text-to-speech converter using:

- **Kokoro TTS** for high-quality speech synthesis 🗣️
- **RVC** (Retrieval-Based Voice Conversion) for flexible voice transformations 🎛️

The interface runs through **Gradio** and opens via a link in your terminal 🌐

## What's New

- ✅ Full support for all Kokoro TTS 1.0 languages
- ⚡ One-step installation using `run_windows.bat`
- 🔁 Auto-setup with `uv`: virtual environment, dependencies, models — all handled for you!

## Installation (Windows)

1. Install [**Python 3.10**](https://www.python.org/downloads/release/python-3100/) and add it to your system `PATH`
2. Download or clone this repository
3. **Install UV manually**:
   
   Open **Command Prompt** or **PowerShell** and run:
   ```bash
   pip install uv
4. Run `run_windows.bat` — that's it!


## Installation (Linux)

1. **Download or clone this repository**.

2. **Open terminal and run in the project folder**:
   ```bash
   chmod +x run_linux_experimental.sh


### these scripts will:
- Set up a virtual environment
- Install all required packages
- Download Kokoro TTS models into `modelTTS/`


## Using RVC Models 🎧

1. Download `.pth` and `.index` files from [weights.gg](https://www.weights.gg/)
2. Place:
   - `.pth` files in `modelRVC/pth/`
   - `.index` files in `modelRVC/index/`

## Credits 🙏

Special thanks to:

- [kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx) — ONNX version of the Kokoro TTS model
- [Hexgrad](https://huggingface.co/hexgrad) — Original author of Kokoro TTS (Kokoro-82M)
- [rvc-python](https://github.com/daswer123/rvc-python) — RVC implementation and tooling