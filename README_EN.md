# Kokoro TTS and RVC Text-to-Audio Converter

[![Français](https://img.shields.io/badge/Langue-Français-blue.svg)](./README_FR.md)
[![English](https://img.shields.io/badge/Language-English-green.svg)](./README_EN.md)

[**Lire ce README en Français / Read this README in French**](./README_FR.md)

## Description

This project is a text-to-audio converter that leverages two powerful technologies:

*   **Kokoro TTS (Text-to-Speech):** For high-quality speech synthesis from text.
*   **RVC (Retrieval-Based Voice Conversion):** To convert the speech synthesized by Kokoro TTS into the voice of different characters or voice styles, using pre-trained models.

The user interface is built with Gradio, providing an interactive and easy-to-use experience directly in your web browser.

## Installation

Follow these steps to install and run the text-to-audio converter:

1.  **Download the project:**
    *   Download the ZIP file of the GitHub repository.
2.  **Extract the archive:**
    *   Extract the contents of the ZIP file to a folder of your choice.
3.  **Create an Anaconda environment:**
    *   If you do not have Anaconda installed, download and install it from [the official Anaconda website](https://www.anaconda.com/products/distribution).
    *   Open Anaconda Prompt or your terminal.
    *   Create a new Anaconda environment named `kokoro_RVC` with Python 3.10 using the following command:
        ```bash
        conda create -n kokoro_RVC python=3.10
        ```
4.  **Activate the environment:**
    *   Activate the `kokoro_RVC` environment with the command:
        ```bash
        conda activate kokoro_RVC
        ```
5.  **Navigate to the project folder:**
    *   Use the `cd` command to navigate to the folder where you extracted the project. For example:
        ```bash
        cd path/to/your/folder/kokoro-rvc-converter
        ```
6.  **Install dependencies:**
    *   Run the following command to install all necessary Python libraries. Make sure you are still in the `kokoro_RVC` environment and in the project folder:
        ```bash
        python.exe -m pip install -r requirement1.txt && pip install -r requirement2.txt
        ```
    *   **Note:** If you are on a non-Windows system and `python.exe` does not work, try simply `python -m pip install -r requirement1.txt && pip install -r requirement2.txt`.
7.  **Run the application:**
    *   Launch the application by running the command:
        ```bash
        python main.py
        ```
    *   The Gradio interface will automatically open in your web browser.

## Usage

Once the application is running, you can use the Gradio interface to:

*   **Synthesize speech with Kokoro TTS** from text.
*   **Convert voices** using RVC models, applying pitch shift.
*   **Combine both** by generating TTS speech and converting it to RVC voice in one step.

Have fun with the text-to-audio converter!

## Sources/Credits

This project utilizes the following resources:

*   **Kokoro-onnx:** [https://github.com/thewh1teagle/kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx) - ONNX implementation of the Kokoro TTS model.
*   **Kokoro-82M version 1.0:** [https://huggingface.co/hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) - Pre-trained Kokoro TTS model on Hugging Face.
*   **rvc-python:** [https://github.com/daswer123/rvc-python](https://github.com/daswer123/rvc-python) - Python library for RVC voice conversion.

---

Feel free to contribute to this project or report issues.
