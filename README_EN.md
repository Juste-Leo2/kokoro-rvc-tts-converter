# Kokoro TTS and RVC Text-to-Speech Converter

[![Français](https://img.shields.io/badge/Langue-Français-blue.svg)](./README.md)
[![English](https://img.shields.io/badge/Language-English-green.svg)](./README_EN.md)

[**Lire ce README en Français / Read this README in French**](./README.md)

## Description

This project is a text-to-speech converter that utilizes two powerful technologies:

*   **Kokoro TTS (Text-to-Speech):** For high-quality speech synthesis from text.
*   **RVC (Retrieval-Based Voice Conversion):** To convert the speech synthesized by Kokoro TTS into the voice of different characters or vocal styles, using pre-trained models.

The user interface is built with Gradio, offering an interactive and user-friendly experience directly in your web browser.

## Download Pre-trained Models (Required)

**Before proceeding, you need to download the pre-trained models necessary for the converter to function.** These models include `.bin`, `.onnx`, `.index`, and `.pth` files and are essential for both Kokoro TTS and RVC.

You can download them from Google Drive:

[**Download Pre-trained Models (Google Drive)**](https://drive.google.com/drive/folders/1G6O0FgyFdwVjn3rMoJZRbg6gC1OrkaI9?usp=sharing)

**Once downloaded, make sure to place the files correctly in the appropriate folders of your project.** Typically, `.pth` and `.index` files for RVC go into `modelRVG/pth/` and `modelRVG/index/` respectively, and Kokoro TTS models (.bin, .onnx) into the `model_tts`
## Installation

Follow these steps to install and run the text-to-speech converter:

1.  **Install Visual Studio Installer (Required for Windows):**
    *   If you are on Windows, it is necessary to install Visual Studio Installer to compile some dependencies.
    *   Download Visual Studio Installer from [the official Microsoft website](https://visualstudio.microsoft.com/fr/downloads/). Choose to download "Visual Studio Community".
    *   Run the downloaded installer program.
    *   In the Visual Studio Installer, when selecting workloads, **check the box** `✅ Desktop development with C++` (you may need to scroll down to find it).
    *   Continue the installation by following the on-screen instructions. You do not need to select other workloads or components for this project.
2.  **Download the project:**
    *   Download the ZIP file of the GitHub repository.
3.  **Extract the archive:**
    *   Extract the contents of the ZIP file into a folder of your choice.
4.  **Create an Anaconda environment:**
    *   If you do not have Anaconda installed, download and install it from [the official Anaconda website](https://www.anaconda.com/products/distribution).
    *   Open Anaconda Prompt or your terminal.
    *   Create a new Anaconda environment named `kokoro_RVC` with Python 3.10 using the following command:
        ```bash
        conda create -n kokoro_RVC python=3.10
        ```
5.  **Activate the environment:**
    *   Activate the `kokoro_RVC` environment with the command:
        ```bash
        conda activate kokoro_RVC
        ```
6.  **Navigate to the project folder:**
    *   Use the `cd` command to move to the folder where you extracted the project. For example:
        ```bash
        cd path/to/your/folder/kokoro-rvc-converter
        ```
7.  **Install dependencies:**
    *   Run the following command to install all the necessary Python libraries. Make sure you are still in the `kokoro_RVC` environment and in the project folder:
        ```bash
        python.exe -m pip install -r requirement1.txt && pip install -r requirement2.txt
        ```
    *   **Note:** If you are on a system other than Windows and `python.exe` does not work, try simply `python -m pip install -r requirement1.txt && pip install -r requirement2.txt`.
8.  **Run the application:**
    *   Launch the application by running the command:
        ```bash
        python main.py
        ```
    *   The Gradio interface will automatically open in your web browser.

## Usage

Once the application is launched, you can use the Gradio interface to:

*   **Synthesize speech with Kokoro TTS** from text.
*   **Convert voices** using RVC models, applying pitch shift.
*   **Combine both** by generating TTS speech and converting it to RVC voices in a single step.

Have fun with the text-to-speech converter!

## RVC Voice Models

You can find a wide variety of pre-trained RVC voice models on [weights.gg](https://www.weights.gg/).

To use these models with this converter:

1.  **Download** the `.pth` and `.index` files of the RVC voice model you want to use from [weights.gg](https://www.weights.gg/).
2.  **Place** the `.pth` file in the `modelRVG/pth/` folder of your project. Create the `pth` folder if it does not already exist.
3.  **Place** the `.index` file in the `modelRVG/index/` folder of your project. Create the `index` folder if it does not already exist.
4.  **Restart** the application `python main.py`. The new RVC voice models should now appear in the dropdown lists of the Gradio interface.

## Sources

This project uses the following resources:

*   **Kokoro-onnx:** [https://github.com/thewh1teagle/kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx) - ONNX implementation of the Kokoro TTS model.
*   **Kokoro-82M version 1.0:** [https://huggingface.co/hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) - Pre-trained Kokoro TTS model on Hugging Face.
*   **rvc-python:** [https://github.com/daswer123/rvc-python](https://github.com/daswer123/rvc-python) - Python library for RVC voice conversion.

---

Feel free to contribute to this project or report issues.
