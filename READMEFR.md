# Convertisseur Text-to-Speech avec Kokoro TTS et RVC 🎙️🧠

[![Français](https://img.shields.io/badge/Langue-Français-blue.svg)](./READMEFR.md)
[![English](https://img.shields.io/badge/Language-English-green.svg)](./README.md)

## Description

Ce projet est un convertisseur texte-vers-parole basé sur :

- **Kokoro TTS** pour une synthèse vocale de haute qualité 🗣️  
- **RVC** (Retrieval-Based Voice Conversion) pour modifier la voix générée 🎛️

L'interface utilise **Gradio** et s’ouvre via un lien cliquable dans votre terminal 🌐

## Nouveautés

- ✅ Prise en charge complète de toutes les langues de Kokoro TTS 1.0  
- ⚡ Installation simplifiée via `run_windows.bat`  
- 🛠️ L'environnement virtuel, les dépendances et les modèles sont automatiquement configurés avec `uv`

## Installation (Windows)

1. Installez [**Python 3.10**](https://www.python.org/downloads/release/python-3100/) et ajoutez-le à votre `PATH` système.
2. Téléchargez ou clonez ce dépôt.
3. **Installez UV manuellement** :
   
   Ouvrez **Invite de commandes** ou **PowerShell** et exécutez :
   ```bash
   pip install uv
   ```
4. Exécutez `run_windows.bat` — et c'est tout !

## Installation (Linux)

1. **Téléchargez ou clonez ce dépôt**.

2. **Ouvrez le terminal et exécutez dans le dossier du projet** :
   ```bash
   chmod +x run_linux_experimental.sh
   ```

### Ces scripts vont :
- Configurer un environnement virtuel
- Installer tous les packages requis
- Télécharger les modèles Kokoro TTS dans le dossier `modelTTS/`

## Utiliser des modèles RVC 🎧

1. Téléchargez les fichiers `.pth` et `.index` depuis [weights.gg](https://www.weights.gg/)
2. Placez :
   - les fichiers `.pth` dans `modelRVC/pth/`
   - les fichiers `.index` dans `modelRVC/index/`

## Remerciements 🙏

Merci à :

- [kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx) — version ONNX de Kokoro TTS  
- [Hexgrad](https://huggingface.co/hexgrad) — auteur du modèle Kokoro TTS (Kokoro-82M)  
- [rvc-python](https://github.com/daswer123/rvc-python) — implémentation Python de RVC