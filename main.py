# -*- coding: utf-8 -*-
import gradio as gr
import soundfile as sf
from kokoro_onnx import Kokoro
import torch
from pathlib import Path
import os
import numpy as np
from rvc_python.infer import RVCInference
import tempfile
import logging
import traceback

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constantes et Configuration ---
RVC_MODEL_DIR = Path("modelRVC")
RVC_INDEX_DIR = RVC_MODEL_DIR / "index"
RVC_PTH_DIR = RVC_MODEL_DIR / "pth"
KOKORO_ONNX_PATH = "modelTTS/kokoro-v1.0.onnx"
KOKORO_VOICES_PATH = "modelTTS/voices-v1.0.bin"
DEFAULT_UI_LANGUAGE = "en" # Default language set here

# --- Variables Globales ---
AVAILABLE_INDEX_FILES = []
AVAILABLE_PTH_FILES = []
RVC_CACHE = {}
kokoro = None

# ==================================================
#           TEXTES DE L'INTERFACE (FR/EN)
# ==================================================
# (Dictionnaire UI_TEXTS reste le même qu'avant)
UI_TEXTS = {
    "fr": {
        "app_title": "Convertisseur Audio Texte-Parole (Kokoro TTS + RVC)",
        "app_description": "Générez de la parole à partir de texte avec Kokoro TTS, puis convertissez la voix avec des modèles RVC.",
        "refresh_button": "🔄 Actualiser les Listes de Modèles RVC",
        "lang_select_label": "Langue de l'Interface",
        # Tab 1: TTS
        "tab1_title": "1. Texte -> Parole (Kokoro TTS)",
        "tts_text_input_label": "Texte à Synthétiser",
        "tts_text_input_placeholder": "Entrez votre texte ici...",
        "tts_voice_select_label": "Voix Kokoro TTS",
        "tts_language_select_label": "Langue (peut affecter la prononciation)",
        "tts_speed_slider_label": "Vitesse de Parole",
        "tts_generate_button": "🔊 Générer Audio TTS",
        "tts_output_audio_label": "Audio TTS Généré",
        # Tab 2: RVC
        "tab2_title": "2. Conversion de Voix (RVC)",
        "rvc_input_audio_label": "Audio d'Entrée pour RVC",
        "rvc_input_audio_info": "👆 Utilisez l'audio généré à l'étape 1 OU téléchargez/enregistrez un fichier.",
        "rvc_pth_select_label": "Modèle de Voix RVC (.pth)",
        "rvc_pth_select_info": "Choisissez le fichier .pth de la voix cible.",
        "rvc_index_select_label": "Fichier Index RVC (.index)",
        "rvc_index_select_info": "Optionnel mais recommandé pour la qualité. Doit correspondre au .pth.",
        "rvc_pitch_slider_label": "Décalage de Hauteur (demi-tons)",
        "rvc_generate_button": "🗣️ Convertir la Voix RVC",
        "rvc_output_audio_label": "Audio Converti par RVC",
        # Tab 3: Text -> RVC
        "tab3_title": "3. Texte -> Voix RVC (Direct)",
        "ttrvc_description": "Combine les étapes 1 et 2 : Génère la parole avec Kokoro puis la convertit avec RVC.",
        "ttrvc_text_input_label": "Texte à Convertir",
        "ttrvc_text_input_placeholder": "Entrez votre texte ici...",
        "ttrvc_rvc_pth_select_label": "Modèle de Voix RVC (.pth)",
        "ttrvc_rvc_index_select_label": "Fichier Index RVC (.index)",
        "ttrvc_pitch_slider_label": "Décalage de Hauteur (demi-tons)",
        "ttrvc_kokoro_voice_label": "Voix Kokoro TTS (Base)",
        "ttrvc_kokoro_voice_info": "Voix utilisée pour générer la parole initiale avant conversion RVC.",
        "ttrvc_language_select_label": "Langue Kokoro",
        "ttrvc_speed_slider_label": "Vitesse de Parole (Base)",
        "ttrvc_generate_button": "🚀 Générer Audio RVC depuis Texte",
        "ttrvc_output_audio_label": "Audio RVC Final",
        # Erreurs (utilisées dans le backend, la langue est fixée au moment de l'erreur)
        "error_kokoro_not_loaded": "Erreur: Le modèle Kokoro TTS n'est pas chargé.",
        "error_tts_failed": "Erreur Kokoro TTS: {e}",
        # ... (autres textes d'erreur et warnings comme avant) ...
    },
    "en": {
        "app_title": "Text-to-Speech Audio Converter (Kokoro TTS + RVC)",
        "app_description": "Generate speech from text using Kokoro TTS, then convert the voice using RVC models.",
        "refresh_button": "🔄 Refresh RVC Model Lists",
        "lang_select_label": "Interface Language",
        # Tab 1: TTS
        "tab1_title": "1. Text -> Speech (Kokoro TTS)",
        "tts_text_input_label": "Text to Synthesize",
        "tts_text_input_placeholder": "Enter your text here...",
        "tts_voice_select_label": "Kokoro TTS Voice",
        "tts_language_select_label": "Language (may affect pronunciation)",
        "tts_speed_slider_label": "Speech Speed",
        "tts_generate_button": "🔊 Generate TTS Audio",
        "tts_output_audio_label": "Generated TTS Audio",
        # Tab 2: RVC
        "tab2_title": "2. Voice Conversion (RVC)",
        "rvc_input_audio_label": "Input Audio for RVC",
        "rvc_input_audio_info": "👆 Use the audio generated in step 1 OR upload/record a file.",
        "rvc_pth_select_label": "RVC Voice Model (.pth)",
        "rvc_pth_select_info": "Choose the target voice's .pth file.",
        "rvc_index_select_label": "RVC Index File (.index)",
        "rvc_index_select_info": "Optional but recommended for quality. Should match the .pth file.",
        "rvc_pitch_slider_label": "Pitch Shift (semitones)",
        "rvc_generate_button": "🗣️ Convert RVC Voice",
        "rvc_output_audio_label": "Converted RVC Audio",
        # Tab 3: Text -> RVC
        "tab3_title": "3. Text -> RVC Voice (Direct)",
        "ttrvc_description": "Combines steps 1 and 2: Generates speech with Kokoro then converts it with RVC.",
        "ttrvc_text_input_label": "Text to Convert",
        "ttrvc_text_input_placeholder": "Enter your text here...",
        "ttrvc_rvc_pth_select_label": "RVC Voice Model (.pth)",
        "ttrvc_rvc_index_select_label": "RVC Index File (.index)",
        "ttrvc_pitch_slider_label": "Pitch Shift (semitones)",
        "ttrvc_kokoro_voice_label": "Kokoro TTS Voice (Base)",
        "ttrvc_kokoro_voice_info": "Voice used to generate the initial speech before RVC conversion.",
        "ttrvc_language_select_label": "Kokoro Language",
        "ttrvc_speed_slider_label": "Speech Speed (Base)",
        "ttrvc_generate_button": "🚀 Generate RVC Audio from Text",
        "ttrvc_output_audio_label": "Final RVC Audio",
        # Errors (used in backend, language is fixed when error occurs)
        "error_kokoro_not_loaded": "Error: Kokoro TTS model is not loaded.",
        "error_tts_failed": "Kokoro TTS Error: {e}",
        # ... (other error texts and warnings as before) ...
    }
}

# --- Fonctions Utilitaires (inchangées) ---
def scan_rvc_models(log_results=True):
    # ... (même code qu'avant) ...
    global AVAILABLE_INDEX_FILES, AVAILABLE_PTH_FILES
    logger.info(f"Scanning RVC model directories: {RVC_PTH_DIR} and {RVC_INDEX_DIR}")
    found_pth = []
    found_index = []
    try:
        RVC_INDEX_DIR.mkdir(parents=True, exist_ok=True)
        RVC_PTH_DIR.mkdir(parents=True, exist_ok=True)
        found_pth = sorted([f for f in os.listdir(RVC_PTH_DIR) if f.endswith(".pth")])
        found_index = sorted([f for f in os.listdir(RVC_INDEX_DIR) if f.endswith(".index")])
    except Exception as e:
        logger.error(f"Error during RVC model scan: {e}", exc_info=True)
    AVAILABLE_PTH_FILES = found_pth
    AVAILABLE_INDEX_FILES = found_index
    if log_results:
        logger.info(f"Scan complete. Found PTH files: {len(AVAILABLE_PTH_FILES)}")
        logger.info(f"Scan complete. Found Index files: {len(AVAILABLE_INDEX_FILES)}")
        if not AVAILABLE_PTH_FILES: logger.warning("No .pth model files found.")

def create_silent_audio(sample_rate=44100):
    # ... (même code qu'avant) ...
    return (sample_rate, np.array([], dtype=np.float32))

def refresh_models_list():
    # ... (même code qu'avant) ...
    logger.info("Refreshing RVC model lists for Tabs 2 & 3...")
    scan_rvc_models(log_results=True)
    return (
        gr.update(choices=AVAILABLE_PTH_FILES),   # rvc_pth_select (Tab 2)
        gr.update(choices=AVAILABLE_INDEX_FILES), # rvc_index_select (Tab 2)
        gr.update(choices=AVAILABLE_PTH_FILES),   # ttrvc_rvc_pth_select (Tab 3)
        gr.update(choices=AVAILABLE_INDEX_FILES)  # ttrvc_rvc_index_select (Tab 3)
    )

# --- Initialisation des modèles au démarrage (inchangée) ---
scan_rvc_models()
try:
    # ... (Initialisation Kokoro) ...
    if os.path.exists(KOKORO_ONNX_PATH) and os.path.exists(KOKORO_VOICES_PATH):
        kokoro = Kokoro(KOKORO_ONNX_PATH, KOKORO_VOICES_PATH)
        logger.info("Kokoro TTS model initialized successfully.")
    else:
        logger.error(f"Kokoro TTS files not found. Check paths: {KOKORO_ONNX_PATH}, {KOKORO_VOICES_PATH}")
        kokoro = None
except Exception as e:
    logger.error(f"Critical error during Kokoro TTS initialization: {e}", exc_info=True)
    kokoro = None

# --- Préparation des listes pour Gradio (Noms de voix/langues, inchangé) ---
# ... (VOICES_KOKORO_RAW, LANGUAGES_KOKORO, COUNTRY_FLAGS, VOICES_KOKORO, LANGUAGES_KOKORO_CHOICES) ...
VOICES_KOKORO_RAW = [
    "af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica", "af_kore", "af_nicole",
    "af_nova", "af_river", "af_sarah", "af_sky", "am_adam", "am_echo", "am_eric",
    "am_fenrir", "am_liam", "am_michael", "am_onyx", "am_puck", "bf_alice",
    "bf_emma", "bf_isabella", "bf_lily", "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
    "ff_siwis", "if_sara", "im_nicola", "jf_alpha", "jf_gongitsune", "jf_nezumi",
    "jf_tebukuro", "jm_kumo", "zf_xiaobei",
    "zf_xiaoni", "zf_xiaoxiao", "zf_xiaoyi"
]
LANGUAGES_KOKORO = {
    "fr-fr": "Français", "it": "Italiano", "en-us": "English (US)", "en-gb": "English (GB)",
    "ja": "日本語", "cmn": "中文", "pt-br": "Português (BR)", "es": "Español"
}
COUNTRY_FLAGS = { "af": "🇺🇸", "am": "🇺🇸", "bf": "🇬🇧", "bm": "🇬🇧", "ff": "🇫🇷", "if": "🇮🇹", "im": "🇮🇹", "jf": "🇯🇵", "jm": "🇯🇵", "zf": "🇨🇳" }
VOICES_KOKORO = []
for voice_code in VOICES_KOKORO_RAW:
    country_code = voice_code[:2]
    flag = COUNTRY_FLAGS.get(country_code, "🌐")
    lang_key_fr = f"{country_code}-fr"; lang_key_base = country_code
    lang_name = LANGUAGES_KOKORO.get(lang_key_fr, LANGUAGES_KOKORO.get(lang_key_base, country_code.upper()))
    display_name = f"{flag} {voice_code.split('_')[1].capitalize()} ({lang_name})"
    VOICES_KOKORO.append((display_name, voice_code))
LANGUAGES_KOKORO_CHOICES = [(name, code) for code, name in LANGUAGES_KOKORO.items()]


# --- Fonctions Core (inchangées, utilisent T pour les erreurs) ---
# ... (text_to_speech_fn, load_rvc_model, voice_conversion_fn, text_to_rvc_voice_fn) ...
# Note: Les gr.Error levées utiliseront la langue active AU MOMENT où l'erreur se produit.
# Elles ne seront pas retraduites si l'utilisateur change la langue *après* l'affichage de l'erreur.
def text_to_speech_fn(text, voice, speed, lang):
    # Utilise T[key] pour les messages d'erreur levés par gr.Error
    T_err = UI_TEXTS[DEFAULT_UI_LANGUAGE] # Utilise la langue par défaut pour les erreurs backend pour simplifier
    if not kokoro: raise gr.Error(T_err["error_kokoro_not_loaded"])
    if not text or not text.strip(): return create_silent_audio(), None
    logger.info(f"TTS Kokoro: Voice='{voice}', Speed={speed}, Lang='{lang}', Text='{text[:50]}...'")
    output_path = None
    try:
        samples, sample_rate = kokoro.create(text, voice=voice, speed=speed, lang=lang)
        if samples is None or len(samples) == 0: return create_silent_audio(), None
        logger.info(f"  Kokoro TTS generated {len(samples)} samples at {sample_rate} Hz.")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            output_path = tmp_file.name
            sf.write(output_path, samples, sample_rate)
            logger.info(f"Temporary Kokoro TTS audio created: {output_path}")
        return (sample_rate, samples), output_path
    except Exception as e:
        logger.error(f"Error during Kokoro TTS synthesis: {e}\n{traceback.format_exc()}")
        if output_path and os.path.exists(output_path):
            try: os.remove(output_path); logger.info(f"Cleaned up temp TTS file after error: {output_path}")
            except OSError as ose: logger.error(f"Failed to delete temp file {output_path}: {ose}")
        raise gr.Error(T_err["error_tts_failed"].format(e=e))

try:
    from fairseq.data.dictionary import Dictionary
    torch.serialization.add_safe_globals([Dictionary])
except Exception as e: logger.warning(f"Could not add Dictionary to safe globals (may be harmless): {e}")

def load_rvc_model(rvc_pth_path, rvc_index_path):
    T_err = UI_TEXTS[DEFAULT_UI_LANGUAGE]
    model_key = (rvc_pth_path, rvc_index_path)
    if model_key in RVC_CACHE:
        logger.info(f"Using cached RVC model: {os.path.basename(rvc_pth_path)}")
        return RVC_CACHE[model_key]
    else:
        logger.info(f"Loading new RVC model: {os.path.basename(rvc_pth_path)}")
        if not os.path.exists(rvc_pth_path): raise FileNotFoundError(T_err["error_pth_not_found"].format(path=rvc_pth_path))
        if rvc_index_path and not os.path.exists(rvc_index_path):
            logger.warning(T_err["error_index_not_found_warning"].format(path=rvc_index_path))
            rvc_index_path = None
        device = "cuda:0" if torch.cuda.is_available() else "cpu"; logger.info(f"Using RVC device: {device}")
        try:
            rvc = RVCInference(device=device)
            if rvc_index_path: logger.info(f"  with index: {os.path.basename(rvc_index_path)}")
            rvc.load_model(rvc_pth_path, index_path=rvc_index_path)
            RVC_CACHE[model_key] = rvc; return rvc
        except Exception as e:
            logger.error(f"Error loading RVC model ({os.path.basename(rvc_pth_path)}): {e}\n{traceback.format_exc()}")
            raise gr.Error(T_err["error_rvc_load_failed"].format(e=e))

def voice_conversion_fn(input_audio_path, rvc_index_file_name, rvc_pth_file_name, pitch_shift):
    T_err = UI_TEXTS[DEFAULT_UI_LANGUAGE]
    if not input_audio_path or not os.path.exists(input_audio_path): raise gr.Error(T_err["error_rvc_input_audio_invalid"])
    if not rvc_pth_file_name: raise gr.Error(T_err["error_rvc_pth_missing"])
    logger.info(f"RVC: Audio='{os.path.basename(str(input_audio_path))}', Index='{rvc_index_file_name}', PTH='{rvc_pth_file_name}', Pitch={pitch_shift}")
    output_rvc_audio_path = None
    try:
        rvc_pth_path = RVC_PTH_DIR / rvc_pth_file_name
        rvc_index_path = RVC_INDEX_DIR / rvc_index_file_name if rvc_index_file_name else None
        rvc = load_rvc_model(str(rvc_pth_path), str(rvc_index_path) if rvc_index_path else None)
        rvc.f0up_key = pitch_shift; logger.info(f"RVC f0up_key set to: {rvc.f0up_key}")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file: output_rvc_audio_path = tmp_file.name
        logger.info(f"  Calling rvc.infer_file (input: {input_audio_path}, output: {output_rvc_audio_path})")
        rvc.infer_file(str(input_audio_path), output_rvc_audio_path)
        logger.info(f"Temporary RVC audio created: {output_rvc_audio_path}")
        converted_audio_rvc, sample_rate_rvc = sf.read(output_rvc_audio_path)
        logger.info(f"RVC conversion successful. Samples: {len(converted_audio_rvc)}, Rate: {sample_rate_rvc}")
        return (sample_rate_rvc, converted_audio_rvc)
    except Exception as e:
        logger.error(f"Error during RVC conversion: {e}\n{traceback.format_exc()}")
        raise gr.Error(T_err["error_rvc_conversion_failed"].format(e=e))
    finally:
        if output_rvc_audio_path and os.path.exists(output_rvc_audio_path):
            try: os.remove(output_rvc_audio_path); logger.info(f"Cleaned up temp RVC file: {output_rvc_audio_path}")
            except OSError as ose: logger.error(f"Failed to delete temp RVC file {output_rvc_audio_path}: {ose}")

def text_to_rvc_voice_fn(text, rvc_index_file_name, rvc_pth_file_name, pitch_shift, kokoro_voice, speed, lang):
    T_err = UI_TEXTS[DEFAULT_UI_LANGUAGE]
    logger.info(f"Pipeline Text -> RVC: Kokoro='{kokoro_voice}', RVC='{rvc_pth_file_name}', Pitch={pitch_shift}, Text='{text[:50]}...'")
    temp_tts_audio_path = None
    try:
        logger.info("  Step 1: Generating Kokoro TTS...")
        tts_audio_data, temp_tts_audio_path = text_to_speech_fn(text, kokoro_voice, speed, lang)
        if temp_tts_audio_path is None or not os.path.exists(temp_tts_audio_path):
             logger.error("  Critical failure: TTS audio file not created or found after call.")
             raise gr.Error(T_err["error_tts_intermediate_failed"])
        logger.info(f"  Step 1 successful. Temp TTS audio: {temp_tts_audio_path}")
        logger.info("  Step 2: RVC Conversion...")
        rvc_output_audio = voice_conversion_fn(temp_tts_audio_path, rvc_index_file_name, rvc_pth_file_name, pitch_shift)
        logger.info("  Step 2 successful.")
        return rvc_output_audio
    except Exception as e:
        if not isinstance(e, gr.Error):
             logger.error(f"Unexpected error in Text to RVC pipeline: {e}\n{traceback.format_exc()}")
             raise gr.Error(T_err["error_unexpected"].format(e=e))
        else:
             raise e
    finally:
        if temp_tts_audio_path and os.path.exists(temp_tts_audio_path):
            try: os.remove(temp_tts_audio_path); logger.info(f"Cleaned up temp TTS file (Text-to-RVC pipeline): {temp_tts_audio_path}")
            except OSError as ose: logger.error(f"Failed to delete temp TTS file {temp_tts_audio_path}: {ose}")


# --- Fonction pour mettre à jour l'UI lors du changement de langue ---
def update_ui_language(selected_lang_code):
    """Retourne une série de gr.update pour changer les textes de l'UI."""
    T_update = UI_TEXTS[selected_lang_code]
    logger.info(f"Updating UI language to: {selected_lang_code.upper()}")
    return (
        # Composants Globaux
        gr.update(value=f"# {T_update['app_title']}"), # app_title_md
        gr.update(value=T_update["app_description"]), # app_description_md
        gr.update(value=T_update["refresh_button"]), # refresh_button
        gr.update(label=T_update["lang_select_label"]), # lang_selector (met à jour son propre label)

        # Tab 1
        gr.update(label=T_update["tab1_title"]), # tab_tts
        gr.update(label=T_update["tts_text_input_label"], placeholder=T_update["tts_text_input_placeholder"]), # tts_text_input
        gr.update(label=T_update["tts_voice_select_label"]), # tts_voice_select
        gr.update(label=T_update["tts_language_select_label"]), # tts_language_select
        gr.update(label=T_update["tts_speed_slider_label"]), # tts_speed_slider
        gr.update(value=T_update["tts_generate_button"]), # tts_generate_button
        gr.update(label=T_update["tts_output_audio_label"]), # tts_output_audio

        # Tab 2
        gr.update(label=T_update["tab2_title"]), # tab_rvc
        gr.update(label=T_update["rvc_input_audio_label"]), # rvc_input_audio
        gr.update(value=T_update["rvc_input_audio_info"]), # rvc_input_audio_info_md
        gr.update(label=T_update["rvc_pth_select_label"], info=T_update["rvc_pth_select_info"]), # rvc_pth_select
        gr.update(label=T_update["rvc_index_select_label"], info=T_update["rvc_index_select_info"]), # rvc_index_select
        gr.update(label=T_update["rvc_pitch_slider_label"]), # rvc_pitch_shift_slider
        gr.update(value=T_update["rvc_generate_button"]), # rvc_generate_button
        gr.update(label=T_update["rvc_output_audio_label"]), # rvc_output_audio

        # Tab 3
        gr.update(label=T_update["tab3_title"]), # tab_text_to_rvc
        gr.update(value=T_update["ttrvc_description"]), # ttrvc_description_md
        gr.update(label=T_update["ttrvc_text_input_label"], placeholder=T_update["ttrvc_text_input_placeholder"]), # ttrvc_text_input
        gr.update(label=T_update["ttrvc_rvc_pth_select_label"]), # ttrvc_rvc_pth_select
        gr.update(label=T_update["ttrvc_rvc_index_select_label"]), # ttrvc_rvc_index_select
        gr.update(label=T_update["ttrvc_pitch_slider_label"]), # ttrvc_pitch_shift_slider
        gr.update(label=T_update["ttrvc_kokoro_voice_label"], info=T_update["ttrvc_kokoro_voice_info"]), # ttrvc_kokoro_voice_select
        gr.update(label=T_update["ttrvc_language_select_label"]), # ttrvc_language_select
        gr.update(label=T_update["ttrvc_speed_slider_label"]), # ttrvc_speed_slider
        gr.update(value=T_update["ttrvc_generate_button"]), # ttrvc_generate_button
        gr.update(label=T_update["ttrvc_output_audio_label"]), # ttrvc_output_audio
    )

# --- Interface Gradio (utilise la langue par défaut initialement) ---
T = UI_TEXTS[DEFAULT_UI_LANGUAGE] # Charge les textes initiaux

with gr.Blocks(title=T["app_title"]) as demo:
    # Section Globale (Titre, Description, Sélecteur de Langue, Bouton Refresh)
    app_title_md = gr.Markdown(f"# {T['app_title']}")
    app_description_md = gr.Markdown(T["app_description"])
    with gr.Row():
        lang_selector = gr.Radio(
            choices=[("🇬🇧 English", "en"), ("🇫🇷 Français", "fr")],
            value=DEFAULT_UI_LANGUAGE, # Définit la valeur initiale
            label=T["lang_select_label"], # Utilise le label initial
            interactive=True
        )
        refresh_button = gr.Button(T["refresh_button"])

    # Définition des Onglets et de leurs composants
    with gr.Tabs() as tabs:
        with gr.Tab(T["tab1_title"], id="tab_tts") as tab_tts:
            with gr.Row():
                with gr.Column(scale=2):
                    tts_text_input = gr.Textbox(lines=4, placeholder=T["tts_text_input_placeholder"], label=T["tts_text_input_label"])
                with gr.Column(scale=1):
                    tts_voice_select = gr.Dropdown(VOICES_KOKORO, label=T["tts_voice_select_label"], value="ff_siwis")
                    tts_language_select = gr.Dropdown(LANGUAGES_KOKORO_CHOICES, label=T["tts_language_select_label"], value="fr-fr")
                    tts_speed_slider = gr.Slider(minimum=0.5, maximum=2.0, step=0.1, value=1.0, label=T["tts_speed_slider_label"])
            tts_generate_button = gr.Button(T["tts_generate_button"], variant="primary")
            tts_output_audio = gr.Audio(label=T["tts_output_audio_label"])
            tts_output_path_state = gr.State(None)

        with gr.Tab(T["tab2_title"], id="tab_rvc") as tab_rvc:
            with gr.Row():
                with gr.Column(scale=1):
                    rvc_input_audio = gr.Audio(label=T["rvc_input_audio_label"], sources=["upload", "microphone"], type="filepath")
                    rvc_input_audio_info_md = gr.Markdown(T["rvc_input_audio_info"]) # Markdown ici
                with gr.Column(scale=1):
                    rvc_pth_select = gr.Dropdown(AVAILABLE_PTH_FILES, label=T["rvc_pth_select_label"], info=T["rvc_pth_select_info"])
                    rvc_index_select = gr.Dropdown(AVAILABLE_INDEX_FILES, label=T["rvc_index_select_label"], info=T["rvc_index_select_info"])
                    rvc_pitch_shift_slider = gr.Slider(minimum=-24, maximum=24, step=1, value=0, label=T["rvc_pitch_slider_label"])
            rvc_generate_button = gr.Button(T["rvc_generate_button"], variant="primary")
            rvc_output_audio = gr.Audio(label=T["rvc_output_audio_label"])

        with gr.Tab(T["tab3_title"], id="tab_text_to_rvc") as tab_text_to_rvc:
            ttrvc_description_md = gr.Markdown(T["ttrvc_description"]) # Markdown ici
            with gr.Row():
                 with gr.Column(scale=2):
                    ttrvc_text_input = gr.Textbox(lines=4, placeholder=T["ttrvc_text_input_placeholder"], label=T["ttrvc_text_input_label"])
                 with gr.Column(scale=1):
                    ttrvc_rvc_pth_select = gr.Dropdown(AVAILABLE_PTH_FILES, label=T["ttrvc_rvc_pth_select_label"])
                    ttrvc_rvc_index_select = gr.Dropdown(AVAILABLE_INDEX_FILES, label=T["ttrvc_rvc_index_select_label"])
                    ttrvc_pitch_shift_slider = gr.Slider(minimum=-24, maximum=24, step=1, value=0, label=T["ttrvc_pitch_slider_label"])
            with gr.Row():
                 with gr.Column(scale=1):
                     ttrvc_kokoro_voice_select = gr.Dropdown(VOICES_KOKORO, label=T["ttrvc_kokoro_voice_label"], value="ff_siwis", info=T["ttrvc_kokoro_voice_info"])
                 with gr.Column(scale=1):
                     ttrvc_language_select = gr.Dropdown(LANGUAGES_KOKORO_CHOICES, label=T["ttrvc_language_select_label"], value="fr-fr")
                 with gr.Column(scale=1):
                    ttrvc_speed_slider = gr.Slider(minimum=0.5, maximum=2.0, step=0.1, value=1.0, label=T["ttrvc_speed_slider_label"])
            ttrvc_generate_button = gr.Button(T["ttrvc_generate_button"], variant="primary")
            ttrvc_output_audio = gr.Audio(label=T["ttrvc_output_audio_label"])

    # --- Définition des actions des boutons et des changements ---

    # Actions des boutons de génération (inchangées)
    tts_generate_button.click(fn=text_to_speech_fn, inputs=[tts_text_input, tts_voice_select, tts_speed_slider, tts_language_select], outputs=[tts_output_audio, tts_output_path_state], api_name="tts")
    rvc_generate_button.click(fn=voice_conversion_fn, inputs=[rvc_input_audio, rvc_index_select, rvc_pth_select, rvc_pitch_shift_slider], outputs=rvc_output_audio, api_name="rvc")
    ttrvc_generate_button.click(fn=text_to_rvc_voice_fn, inputs=[ttrvc_text_input, ttrvc_rvc_index_select, ttrvc_rvc_pth_select, ttrvc_pitch_shift_slider, ttrvc_kokoro_voice_select, ttrvc_speed_slider, ttrvc_language_select], outputs=ttrvc_output_audio, api_name="text_to_rvc")

    # Lien entre sortie TTS et entrée RVC (inchangé)
    tts_output_path_state.change(fn=lambda x: x, inputs=tts_output_path_state, outputs=rvc_input_audio)

    # Action du bouton Refresh (met à jour seulement les listes RVC des onglets 2 et 3)
    refresh_button.click(
        fn=refresh_models_list,
        inputs=None,
        outputs=[
            rvc_pth_select,         # Tab 2 PTH
            rvc_index_select,       # Tab 2 Index
            ttrvc_rvc_pth_select,   # Tab 3 PTH
            ttrvc_rvc_index_select  # Tab 3 Index
        ]
    )

    # ----> ACTION POUR CHANGER LA LANGUE DE L'INTERFACE <----
    # Liste de TOUS les composants dont le texte doit être mis à jour
    # L'ordre DOIT correspondre à l'ordre des gr.update() retournés par update_ui_language
    components_to_update = [
        # Globaux
        app_title_md, app_description_md, refresh_button, lang_selector,
        # Tab 1
        tab_tts, tts_text_input, tts_voice_select, tts_language_select, tts_speed_slider,
        tts_generate_button, tts_output_audio,
        # Tab 2
        tab_rvc, rvc_input_audio, rvc_input_audio_info_md, rvc_pth_select, rvc_index_select,
        rvc_pitch_shift_slider, rvc_generate_button, rvc_output_audio,
        # Tab 3
        tab_text_to_rvc, ttrvc_description_md, ttrvc_text_input, ttrvc_rvc_pth_select,
        ttrvc_rvc_index_select, ttrvc_pitch_shift_slider, ttrvc_kokoro_voice_select,
        ttrvc_language_select, ttrvc_speed_slider, ttrvc_generate_button, ttrvc_output_audio,
    ]
    lang_selector.change(
        fn=update_ui_language,
        inputs=lang_selector,
        outputs=components_to_update # La liste complète des composants
    )

# --- Lancement de l'application ---
if __name__ == "__main__":
    # Utilise les textes de la langue par défaut pour les messages console initiaux
    T_console = UI_TEXTS[DEFAULT_UI_LANGUAGE]
    if kokoro is None:
        print("\n" + "="*50)
        print(T_console["warn_kokoro_failed"])
        print(T_console["warn_kokoro_paths"])
        print(T_console["warn_tts_disabled"])
        print("="*50 + "\n")
    elif not AVAILABLE_PTH_FILES:
         print("\n" + "="*50)
         print(T_console["warn_rvc_pth_none_found"])
         print(T_console["warn_rvc_pth_location"].format(path=RVC_PTH_DIR))
         print(T_console["warn_rvc_disabled"])
         print("="*50 + "\n")

    print(f"Starting Gradio Interface in {DEFAULT_UI_LANGUAGE.upper()} (configurable in UI)...")
    demo.launch(share=False, server_port=7860)