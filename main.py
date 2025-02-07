import gradio as gr
import soundfile as sf
from kokoro_onnx import Kokoro
import torch
import librosa
import faiss
from pathlib import Path
import os
import numpy as np
from rvc_python.infer import RVCInference  # Import RVCInference from rvc-python

# --- Scanner les dossiers de modèles RVC ---
RVC_INDEX_DIR = "modelRVC/index"
RVC_PTH_DIR = "modelRVC/pth"

AVAILABLE_INDEX_FILES = []
AVAILABLE_PTH_FILES = []

try:
    AVAILABLE_INDEX_FILES = [f for f in os.listdir(RVC_INDEX_DIR) if f.endswith(".index")]
    AVAILABLE_PTH_FILES = [f for f in os.listdir(RVC_PTH_DIR) if f.endswith(".pth")]
except FileNotFoundError:
    print(f"Erreur: Les dossiers RVC '{RVC_INDEX_DIR}' ou '{RVC_PTH_DIR}' n'ont pas été trouvés. Veuillez vérifier les chemins.")
    AVAILABLE_INDEX_FILES = []
    AVAILABLE_PTH_FILES = []

print(f"Fichiers index RVC disponibles: {AVAILABLE_INDEX_FILES}")
print(f"Fichiers PTH RVC disponibles: {AVAILABLE_PTH_FILES}")
# --- Fin scan des dossiers de modèles RVC ---

# Initialisation du modèle Kokoro TTS (une seule fois au démarrage)
kokoro = Kokoro("modelTTS/kokoro-v1.0.onnx", "modelTTS/voices-1.1.bin")
print("Modèle Kokoro TTS initialisé")

# Liste des voix et langues pour les dropdowns Gradio
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
    "fr-fr": "Français",
    "it": "Italien",
    "en-us": "Anglais (US)",
    "en-gb": "Anglais (GB)",
    "ja": "Japonais",
    "cmn": "Chinois"
}

COUNTRY_FLAGS = {
    "af": "🇺🇸", # USA (pour af_voices dans l'image, à vérifier si c'est bien US)
    "am": "🇺🇸", # USA
    "bf": "🇬🇧", # UK
    "bm": "🇬🇧", # UK
    "ff": "🇫🇷", # France
    "if": "🇮🇹", # Italie (correction selon l'image, avant c'était Inde)
    "im": "🇮🇹", # Italie (correction selon l'image, avant c'était Inde)
    "jf": "🇯🇵", # Japon
    "jm": "🇯🇵", # Japon
    "zf": "🇨🇳", # Chine
}

VOICES_KOKORO = []
for voice_code in VOICES_KOKORO_RAW:
    country_code = voice_code[:2]
    flag = COUNTRY_FLAGS.get(country_code, "🌐") # Drapeau generique si non trouvé
    display_name = f"{flag} {voice_code.split('_')[1].capitalize()} ({LANGUAGES_KOKORO.get(voice_code[:2] + '-fr', LANGUAGES_KOKORO.get(voice_code[:2], voice_code[:2]))})" # Ajout de la langue approximative, corrige si besoin
    VOICES_KOKORO.append((display_name, voice_code))

LANGUAGES_KOKORO_CHOICES = [(LANGUAGES_KOKORO[key], key) for key in LANGUAGES_KOKORO]


def text_to_speech_fn(text, voice, speed, lang):
    print(f"Synthèse vocale TTS Kokoro : Texte='{text}', Voix='{voice}', Vitesse={speed}, Langue='{lang}'")
    try:
        samples, sample_rate = kokoro.create(text, voice=voice, speed=speed, lang=lang)
        print(f"  Kokoro TTS a créé : samples type={type(samples)}, sample_rate={sample_rate}") # DEBUG
        temp_audio_path = "temp_tts_audio.wav"
        sf.write(temp_audio_path, samples, sample_rate)  # Enregistrer temporairement pour RVC
        print(f"Audio TTS Kokoro créé: {temp_audio_path}")
        result = (sample_rate, samples), temp_audio_path
        print(f"  text_to_speech_fn retourne (succès) : type(audio)={type(result[0])}, audio[0] type={type(result[0][0]) if result[0] else None}, audio[1] type={type(result[0][1]) if result[0] else None}, temp_audio_path={temp_audio_path}") # DEBUG
        return result  # Retourne un tuple (sample_rate, samples) et le chemin
    except Exception as e:
        error_message = f"Erreur lors de la synthèse vocale Kokoro TTS: {e}"
        print(error_message)
        print(f"  Erreur TTS Kokoro: {e}") # DEBUG
        # Retourne un audio silencieux et le message d'erreur
        sample_rate = 44100 # Taux d'échantillonnage par défaut, ajustez si nécessaire
        silent_audio = np.array([])
        result = (sample_rate, silent_audio), None
        print(f"  text_to_speech_fn retourne (erreur) : type(audio)={type(result[0])}, audio[0] type={type(result[0][0]) if result[0] else None}, audio[1] type={type(result[0][1]) if result[0] else None}, temp_audio_path={result[1]}") # DEBUG
        return result # Retourne un tuple pour l'audio, et None pour l'état


def voice_conversion_fn(input_audio, rvc_index_file_name, rvc_pth_file_name, pitch_shift):
    print(f"Conversion vocale RVC : Audio in='{input_audio}', Index='{rvc_index_file_name}', PTH='{rvc_pth_file_name}', Pitch Shift={pitch_shift}")
    print(f"  voice_conversion_fn a reçu input_audio de type: {type(input_audio)}") # DEBUG
    if input_audio is None or rvc_index_file_name is None or rvc_pth_file_name is None:
        return "Veuillez fournir un audio et choisir un fichier index et un fichier pth pour la conversion RVC."

    # Ajout de la classe Dictionary aux safe globals pour permettre le chargement des checkpoints
    try:
        from fairseq.data.dictionary import Dictionary
        torch.serialization.add_safe_globals([Dictionary])
    except Exception as e:
        print("Erreur lors de l'ajout des safe globals:", e)

    try:
        # --- Construction des chemins complets vers les fichiers ---
        rvc_index_path = os.path.join(RVC_INDEX_DIR, rvc_index_file_name)
        rvc_pth_path = os.path.join(RVC_PTH_DIR, rvc_pth_file_name)
        # --- Fin construction des chemins ---

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        rvc = RVCInference(device=device)
        rvc.load_model(rvc_pth_path, index_path=rvc_index_path)

        # Définir le pitch shift en utilisant rvc.f0up_key
        rvc.f0up_key = pitch_shift
        print(f"Paramètre f0up_key RVC défini à : {rvc.f0up_key}")

        output_rvc_audio_path = "output_rvc_audio.wav"
        print(f"  Appel de rvc.infer_file avec input_audio: {input_audio}, output_rvc_audio_path: {output_rvc_audio_path}") # DEBUG
        rvc.infer_file(input_audio, output_rvc_audio_path) # Appel à infer_file sans argument de pitch shift

        converted_audio_rvc, sample_rate_rvc = sf.read(output_rvc_audio_path)
        print(f"Audio RVC créé: {output_rvc_audio_path}")
        result = sample_rate_rvc, converted_audio_rvc
        print(f"  voice_conversion_fn retourne (succès) : type(audio)={type(result)}, audio[0] type={type(result[0]) if result else None}, audio[1] type={type(result[1]) if result else None}") # DEBUG
        return result
    except Exception as e:
        error_message = f"Erreur lors de la conversion RVC avec rvc-python: {e}"
        print(error_message)
        print(f"  Erreur RVC: {e}") # DEBUG
        # Retourne une audio vide pour éviter que Gradio ne tente d'accéder à .dtype sur un tuple ou une chaîne
        sample_rate = 44100
        silent_audio = np.array([])
        result = sample_rate, silent_audio
        print(f"  voice_conversion_fn retourne (erreur) : type(audio)={type(result)}, audio[0] type={type(result[0]) if result else None}, audio[1] type={type(result[1]) if result else None}") # DEBUG
        return result

def text_to_rvc_voice_fn(text, rvc_index_file_name, rvc_pth_file_name, pitch_shift, kokoro_voice, speed, lang):
    print(f"Text to RVC Voice : Texte='{text}', Index='{rvc_index_file_name}', PTH='{rvc_pth_file_name}', Pitch Shift={pitch_shift}, Kokoro Voice='{kokoro_voice}', Speed={speed}, Langue='{lang}'")

    try:
        print(f"  Appel de text_to_speech_fn...") # DEBUG
        tts_output, temp_audio_path = text_to_speech_fn(text, kokoro_voice, speed, lang)
        print(f"  Retour de text_to_speech_fn: type(tts_output)={type(tts_output)}, type(temp_audio_path)={type(temp_audio_path)}, temp_audio_path={temp_audio_path}") # DEBUG
        if temp_audio_path is None: # Gérer le cas où temp_audio_path est None
            sample_rate = 44100
            silent_audio = np.array([])
            return (sample_rate, silent_audio) # Retourne uniquement le tuple audio en cas d'erreur TTS

        print(f"  Appel de voice_conversion_fn avec temp_audio_path: {temp_audio_path}...") # DEBUG
        rvc_output = voice_conversion_fn(temp_audio_path, rvc_index_file_name, rvc_pth_file_name, pitch_shift)
        print(f"  Retour de voice_conversion_fn: type(rvc_output)={type(rvc_output)}, type(rvc_output[0])={type(rvc_output[0]) if rvc_output else None}, type(rvc_output[1])={type(rvc_output[1]) if rvc_output else None}") # DEBUG
        sample_rate_rvc, converted_audio_rvc = rvc_output
        return (sample_rate_rvc, converted_audio_rvc) # Retourne uniquement le tuple audio en cas de succès

    except Exception as e:
        error_message = f"Erreur dans Text to RVC Voice: {e}"
        print(error_message)
        print(f"  Erreur Text to RVC Voice: {e}") # DEBUG
        sample_rate = 44100
        silent_audio = np.array([])
        return (sample_rate, silent_audio) # Retourne uniquement le tuple audio en cas d'erreur générale


with gr.Blocks() as demo:
    gr.Markdown("# Text to Audio Converter avec Kokoro TTS et RVC")

    with gr.Tab("Text to Speech (Kokoro TTS)"):
        text_input = gr.Textbox(lines=3, placeholder="Entrez votre texte ici...", label="Texte à convertir en audio")
        voice_select = gr.Dropdown(VOICES_KOKORO, label="Voix Kokoro TTS", value="ff_siwis")
        language_select = gr.Dropdown(LANGUAGES_KOKORO_CHOICES, label="Langue", value="fr-fr", choices=LANGUAGES_KOKORO_CHOICES)
        speed_slider = gr.Slider(minimum=0.5, maximum=2.0, step=0.1, value=1.0, label="Vitesse de parole")
        tts_output_audio = gr.Audio(label="Audio TTS Kokoro")
        tts_generate_button = gr.Button("Générer Audio TTS")

        tts_output_state = gr.State(None)  # Pour stocker le chemin audio TTS temporaire

        tts_generate_button.click(
            fn=text_to_speech_fn,
            inputs=[text_input, voice_select, speed_slider, language_select],
            outputs=[tts_output_audio, tts_output_state],
            api_name="tts"
        )

    with gr.Tab("Voice Conversion (RVC)"):
        # Forcer le type "filepath" pour que l'input renvoie le chemin du fichier audio
        rvc_input_audio = gr.Audio(label="Audio d'entrée pour RVC (peut être l'output TTS précédent ou un fichier uploadé)", type="filepath")
        rvc_index_select = gr.Dropdown(AVAILABLE_INDEX_FILES, label="Fichier Index RVC (.index)")
        rvc_pth_select = gr.Dropdown(AVAILABLE_PTH_FILES, label="Fichier PTH RVC (.pth)")
        pitch_shift_slider = gr.Slider(minimum=-12, maximum=12, step=1, value=0, label="Pitch Shift (demi-tons)")
        rvc_output_audio = gr.Audio(label="Audio Converti RVC")
        rvc_generate_button = gr.Button("Générer Audio RVC")

        rvc_generate_button.click(
            fn=voice_conversion_fn,
            inputs=[rvc_input_audio, rvc_index_select, rvc_pth_select, pitch_shift_slider],
            outputs=rvc_output_audio,
            api_name="rvc"
        )

        # Permettre d'utiliser l'audio TTS comme entrée RVC
        tts_output_audio.change(fn=lambda x: x, inputs=tts_output_audio, outputs=rvc_input_audio)

    with gr.Tab("Text to RVC Voice"):
        text_to_rvc_input = gr.Textbox(lines=3, placeholder="Entrez votre texte ici...", label="Texte à convertir en voix RVC")
        rvc_voice_select_rvc_tab = gr.Dropdown(AVAILABLE_PTH_FILES, label="Voix RVC (.pth)") # Choix du modèle RVC via PTH
        rvc_index_select_rvc_tab = gr.Dropdown(AVAILABLE_INDEX_FILES, label="Fichier Index RVC (.index)")
        pitch_shift_slider_rvc_tab = gr.Slider(minimum=-12, maximum=12, step=1, value=0, label="Pitch Shift (demi-tons)")
        speed_slider_rvc_tab = gr.Slider(minimum=0.5, maximum=2.0, step=0.1, value=1.0, label="Vitesse de parole")
        language_select_rvc_tab = gr.Dropdown(LANGUAGES_KOKORO_CHOICES, label="Langue", value="fr-fr", choices=LANGUAGES_KOKORO_CHOICES)
        kokoro_voice_select_rvc_tab = gr.Dropdown(VOICES_KOKORO, label="Voix Kokoro TTS (pour la base)", value="ff_siwis") # Voix Kokoro pour generer la base TTS
        rvc_output_audio_rvc_tab = gr.Audio(label="Audio RVC Directement depuis le Texte")
        rvc_generate_button_rvc_tab = gr.Button("Générer Audio RVC depuis Texte")

        rvc_generate_button_rvc_tab.click(
            fn=text_to_rvc_voice_fn,
            inputs=[text_to_rvc_input, rvc_index_select_rvc_tab, rvc_voice_select_rvc_tab, pitch_shift_slider_rvc_tab, kokoro_voice_select_rvc_tab, speed_slider_rvc_tab, language_select_rvc_tab],
            outputs=rvc_output_audio_rvc_tab,
            api_name="text_to_rvc"
        )


demo.launch(share=True)
