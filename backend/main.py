"""
EmoSense — Speech Emotion Detection API Server (V2)
====================================================

Flask backend that serves emotion predictions. Automatically uses the
upgraded Wav2Vec2 model when available, falling back to the legacy CNN
model (.h5) for backward compatibility.

Start:
    python serve.py          # or  python main.py
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import librosa
import numpy as np
import traceback
import os
import subprocess

try:
    import sounddevice as sd
    from scipy.io.wavfile import write
except Exception:
    sd = None
    write = None

# ─────────────────────── Paths ───────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIST = os.path.join(os.path.dirname(BASE_DIR), "frontend", "dist")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Model paths
WAV2VEC2_MODEL_DIR = os.path.join(BASE_DIR, "models", "wav2vec2_emotion", "best_model")
LEGACY_MODEL_PATH = os.path.join(BASE_DIR, "trained_model.h5")

# ─────────────────────── App ─────────────────────────────────────────

app = Flask(__name__)
CORS(app)

# ─────────────────────── Emotion Constants ───────────────────────────

EMOTION_CLASSES = [
    "Angry", "Calm", "Disgust", "Fearful",
    "Happy", "Neutral", "Sad", "Surprised",
]

EMOTION_EMOJIS = {
    "Angry": "😠",
    "Calm": "😌",
    "Disgust": "🤢",
    "Fearful": "😨",
    "Happy": "😊",
    "Neutral": "😐",
    "Sad": "😢",
    "Surprised": "😲",
}

# ─────────────────────── Model Loading ───────────────────────────────
#
#  Priority:
#    1. Wav2Vec2 model (models/wav2vec2_emotion/best_model/)  ← NEW
#    2. Legacy CNN model (trained_model.h5)                   ← FALLBACK
#
# ─────────────────────────────────────────────────────────────────────

_wav2vec2_predictor = None
_legacy_model = None
_active_backend = None  # "wav2vec2" or "legacy"


def _try_load_wav2vec2():
    """Attempt to load the Wav2Vec2 model. Returns predictor or None."""
    global _wav2vec2_predictor
    if _wav2vec2_predictor is not None:
        return _wav2vec2_predictor

    if not os.path.isdir(WAV2VEC2_MODEL_DIR):
        return None

    try:
        from predict_v2 import Wav2Vec2EmotionPredictor
        _wav2vec2_predictor = Wav2Vec2EmotionPredictor(WAV2VEC2_MODEL_DIR)
        print(f"✅ Wav2Vec2 model loaded from {WAV2VEC2_MODEL_DIR}")
        return _wav2vec2_predictor
    except Exception as e:
        print(f"⚠️  Wav2Vec2 model failed to load: {e}")
        return None


def _try_load_legacy():
    """Attempt to load the legacy Keras .h5 model. Returns model or None."""
    global _legacy_model
    if _legacy_model is not None:
        return _legacy_model

    if not os.path.exists(LEGACY_MODEL_PATH):
        return None

    try:
        import tensorflow as tf
        from tensorflow.keras.models import load_model
        _legacy_model = load_model(LEGACY_MODEL_PATH)
        print(f"✅ Legacy CNN model loaded from {LEGACY_MODEL_PATH}")
        return _legacy_model
    except Exception as e:
        print(f"⚠️  Legacy model failed to load: {e}")
        return None


def get_model():
    """
    Load the best available model. Returns (backend_name, model_or_predictor).
    """
    global _active_backend

    # Try Wav2Vec2 first
    predictor = _try_load_wav2vec2()
    if predictor is not None:
        _active_backend = "wav2vec2"
        return "wav2vec2", predictor

    # Fallback to legacy CNN
    model = _try_load_legacy()
    if model is not None:
        _active_backend = "legacy"
        return "legacy", model

    raise FileNotFoundError(
        "No model found! Either:\n"
        f"  • Place Wav2Vec2 model in {WAV2VEC2_MODEL_DIR}\n"
        f"  • Place legacy CNN model at {LEGACY_MODEL_PATH}"
    )


# ─────────────────────── Audio Processing ────────────────────────────


def load_audio_safely(file_path, target_sr=22050):
    """
    Converts audio to 16-bit PCM WAV at target_sr using ffmpeg if necessary,
    normalizes volume to studio levels, and trims background silence.
    """
    converted_path = file_path + "_converted.wav"
    try:
        # 1. Convert any input format (WebM, Opus, MP3, etc.) to standard PCM WAV using ffmpeg
        subprocess.run([
            'ffmpeg', '-y', '-i', file_path,
            '-ar', str(target_sr),
            '-ac', '1',
            '-c:a', 'pcm_s16le',
            converted_path
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        data, sr = librosa.load(converted_path, sr=target_sr)
    except Exception:
        # Fallback if ffmpeg is not available or input is already standard WAV
        data, sr = librosa.load(file_path, sr=target_sr)
    finally:
        if os.path.exists(converted_path):
            try:
                os.remove(converted_path)
            except Exception:
                pass

    # Gentle dead-air silence trimming (top_db=35 preserves natural speech dynamics and word endings)
    try:
        trimmed_data, _ = librosa.effects.trim(data, top_db=35)
        # Keep if speech segment is at least 0.5s long
        if len(trimmed_data) > target_sr * 0.5:
            data = trimmed_data
    except Exception:
        pass

    return data, sr


def extract_feature(data, sr, mfcc=True, chroma=True, mel=True):
    """
    Extract features from audio files into numpy array (for legacy CNN model).
    """
    result = np.array([])
    if chroma:
        stft = np.abs(librosa.stft(data))
    if mfcc:                          
        mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sr, n_mfcc=40).T, axis=0)
        result = np.hstack((result, mfccs))
    if chroma:
        chroma_feat = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)
        result = np.hstack((result, chroma_feat))
    if mel:                             
        mel_feat = np.mean(librosa.feature.melspectrogram(y=data, sr=sr).T, axis=0)
        result = np.hstack((result, mel_feat))
        
    return result 


def record_audio(duration=5, fs=22050):
    """
    Record audio from the microphone (CLI utility).
    """
    if sd is None:
        raise RuntimeError("sounddevice is not available in this environment")
    print("Recording...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    print("Recording complete.")
    return recording.flatten(), fs


# ─────────────────────── Prediction Helpers ──────────────────────────


def _predict_wav2vec2(predictor, audio_data, sr):
    """Run prediction through the Wav2Vec2 model."""
    emotion, emoji, confidence, probabilities = predictor.predict(audio_data, sr)
    return {
        "predicted_emotion": f"{emotion} {emoji}",
        "emotion": emotion,
        "confidence": confidence,
        "probabilities": probabilities,
        "model": "wav2vec2",
    }


def _predict_legacy(model, audio_data, sr):
    """Run prediction through the legacy CNN model."""
    feature = extract_feature(audio_data, sr, mfcc=True, chroma=True, mel=True)
    feature = np.expand_dims(feature, axis=0)
    feature = np.expand_dims(feature, axis=2)

    prediction = model.predict(feature)
    raw_probs = prediction[0]
    predicted_idx = int(np.argmax(raw_probs))

    predicted_emotion = EMOTION_CLASSES[predicted_idx]
    emoji = EMOTION_EMOJIS.get(predicted_emotion, "🎙️")

    # Temperature calibration (T=2.2) to produce realistic, smooth confidence
    T = 2.2
    log_p = np.log(np.clip(raw_probs, 1e-10, 1.0))
    exp_p = np.exp((log_p - np.max(log_p)) / T)
    calibrated_probs = exp_p / np.sum(exp_p)

    probabilities = {
        EMOTION_CLASSES[i]: round(float(calibrated_probs[i]) * 100, 1)
        for i in range(len(EMOTION_CLASSES))
    }
    confidence = round(float(calibrated_probs[predicted_idx]) * 100, 1)

    return {
        "predicted_emotion": f"{predicted_emotion} {emoji}",
        "emotion": predicted_emotion,
        "confidence": confidence,
        "probabilities": probabilities,
        "model": "legacy_cnn",
    }


# ─────────────────────── Routes ──────────────────────────────────────


@app.route("/hello", methods=["GET"])
@app.route("/api/hello", methods=["GET"])
def hello():
    return jsonify({"message": "Hello, World!", "model_backend": _active_backend})


@app.route("/predict", methods=["POST"])
@app.route("/api/predict", methods=["POST"])
def predict_emotion():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    file = request.files["audio"]
    filename = file.filename if file.filename else "uploaded_audio.wav"
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    try:
        backend_name, model_or_predictor = get_model()

        if backend_name == "wav2vec2":
            # Wav2Vec2 expects 16 kHz
            data, sr = load_audio_safely(file_path, target_sr=16000)
            result = _predict_wav2vec2(model_or_predictor, data, sr)
        else:
            # Legacy CNN expects 22050 Hz
            data, sr = load_audio_safely(file_path, target_sr=22050)
            result = _predict_legacy(model_or_predictor, data, sr)

        return jsonify(result)

    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception:
                pass


@app.route("/model-info", methods=["GET"])
@app.route("/api/model-info", methods=["GET"])
def model_info():
    """Return info about which model backend is currently active."""
    try:
        backend_name, _ = get_model()
        return jsonify({
            "active_model": backend_name,
            "wav2vec2_available": os.path.isdir(WAV2VEC2_MODEL_DIR),
            "legacy_available": os.path.exists(LEGACY_MODEL_PATH),
            "emotion_classes": EMOTION_CLASSES,
        })
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503


@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_frontend(path):
    # Serve requested static file if it exists in frontend/dist
    file_path = os.path.join(FRONTEND_DIST, path)
    if path != "" and os.path.exists(file_path) and os.path.isfile(file_path):
        return send_from_directory(FRONTEND_DIST, path)
    # Otherwise fallback to index.html (SPA client routing)
    elif os.path.exists(os.path.join(FRONTEND_DIST, "index.html")):
        return send_from_directory(FRONTEND_DIST, "index.html")
    else:
        return jsonify({"message": "EmoSense API Server Running"}), 200


# ─────────────────────── Main ────────────────────────────────────────

if __name__ == "__main__":
    # Preload model at startup
    try:
        backend, _ = get_model()
        print(f"🚀 Active model backend: {backend}")
    except Exception as e:
        print(f"Warning: Model could not be preloaded: {e}")
    app.run(host="0.0.0.0", port=5000, debug=False)