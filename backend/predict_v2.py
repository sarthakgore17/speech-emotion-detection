"""
Speech Emotion Recognition — Wav2Vec2 Inference Module (V2)
===========================================================

Provides a Wav2Vec2-based predictor class that loads the fine-tuned model
and predicts emotions from raw audio.  Drop-in replacement for the old
CNN-based predict.py.

Usage (CLI):
    python predict_v2.py path/to/audio.wav
    python predict_v2.py path/to/audio.wav --model models/wav2vec2_emotion/best_model

Usage (as module):
    from predict_v2 import Wav2Vec2EmotionPredictor
    predictor = Wav2Vec2EmotionPredictor("models/wav2vec2_emotion/best_model")
    emotion, emoji, confidence, probs = predictor.predict_file("audio.wav")
"""

import os
import json
import argparse

import numpy as np
import librosa
import torch
from transformers import (
    Wav2Vec2ForSequenceClassification,
    Wav2Vec2FeatureExtractor,
)

# ─────────────────────── Defaults ────────────────────────────────────

SAMPLE_RATE = 16_000
MAX_DURATION_SEC = 5.0
MAX_LENGTH = int(SAMPLE_RATE * MAX_DURATION_SEC)

# Alphabetically sorted — MUST match training script
EMOTION_LABELS = [
    "angry", "calm", "disgust", "fearful",
    "happy", "neutral", "sad", "surprised",
]

EMOTION_DISPLAY = {
    "angry":     "Angry",
    "calm":      "Calm",
    "disgust":   "Disgust",
    "fearful":   "Fearful",
    "happy":     "Happy",
    "neutral":   "Neutral",
    "sad":       "Sad",
    "surprised": "Surprised",
}

EMOTION_EMOJIS = {
    "Angry":     "😠",
    "Calm":      "😌",
    "Disgust":   "🤢",
    "Fearful":   "😨",
    "Happy":     "😊",
    "Neutral":   "😐",
    "Sad":       "😢",
    "Surprised": "😲",
}


# ─────────────────────── Predictor Class ─────────────────────────────


class Wav2Vec2EmotionPredictor:
    """
    Loads a fine-tuned Wav2Vec2ForSequenceClassification model and provides
    methods to predict emotion from audio files or raw numpy arrays.
    """

    def __init__(self, model_path: str):
        """
        Parameters
        ----------
        model_path : str
            Path to the saved model directory (must contain config.json,
            pytorch_model.bin / model.safetensors, and preprocessor_config.json).
        """
        if not os.path.isdir(model_path):
            raise FileNotFoundError(f"Model directory not found: {model_path}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

        # Try to load custom emotion config (written by train_v2.py)
        config_path = os.path.join(model_path, "emotion_config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                ecfg = json.load(f)
            self.emotion_labels = ecfg.get("emotion_labels", EMOTION_LABELS)
            self.sample_rate = ecfg.get("sample_rate", SAMPLE_RATE)
            self.max_duration = ecfg.get("max_duration_sec", MAX_DURATION_SEC)
        else:
            self.emotion_labels = EMOTION_LABELS
            self.sample_rate = SAMPLE_RATE
            self.max_duration = MAX_DURATION_SEC

        self.max_length = int(self.sample_rate * self.max_duration)

    def _preprocess(self, audio: np.ndarray, sr: int) -> torch.Tensor:
        """Resample, pad/truncate, and run through feature extractor."""
        # Resample if needed
        if sr != self.sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)

        # Pad or truncate
        if len(audio) > self.max_length:
            audio = audio[: self.max_length]
        else:
            audio = np.pad(audio, (0, self.max_length - len(audio)), mode="constant")

        inputs = self.feature_extractor(
            audio,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=False,
        )
        return inputs.input_values.to(self.device)

    @torch.no_grad()
    def predict(self, audio: np.ndarray, sr: int):
        """
        Predict emotion from raw audio array.

        Parameters
        ----------
        audio : np.ndarray  — 1-D float audio waveform
        sr    : int          — sample rate of the audio

        Returns
        -------
        emotion    : str   — display name e.g. "Happy"
        emoji      : str   — corresponding emoji
        confidence : float — 0–100 confidence percentage
        probs      : dict  — {emotion_display_name: percentage} for all classes
        """
        input_values = self._preprocess(audio, sr)
        outputs = self.model(input_values=input_values)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()[0]

        predicted_idx = int(np.argmax(probabilities))
        predicted_label = self.emotion_labels[predicted_idx]
        display_name = EMOTION_DISPLAY.get(predicted_label, predicted_label.title())
        emoji = EMOTION_EMOJIS.get(display_name, "🎙️")
        confidence = round(float(probabilities[predicted_idx]) * 100, 1)

        probs_dict = {}
        for i, label in enumerate(self.emotion_labels):
            dname = EMOTION_DISPLAY.get(label, label.title())
            probs_dict[dname] = round(float(probabilities[i]) * 100, 1)

        return display_name, emoji, confidence, probs_dict

    def predict_file(self, audio_path: str):
        """Convenience method: load audio file and predict."""
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        return self.predict(audio, sr)


# ─────────────────────── CLI ─────────────────────────────────────────

DEFAULT_MODEL_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "models", "wav2vec2_emotion", "best_model",
)


def main():
    parser = argparse.ArgumentParser(
        description="Predict emotion from an audio file using Wav2Vec2"
    )
    parser.add_argument("audio_path", help="Path to audio file (.wav, .mp3, .flac)")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_DIR,
        help="Path to fine-tuned model directory",
    )
    args = parser.parse_args()

    try:
        predictor = Wav2Vec2EmotionPredictor(args.model)
        emotion, emoji, confidence, probs = predictor.predict_file(args.audio_path)
        print(f"\nPredicted Emotion: {emotion} {emoji}")
        print(f"Confidence: {confidence}%")
        print("\nAll probabilities:")
        for emo, prob in sorted(probs.items(), key=lambda x: -x[1]):
            bar = "█" * int(prob / 2)
            print(f"  {emo:>12s}: {prob:5.1f}% {bar}")
    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
