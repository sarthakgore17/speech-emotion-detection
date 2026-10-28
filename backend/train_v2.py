"""
Speech Emotion Recognition - Wav2Vec2 Training Script (V2 Architecture)
======================================================================

Upgrades the original CNN model to a pre-trained Wav2Vec2 transformer that
learns directly from raw audio waveforms — no manual feature extraction needed.

Usage (Local with GPU):
    python train_v2.py --data_dir ./audio_dataset --epochs 30

Usage (Google Colab):
    1. Upload this file to Colab
    2. Install deps: !pip install transformers torch librosa datasets accelerate
    3. Mount Google Drive or upload your RAVDESS dataset
    4. Run: !python train_v2.py --data_dir /content/audio_dataset --epochs 30

Expected RAVDESS directory structure:
    audio_dataset/
    ├── Actor_01/
    │   ├── 03-01-01-01-01-01-01.wav
    │   ├── 03-01-02-01-01-01-01.wav
    │   └── ...
    ├── Actor_02/
    └── ...
    OR
    ├── Audio_Speech_Actors_01-24/
    │   ├── Actor_01/*.wav
    │   └── ...
    └── Audio_Song_Actors_01-24/
        ├── Actor_01/*.wav
        └── ...
"""

import os
import sys
import glob
import json
import logging
import argparse
from datetime import datetime

import numpy as np
import librosa
import torch
from torch.utils.data import Dataset
from transformers import (
    Wav2Vec2ForSequenceClassification,
    Wav2Vec2FeatureExtractor,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# ─────────────────────────── Configuration ───────────────────────────

SAMPLE_RATE = 16_000           # Wav2Vec2 requires 16 kHz
MAX_DURATION_SEC = 5.0         # Max audio clip length in seconds
MAX_LENGTH = int(SAMPLE_RATE * MAX_DURATION_SEC)  # 80,000 samples

# RAVDESS emotion code mapping (filename position 3, 1-indexed)
RAVDESS_EMOTION_CODES = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised',
}

# Alphabetically sorted label list (must match inference code)
EMOTION_LABELS = sorted(set(RAVDESS_EMOTION_CODES.values()))
# => ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

LABEL2ID = {label: idx for idx, label in enumerate(EMOTION_LABELS)}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}

# ─────────────────────────── Logging ─────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("training_v2.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ─────────────────────────── Dataset ─────────────────────────────────


def discover_ravdess_files(data_dir: str):
    """
    Discover RAVDESS .wav files in common directory layouts and return
    (audio_paths, labels) where labels are integer class indices.
    """
    search_patterns = [
        # Flat: Actor_XX/*.wav
        os.path.join(data_dir, "Actor_*", "*.wav"),
        os.path.join(data_dir, "Actor_*", "*.WAV"),
        # Nested: Audio_*_Actors_*/Actor_XX/*.wav
        os.path.join(data_dir, "Audio_*_Actors_*", "Actor_*", "*.wav"),
        os.path.join(data_dir, "Audio_*_Actors_*", "Actor_*", "*.WAV"),
        # Deep glob fallback
        os.path.join(data_dir, "**", "*.wav"),
    ]

    found_files = set()
    for pat in search_patterns:
        found_files.update(glob.glob(pat, recursive=True))

    audio_paths, labels = [], []
    for fpath in sorted(found_files):
        basename = os.path.basename(fpath)
        parts = basename.replace(".wav", "").replace(".WAV", "").split("-")
        if len(parts) < 3:
            continue
        emotion_code = parts[2]
        emotion = RAVDESS_EMOTION_CODES.get(emotion_code)
        if emotion is None:
            logger.warning(f"Unknown emotion code '{emotion_code}' in {basename}")
            continue
        audio_paths.append(fpath)
        labels.append(LABEL2ID[emotion])

    logger.info(f"Discovered {len(audio_paths)} audio files in {data_dir}")
    return audio_paths, labels


class EmotionAudioDataset(Dataset):
    """
    PyTorch Dataset that loads raw audio, resamples to 16 kHz,
    pads/truncates to a fixed length, and returns Wav2Vec2-ready tensors.
    """

    def __init__(
        self,
        audio_paths: list,
        labels: list,
        feature_extractor: Wav2Vec2FeatureExtractor,
        max_length: int = MAX_LENGTH,
        augment: bool = False,
    ):
        self.audio_paths = audio_paths
        self.labels = labels
        self.feature_extractor = feature_extractor
        self.max_length = max_length
        self.augment = augment

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        # Load audio at 16 kHz mono
        audio, sr = librosa.load(self.audio_paths[idx], sr=SAMPLE_RATE)

        # ── Optional augmentation (training only) ──
        if self.augment:
            audio = self._apply_augmentation(audio)

        # Pad or truncate to fixed length
        if len(audio) > self.max_length:
            audio = audio[: self.max_length]
        else:
            audio = np.pad(audio, (0, self.max_length - len(audio)), mode="constant")

        # Process through Wav2Vec2 feature extractor
        inputs = self.feature_extractor(
            audio,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=False,
        )

        return {
            "input_values": inputs.input_values.squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }

    def _apply_augmentation(self, audio: np.ndarray) -> np.ndarray:
        """Apply random augmentations to improve generalization."""
        rng = np.random.default_rng()

        # 1. Add white noise (50% chance)
        if rng.random() < 0.5:
            noise_factor = rng.uniform(0.001, 0.005)
            noise = rng.standard_normal(len(audio))
            audio = audio + noise_factor * noise

        # 2. Time shift (40% chance)
        if rng.random() < 0.4:
            shift = rng.integers(-SAMPLE_RATE // 4, SAMPLE_RATE // 4)
            audio = np.roll(audio, shift)
            if shift > 0:
                audio[:shift] = 0
            else:
                audio[shift:] = 0

        # 3. Pitch shift (30% chance)
        if rng.random() < 0.3:
            n_steps = rng.uniform(-2, 2)
            audio = librosa.effects.pitch_shift(
                audio, sr=SAMPLE_RATE, n_steps=n_steps
            )

        # 4. Speed perturbation (30% chance)
        if rng.random() < 0.3:
            rate = rng.uniform(0.85, 1.15)
            audio = librosa.effects.time_stretch(audio, rate=rate)
            # Re-pad/truncate after stretching
            if len(audio) > self.max_length:
                audio = audio[: self.max_length]
            else:
                audio = np.pad(
                    audio, (0, max(0, self.max_length - len(audio))), mode="constant"
                )

        return audio.astype(np.float32)


# ─────────────────────────── Metrics ─────────────────────────────────


def compute_metrics(eval_pred):
    """Compute accuracy and weighted F1 for the Trainer."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    return {"accuracy": acc, "f1": f1}


# ─────────────────────────── Plotting ────────────────────────────────


def save_plots(trainer, y_true, y_pred, output_dir):
    """Save training curves and confusion matrix."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        logger.warning("matplotlib/seaborn not installed — skipping plots")
        return

    os.makedirs(output_dir, exist_ok=True)

    # ── Training curves from Trainer log history ──
    log_history = trainer.state.log_history
    train_loss, eval_loss, eval_acc = [], [], []
    for entry in log_history:
        if "loss" in entry and "eval_loss" not in entry:
            train_loss.append(entry["loss"])
        if "eval_loss" in entry:
            eval_loss.append(entry["eval_loss"])
        if "eval_accuracy" in entry:
            eval_acc.append(entry["eval_accuracy"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    if train_loss and eval_loss:
        # Resample train_loss to match eval_loss length
        step = max(1, len(train_loss) // len(eval_loss))
        train_loss_resampled = train_loss[::step][: len(eval_loss)]
        epochs_range = range(1, len(eval_loss) + 1)
        axes[0].plot(epochs_range, train_loss_resampled[: len(epochs_range)], label="Train Loss")
        axes[0].plot(epochs_range, eval_loss[: len(epochs_range)], label="Val Loss")
        axes[0].set_title("Loss Over Epochs")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()

    if eval_acc:
        axes[1].plot(range(1, len(eval_acc) + 1), eval_acc, label="Val Accuracy", color="green")
        axes[1].set_title("Validation Accuracy Over Epochs")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_curves_v2.png"), dpi=150)
    plt.close()
    logger.info(f"Training curves saved to {output_dir}/training_curves_v2.png")

    # ── Confusion matrix ──
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=EMOTION_LABELS,
        yticklabels=EMOTION_LABELS,
    )
    plt.title("Confusion Matrix — Wav2Vec2 Model")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix_v2.png"), dpi=150)
    plt.close()
    logger.info(f"Confusion matrix saved to {output_dir}/confusion_matrix_v2.png")


# ─────────────────────────── Main ────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Train Wav2Vec2 model for Speech Emotion Recognition"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to RAVDESS dataset directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/wav2vec2_emotion",
        help="Directory to save the trained model",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="facebook/wav2vec2-base",
        help="HuggingFace model ID for pre-trained Wav2Vec2",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no_augment",
        action="store_true",
        help="Disable data augmentation",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Speech Emotion Recognition — Wav2Vec2 Training (V2)")
    logger.info("=" * 60)
    logger.info(f"Config: {vars(args)}")
    logger.info(f"Device: {'CUDA — ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    logger.info(f"Labels ({len(EMOTION_LABELS)}): {EMOTION_LABELS}")

    # ── 1. Discover dataset ──
    audio_paths, labels = discover_ravdess_files(args.data_dir)
    if len(audio_paths) == 0:
        logger.error(
            f"No RAVDESS audio files found in {args.data_dir}. "
            "Check directory structure (see docstring at top of this file)."
        )
        sys.exit(1)

    # Print class distribution
    unique, counts = np.unique(labels, return_counts=True)
    for u, c in zip(unique, counts):
        logger.info(f"  {ID2LABEL[u]:>12s}: {c} samples")

    # ── 2. Train/Val split ──
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        audio_paths,
        labels,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=labels,
    )
    logger.info(f"Train: {len(train_paths)} | Val: {len(val_paths)}")

    # ── 3. Load pre-trained Wav2Vec2 ──
    logger.info(f"Loading pre-trained model: {args.model_name}")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.model_name)

    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(EMOTION_LABELS),
        label2id=LABEL2ID,
        id2label=ID2LABEL,
        problem_type="single_label_classification",
    )

    # Freeze the CNN feature encoder (first few transformer layers learn
    # high-level representations; the CNN already extracts good low-level
    # features from pre-training)
    model.freeze_feature_encoder()
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"Model params: {total_params:,} total | {trainable_params:,} trainable "
        f"({100 * trainable_params / total_params:.1f}%)"
    )

    # ── 4. Create datasets ──
    train_dataset = EmotionAudioDataset(
        train_paths,
        train_labels,
        feature_extractor,
        augment=not args.no_augment,
    )
    val_dataset = EmotionAudioDataset(
        val_paths,
        val_labels,
        feature_extractor,
        augment=False,
    )

    # ── 5. Training arguments ──
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=10,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        save_total_limit=3,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=2 if os.name != "nt" else 0,  # 0 on Windows
        seed=args.seed,
        report_to="none",  # disable wandb etc.
    )

    # ── 6. Trainer ──
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    # ── 7. Train! ──
    logger.info("Starting training...")
    start_time = datetime.now()
    trainer.train()
    elapsed = datetime.now() - start_time
    logger.info(f"Training completed in {elapsed}")

    # ── 8. Save best model ──
    best_model_dir = os.path.join(args.output_dir, "best_model")
    trainer.save_model(best_model_dir)
    feature_extractor.save_pretrained(best_model_dir)

    # Save label mapping alongside the model
    label_config = {
        "label2id": LABEL2ID,
        "id2label": {str(k): v for k, v in ID2LABEL.items()},
        "emotion_labels": EMOTION_LABELS,
        "sample_rate": SAMPLE_RATE,
        "max_duration_sec": MAX_DURATION_SEC,
    }
    with open(os.path.join(best_model_dir, "emotion_config.json"), "w") as f:
        json.dump(label_config, f, indent=2)

    logger.info(f"Best model saved to {best_model_dir}")

    # ── 9. Final evaluation ──
    logger.info("Running final evaluation on validation set...")
    eval_result = trainer.evaluate()
    logger.info(f"Final Val Accuracy: {eval_result['eval_accuracy']:.4f}")
    logger.info(f"Final Val F1:       {eval_result['eval_f1']:.4f}")

    # Detailed classification report
    predictions = trainer.predict(val_dataset)
    y_pred = np.argmax(predictions.predictions, axis=-1)
    y_true = np.array(val_labels)

    report = classification_report(
        y_true, y_pred, target_names=EMOTION_LABELS, digits=4
    )
    logger.info(f"\nClassification Report:\n{report}")

    # ── 10. Save plots ──
    plots_dir = os.path.join(args.output_dir, "plots")
    save_plots(trainer, y_true, y_pred, plots_dir)

    logger.info("=" * 60)
    logger.info("DONE — Model ready for inference!")
    logger.info(f"  Model path: {best_model_dir}")
    logger.info(f"  Use with:   python predict_v2.py <audio_file> --model {best_model_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
