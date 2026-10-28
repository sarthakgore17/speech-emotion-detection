# ============================================================================
#  🎙️ EmoSense V2 — Wav2Vec2 Training on Google Colab
# ============================================================================
#
#  HOW TO USE:
#  1. Open Google Colab:  https://colab.research.google.com
#  2. File → Upload Notebook → Upload this file (rename to .ipynb)
#     OR simply copy-paste each section into separate Colab cells
#  3. Runtime → Change runtime type → GPU (T4)
#  4. Run all cells top to bottom
#  5. Download the trained model from your Google Drive
#
#  Each "# %%"  below marks a new Colab cell. Copy everything between
#  two "# %%" markers into one Colab cell.
#
# ============================================================================

# %% [markdown]
# # 🎙️ EmoSense V2 — Train Wav2Vec2 for Speech Emotion Recognition
# 
# This notebook fine-tunes **facebook/wav2vec2-base** on the RAVDESS dataset.
# 
# **Requirements:** GPU runtime (Runtime → Change runtime type → T4 GPU)

# %% ── Cell 1: Install Dependencies ──────────────────────────────────────────

!pip install -q transformers torch librosa accelerate scikit-learn matplotlib seaborn datasets

import torch
print(f"✅ PyTorch {torch.__version__}")
print(f"✅ CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️  No GPU detected! Go to Runtime → Change runtime type → T4 GPU")

# %% ── Cell 2: Mount Google Drive (to save the trained model) ───────────────

from google.colab import drive
drive.mount('/content/drive')

# Create output directory on Drive
import os
OUTPUT_DIR = '/content/drive/MyDrive/EmoSense_V2_Model'
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"✅ Model will be saved to: {OUTPUT_DIR}")

# %% ── Cell 3: Download RAVDESS Dataset ─────────────────────────────────────

import os
import zipfile
import urllib.request

DATASET_DIR = '/content/ravdess_dataset'

if not os.path.exists(DATASET_DIR):
    os.makedirs(DATASET_DIR, exist_ok=True)

    print("📥 Downloading RAVDESS dataset from Zenodo...")
    print("   (This is the official public source — ~2 GB total)")

    # RAVDESS Speech files (24 actors)
    speech_url = "https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip"
    song_url   = "https://zenodo.org/records/1188976/files/Audio_Song_Actors_01-24.zip"

    for name, url in [("Speech", speech_url), ("Song", song_url)]:
        zip_path = f"/content/{name}.zip"
        print(f"   Downloading {name}...")
        urllib.request.urlretrieve(url, zip_path)
        print(f"   Extracting {name}...")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(DATASET_DIR)
        os.remove(zip_path)

    print(f"✅ Dataset ready at {DATASET_DIR}")
else:
    print(f"✅ Dataset already exists at {DATASET_DIR}")

# Count files
wav_count = sum(1 for root, dirs, files in os.walk(DATASET_DIR) 
                for f in files if f.lower().endswith('.wav'))
print(f"📊 Total audio files found: {wav_count}")

# %% ── Cell 4: Configuration ────────────────────────────────────────────────

import json
import glob
import logging
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
import matplotlib.pyplot as plt
import seaborn as sns

# ── Hyperparameters (tweak these if needed) ──
MODEL_NAME     = "facebook/wav2vec2-base"   # Pre-trained model
EPOCHS         = 30                          # Training epochs
BATCH_SIZE     = 8                           # Reduce to 4 if OOM
LEARNING_RATE  = 1e-5                        # Fine-tuning LR
WARMUP_RATIO   = 0.1                         # LR warmup
TEST_SIZE      = 0.2                         # Val split ratio
SEED           = 42
SAMPLE_RATE    = 16_000                      # Wav2Vec2 requires 16kHz
MAX_DURATION   = 5.0                         # seconds
MAX_LENGTH     = int(SAMPLE_RATE * MAX_DURATION)

# ── Emotion mapping ──
RAVDESS_CODES = {
    '01': 'neutral', '02': 'calm',    '03': 'happy',     '04': 'sad',
    '05': 'angry',   '06': 'fearful', '07': 'disgust',   '08': 'surprised',
}
EMOTION_LABELS = sorted(set(RAVDESS_CODES.values()))
LABEL2ID = {l: i for i, l in enumerate(EMOTION_LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}

print(f"✅ Config ready")
print(f"   Model: {MODEL_NAME}")
print(f"   Epochs: {EPOCHS}, Batch: {BATCH_SIZE}, LR: {LEARNING_RATE}")
print(f"   Labels ({len(EMOTION_LABELS)}): {EMOTION_LABELS}")

# %% ── Cell 5: Dataset & Data Loading ───────────────────────────────────────

def discover_files(data_dir):
    """Find RAVDESS .wav files and extract emotion labels."""
    patterns = [
        os.path.join(data_dir, "Actor_*", "*.wav"),
        os.path.join(data_dir, "Audio_*_Actors_*", "Actor_*", "*.wav"),
        os.path.join(data_dir, "**", "*.wav"),
    ]
    found = set()
    for p in patterns:
        found.update(glob.glob(p, recursive=True))

    paths, labels = [], []
    for f in sorted(found):
        parts = os.path.basename(f).replace(".wav","").replace(".WAV","").split("-")
        if len(parts) >= 3:
            emotion = RAVDESS_CODES.get(parts[2])
            if emotion:
                paths.append(f)
                labels.append(LABEL2ID[emotion])
    return paths, labels


class EmotionDataset(Dataset):
    def __init__(self, audio_paths, labels, feat_ext, max_len=MAX_LENGTH, augment=False):
        self.paths = audio_paths
        self.labels = labels
        self.feat_ext = feat_ext
        self.max_len = max_len
        self.augment = augment

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        audio, _ = librosa.load(self.paths[idx], sr=SAMPLE_RATE)

        if self.augment:
            audio = self._augment(audio)

        # Pad / truncate
        if len(audio) > self.max_len:
            audio = audio[:self.max_len]
        else:
            audio = np.pad(audio, (0, self.max_len - len(audio)))

        inputs = self.feat_ext(audio, sampling_rate=SAMPLE_RATE,
                               return_tensors="pt", padding=False)
        return {
            "input_values": inputs.input_values.squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }

    def _augment(self, audio):
        rng = np.random.default_rng()
        # White noise (50%)
        if rng.random() < 0.5:
            audio = audio + rng.uniform(0.001, 0.005) * rng.standard_normal(len(audio))
        # Time shift (40%)
        if rng.random() < 0.4:
            shift = rng.integers(-SAMPLE_RATE//4, SAMPLE_RATE//4)
            audio = np.roll(audio, shift)
            if shift > 0: audio[:shift] = 0
            else: audio[shift:] = 0
        # Pitch shift (30%)
        if rng.random() < 0.3:
            audio = librosa.effects.pitch_shift(audio, sr=SAMPLE_RATE,
                                                 n_steps=rng.uniform(-2, 2))
        # Speed perturb (30%)
        if rng.random() < 0.3:
            audio = librosa.effects.time_stretch(audio, rate=rng.uniform(0.85, 1.15))
            if len(audio) > self.max_len: audio = audio[:self.max_len]
            else: audio = np.pad(audio, (0, max(0, self.max_len - len(audio))))
        return audio.astype(np.float32)


# Discover files
audio_paths, labels = discover_files(DATASET_DIR)
print(f"✅ Found {len(audio_paths)} audio files")

# Class distribution
unique, counts = np.unique(labels, return_counts=True)
for u, c in zip(unique, counts):
    print(f"   {ID2LABEL[u]:>12s}: {c}")

# Train/val split
train_paths, val_paths, train_labels, val_labels = train_test_split(
    audio_paths, labels, test_size=TEST_SIZE, random_state=SEED, stratify=labels
)
print(f"\n📊 Train: {len(train_paths)} | Val: {len(val_paths)}")

# %% ── Cell 6: Load Model & Create Datasets ─────────────────────────────────

print(f"📥 Loading pre-trained model: {MODEL_NAME}")
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)

model = Wav2Vec2ForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(EMOTION_LABELS),
    label2id=LABEL2ID,
    id2label=ID2LABEL,
    problem_type="single_label_classification",
)

# Freeze CNN feature encoder (fine-tune only transformer layers)
model.freeze_feature_encoder()

total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"✅ Model loaded: {total:,} params total, {trainable:,} trainable ({100*trainable/total:.1f}%)")

# Create datasets
train_dataset = EmotionDataset(train_paths, train_labels, feature_extractor, augment=True)
val_dataset   = EmotionDataset(val_paths, val_labels, feature_extractor, augment=False)
print(f"✅ Datasets ready")

# %% ── Cell 7: Train! ───────────────────────────────────────────────────────

CHECKPOINT_DIR = "/content/wav2vec2_checkpoints"

training_args = TrainingArguments(
    output_dir=CHECKPOINT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=10,
    learning_rate=LEARNING_RATE,
    warmup_ratio=WARMUP_RATIO,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    save_total_limit=3,
    fp16=torch.cuda.is_available(),
    dataloader_num_workers=2,
    seed=SEED,
    report_to="none",
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted"),
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
)

print("🚀 Starting training...")
start = datetime.now()
trainer.train()
elapsed = datetime.now() - start
print(f"\n✅ Training complete in {elapsed}")

# %% ── Cell 8: Evaluate & Show Results ──────────────────────────────────────

# Final evaluation
eval_result = trainer.evaluate()
print(f"\n{'='*50}")
print(f"📊 FINAL RESULTS")
print(f"{'='*50}")
print(f"   Validation Accuracy: {eval_result['eval_accuracy']*100:.2f}%")
print(f"   Validation F1 Score: {eval_result['eval_f1']*100:.2f}%")

# Classification report
predictions = trainer.predict(val_dataset)
y_pred = np.argmax(predictions.predictions, axis=-1)
y_true = np.array(val_labels)

print(f"\n📋 Classification Report:")
print(classification_report(y_true, y_pred, target_names=EMOTION_LABELS, digits=4))

# ── Plot Training Curves ──
log_history = trainer.state.log_history
train_loss, eval_loss, eval_acc = [], [], []
for entry in log_history:
    if "loss" in entry and "eval_loss" not in entry:
        train_loss.append(entry["loss"])
    if "eval_loss" in entry:
        eval_loss.append(entry["eval_loss"])
    if "eval_accuracy" in entry:
        eval_acc.append(entry["eval_accuracy"])

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Loss curve
if train_loss and eval_loss:
    step = max(1, len(train_loss) // len(eval_loss))
    tl = train_loss[::step][:len(eval_loss)]
    ep = range(1, len(eval_loss)+1)
    axes[0].plot(ep, tl[:len(list(ep))], 'b-', label='Train Loss')
    axes[0].plot(ep, eval_loss[:len(list(ep))], 'r-', label='Val Loss')
    axes[0].set_title('Loss Over Epochs')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()

# Accuracy curve
if eval_acc:
    axes[1].plot(range(1, len(eval_acc)+1), [a*100 for a in eval_acc], 'g-')
    axes[1].set_title('Validation Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy %')

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=EMOTION_LABELS, yticklabels=EMOTION_LABELS, ax=axes[2])
axes[2].set_title('Confusion Matrix')
axes[2].set_xlabel('Predicted')
axes[2].set_ylabel('True')

plt.tight_layout()
plt.savefig('/content/training_results.png', dpi=150)
plt.show()
print("✅ Plots saved to /content/training_results.png")

# %% ── Cell 9: Save Model to Google Drive ───────────────────────────────────

BEST_MODEL_DIR = os.path.join(OUTPUT_DIR, "best_model")

# Save model + feature extractor
trainer.save_model(BEST_MODEL_DIR)
feature_extractor.save_pretrained(BEST_MODEL_DIR)

# Save emotion config
config = {
    "label2id": LABEL2ID,
    "id2label": {str(k): v for k, v in ID2LABEL.items()},
    "emotion_labels": EMOTION_LABELS,
    "sample_rate": SAMPLE_RATE,
    "max_duration_sec": MAX_DURATION,
}
with open(os.path.join(BEST_MODEL_DIR, "emotion_config.json"), "w") as f:
    json.dump(config, f, indent=2)

# Also save the training results plot
import shutil
shutil.copy('/content/training_results.png', os.path.join(OUTPUT_DIR, 'training_results.png'))

print(f"\n{'='*50}")
print(f"✅ MODEL SAVED TO GOOGLE DRIVE")
print(f"{'='*50}")
print(f"📁 Location: {BEST_MODEL_DIR}")
print(f"\nFiles saved:")
for f in os.listdir(BEST_MODEL_DIR):
    size = os.path.getsize(os.path.join(BEST_MODEL_DIR, f))
    print(f"   {f:>35s}  ({size/1024/1024:.1f} MB)" if size > 1024*1024 
          else f"   {f:>35s}  ({size/1024:.1f} KB)")

print(f"""
{'='*50}
📋 NEXT STEPS
{'='*50}

1. Go to Google Drive → EmoSense_V2_Model → best_model/
2. Download the entire 'best_model' folder
3. Place it in your project:

   Speech-emotion-detection/
   └── backend/
       └── models/
           └── wav2vec2_emotion/
               └── best_model/        ← PUT FILES HERE
                   ├── config.json
                   ├── model.safetensors
                   ├── preprocessor_config.json
                   └── emotion_config.json

4. Restart your Flask server:
   python serve.py

   You should see:
   ✅ Wav2Vec2 model loaded from models/wav2vec2_emotion/best_model
   🚀 Active model backend: wav2vec2
""")

# %% ── Cell 10 (Optional): Quick Test ───────────────────────────────────────

# Test on a random validation sample
test_idx = np.random.randint(len(val_paths))
test_audio, test_sr = librosa.load(val_paths[test_idx], sr=SAMPLE_RATE)

# Pad/truncate
if len(test_audio) > MAX_LENGTH:
    test_audio = test_audio[:MAX_LENGTH]
else:
    test_audio = np.pad(test_audio, (0, MAX_LENGTH - len(test_audio)))

inputs = feature_extractor(test_audio, sampling_rate=SAMPLE_RATE,
                           return_tensors="pt", padding=False)

model.eval()
with torch.no_grad():
    device = next(model.parameters()).device
    outputs = model(input_values=inputs.input_values.to(device))
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()[0]

pred_idx = np.argmax(probs)
true_label = ID2LABEL[val_labels[test_idx]]
pred_label = ID2LABEL[pred_idx]

emojis = {'angry':'😠','calm':'😌','disgust':'🤢','fearful':'😨',
          'happy':'😊','neutral':'😐','sad':'😢','surprised':'😲'}

print(f"\n🎵 Test file: {os.path.basename(val_paths[test_idx])}")
print(f"   True emotion:      {true_label} {emojis.get(true_label,'')}")
print(f"   Predicted emotion:  {pred_label} {emojis.get(pred_label,'')}")
print(f"   Confidence:         {probs[pred_idx]*100:.1f}%")
print(f"   {'✅ CORRECT!' if true_label == pred_label else '❌ Incorrect'}")
