---
language:
- en
license: mit
tags:
- audio
- speech
- emotion-recognition
- tensorflow
- keras
- audio-classification
- ravdess
datasets:
- ravdess
metrics:
- accuracy
- f1
model-index:
- name: Speech Emotion Recognition
  results:
  - task:
      type: audio-classification
      name: Audio Classification
    dataset:
      type: ravdess
      name: RAVDESS
    metrics:
    - type: accuracy
      name: Accuracy
      value: "See confusion matrix"
pipeline_tag: audio-classification
library_name: tensorflow
---

# Speech Emotion Recognition Model

This model performs speech emotion recognition, classifying audio into 8 different emotional states.

## Model Description

This is a deep learning model trained to recognize emotions from speech audio. The model can classify audio into the following emotions:

- 😐 Neutral
- 😌 Calm  
- 😊 Happy
- 😢 Sad
- 😠 Angry
- 😨 Fearful
- 🤢 Disgust
- 😲 Surprised

## Model Architecture

The model uses audio features extraction including:
- MFCC (Mel-frequency cepstral coefficients)
- Chroma features
- Mel-spectrogram features

## Usage

```python
import librosa
import numpy as np
from tensorflow.keras.models import load_model

# Load the model
model = load_model('trained_model.h5')

# Load and preprocess audio
def extract_feature(data, sr, mfcc=True, chroma=True, mel=True):
    result = np.array([])
    if mfcc:                          
        mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sr, n_mfcc=40).T, axis=0)
        result = np.hstack((result, mfccs))
    if chroma:
        stft = np.abs(librosa.stft(data))
        chroma_feat = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)
        result = np.hstack((result, chroma_feat))
    if mel:                             
        mel_feat = np.mean(librosa.feature.melspectrogram(y=data, sr=sr).T, axis=0)
        result = np.hstack((result, mel_feat))
    return result

# Load audio file
audio_path = "your_audio_file.wav"
data, sr = librosa.load(audio_path, sr=22050)

# Extract features
feature = extract_feature(data, sr, mfcc=True, chroma=True, mel=True)
feature = np.expand_dims(feature, axis=0)
feature = np.expand_dims(feature, axis=2)

# Make prediction
prediction = model.predict(feature)
predicted_class = np.argmax(prediction, axis=1)

# Map to emotion labels
emotions = {
    0: 'Neutral',
    1: 'Calm',
    2: 'Happy',
    3: 'Sad',
    4: 'Angry',
    5: 'Fearful',
    6: 'Disgust',
    7: 'Surprised'
}

predicted_emotion = emotions[predicted_class[0]]
print(f"Predicted emotion: {predicted_emotion}")
```

## Requirements

```
librosa
tensorflow
numpy
scikit-learn
```

## Training Data

The model was trained on the RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song) dataset, which contains speech emotion recordings with the following emotion categories:

- Neutral
- Calm
- Happy  
- Sad
- Angry
- Fearful
- Disgust
- Surprised

The dataset provides high-quality audio recordings from multiple speakers, allowing the model to learn robust emotion recognition patterns across different voices and speaking styles.

## Model Performance

The model has been trained and evaluated with the following performance metrics:

### Training Progress
![Loss and Accuracy](loss%20and%20accuracy.png)

The training curves show the model's learning progress over epochs, demonstrating convergence and good generalization.

### Confusion Matrix
![Confusion Matrix](Confusion-matrix-of-speaker-dependent-emotions-prediction-on-RAVDESS-corpus-with-8202.png)

The confusion matrix shows the model's performance on the RAVDESS dataset, demonstrating how well the model distinguishes between different emotional states.

## License

[Specify your license here]

## Citation

If you use this model, please cite:

```
@misc{speech-emotion-recognition,
  author = {JagjeevanAK},
  title = {Speech Emotion Recognition Model},
  year = {2025},
  publisher = {Hugging Face},
  url = {https://huggingface.co/JagjeevanAK/Speech-emotion-detection}
}
```
