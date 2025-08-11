"""
Speech Emotion Recognition Inference Script
"""

import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import argparse

def extract_feature(data, sr, mfcc=True, chroma=True, mel=True):
    """
    Extract features from audio files into numpy array
    """
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

def predict_emotion(audio_path, model_path='trained_model.h5'):
    """
    Predict emotion from audio file
    """
    # Load audio
    data, sr = librosa.load(audio_path, sr=22050)
    
    # Extract features
    feature = extract_feature(data, sr, mfcc=True, chroma=True, mel=True)
    feature = np.expand_dims(feature, axis=0)
    feature = np.expand_dims(feature, axis=2)
    
    # Load model and predict
    model = load_model(model_path)
    prediction = model.predict(feature)
    predicted_class = np.argmax(prediction, axis=1)
    
    # Map to emotion labels
    emotions = {
        '01': 'Neutral',
        '02': 'Calm',
        '03': 'Happy',
        '04': 'Sad',
        '05': 'Angry',
        '06': 'Fearful',
        '07': 'Disgust',
        '08': 'Surprised'
    }
    
    emojis = {
        'Neutral': '😐',
        'Calm': '😌',
        'Happy': '😊',
        'Sad': '😢',
        'Angry': '😠',
        'Fearful': '😨',
        'Disgust': '🤢',
        'Surprised': '😲'
    }
    
    label_encoder = LabelEncoder()
    label_encoder.fit(list(emotions.values()))
    predicted_emotion = label_encoder.inverse_transform(predicted_class)[0]
    
    return predicted_emotion, emojis[predicted_emotion], prediction[0]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict emotion from audio file')
    parser.add_argument('audio_path', help='Path to audio file')
    parser.add_argument('--model', default='trained_model.h5', help='Path to model file')
    
    args = parser.parse_args()
    
    try:
        emotion, emoji, confidence = predict_emotion(args.audio_path, args.model)
        print(f"Predicted Emotion: {emotion} {emoji}")
        print(f"Confidence scores: {confidence}")
    except Exception as e:
        print(f"Error: {e}")
