from flask import Flask, request, jsonify
from flask_cors import CORS
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import sounddevice as sd
from scipy.io.wavfile import write
import traceback
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def extract_feature(data, sr, mfcc, chroma, mel):
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

def record_audio(duration=5, fs=22050):
    """
    Record audio from the microphone.
    
    Parameters
    ----------
    duration : int, duration of recording in seconds
    fs : int, sampling rate
    """
    print("Recording...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()  # Wait until recording is finished
    print("Recording complete.")
    return recording.flatten(), fs

@app.route('/hello', methods=['GET'])  # Added hello route
def hello():
    return jsonify({'message': 'Hello, World!'})

@app.route('/predict', methods=['POST'])  # Added prediction route
def predict_emotion():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    file = request.files['audio']
    filename = file.filename if file.filename else "uploaded_audio.wav"
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)
    
    try:
        data, sr = librosa.load(file_path, sr=22050)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
    try:
        feature = extract_feature(data, sr, mfcc=True, chroma=True, mel=True)
        feature = np.expand_dims(feature, axis=0)
        feature = np.expand_dims(feature, axis=2)
        
        model = load_model('trained_model.h5')  # Ensure this file exists
        prediction = model.predict(feature)
        predicted_class = np.argmax(prediction, axis=1)
        
        label_encoder = LabelEncoder()
        emotions = {
            '01':'Neutral',
            '02':'Calm',
            '03':'Happy',
            '04':'Sad',
            '05':'Angry',
            '06':'Fearful',
            '07':'Disgust',
            '08':'Surprised'
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
        label_encoder.fit(list(emotions.values()))
        predicted_emotion = label_encoder.inverse_transform(predicted_class)[0]
        predicted_emotion_with_emoji = f"{predicted_emotion} {emojis[predicted_emotion]}"
        
        return jsonify({'predicted_emotion': predicted_emotion_with_emoji})  # Return JSON response
    except Exception as e:
        print(traceback.format_exc())  # Print the full traceback for debugging
        return jsonify({'error': str(e)}), 500
    finally:
        os.remove(file_path)  # Clean up the saved file

if __name__ == '__main__':  # Updated entry point
    app.run(debug=True)  # Run Flask app