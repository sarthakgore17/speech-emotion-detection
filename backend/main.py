from flask import Flask, request, jsonify
from flask_cors import CORS
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import traceback
import os

try:
    import sounddevice as sd
    from scipy.io.wavfile import write
except Exception:
    sd = None
    write = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
MODEL_PATH = os.path.join(BASE_DIR, 'trained_model.h5')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

_model = None

def get_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        print(f"Loading trained model from {MODEL_PATH}...")
        _model = load_model(MODEL_PATH)
        print("Model loaded successfully.")
    return _model

def extract_feature(data, sr, mfcc=True, chroma=True, mel=True):
    """
    Extract features from audio files into numpy array
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

@app.route('/hello', methods=['GET'])
def hello():
    return jsonify({'message': 'Hello, World!'})

@app.route('/predict', methods=['POST'])
def predict_emotion():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    file = request.files['audio']
    filename = file.filename if file.filename else "uploaded_audio.wav"
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)
    
    try:
        data, sr = librosa.load(file_path, sr=22050)
        feature = extract_feature(data, sr, mfcc=True, chroma=True, mel=True)
        feature = np.expand_dims(feature, axis=0)
        feature = np.expand_dims(feature, axis=2)
        
        model = get_model()
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
        predicted_emotion_with_emoji = f"{predicted_emotion} {emojis.get(predicted_emotion, '🎙️')}"
        
        raw_probs = prediction[0]
        probabilities = {
            cls: round(float(raw_probs[i]) * 100, 1)
            for i, cls in enumerate(label_encoder.classes_)
        }
        confidence = round(float(np.max(raw_probs)) * 100, 1)
        
        return jsonify({
            'predicted_emotion': predicted_emotion_with_emoji,
            'emotion': predicted_emotion,
            'confidence': confidence,
            'probabilities': probabilities
        })
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception:
                pass

if __name__ == '__main__':
    # Preload model at startup
    try:
        get_model()
    except Exception as e:
        print(f"Warning: Model could not be preloaded: {e}")
    app.run(host='0.0.0.0', port=5000, debug=False)