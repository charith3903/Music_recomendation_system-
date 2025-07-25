import cv2
import sounddevice as sd
import scipy.io.wavfile as wav
import librosa
import numpy as np
import joblib
from deepface import DeepFace
import pandas as pd

# Load pretrained models
voice_model = joblib.load("voice_emotion_model.pkl")
mood_pipeline = joblib.load("user_mood_kmeans_model.pkl")

# Mapping clusters to mood labels (adjust if needed)
cluster_labels = {
    0: "Happy",
    1: "Sad",
    2: "Angry",
    3: "Disgusted",
    4: "Neutral",
    # Add more if needed
}

# Voice emotion extraction
def record_audio(filename="live_audio.wav", duration=3, fs=44100):
    print("🎙️ Recording audio for emotion...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    wav.write(filename, fs, audio)
    print("✅ Audio recorded.")
    
def extract_voice_features(file):
    y, sr = librosa.load(file)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return np.mean(mfccs.T, axis=0)

def predict_voice_emotion(file):
    features = extract_voice_features(file).reshape(1, -1)
    pred = voice_model.predict(features)[0]
    proba = voice_model.predict_proba(features).max()
    return pred, proba

# Face emotion detection
def get_face_emotion():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return None
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        cap.release()
        return None
    
    try:
        results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        if isinstance(results, list):
            dominant_emotion = results[0]['dominant_emotion']
        else:
            dominant_emotion = results['dominant_emotion']
    except Exception as e:
        print("Face analysis error:", e)
        dominant_emotion = None
    
    cap.release()
    cv2.destroyAllWindows()
    return dominant_emotion

# Mood cluster prediction based on user historical data
def predict_mood_cluster(user_id):
    # Load user historical data CSV
    df = pd.read_csv("dataset/user_7_day_listening_history_sample.csv")
    user_data = df[df['user_id'] == user_id]
    if user_data.empty:
        print(f"No historical data found for user_id {user_id}")
        return None
    
    # Aggregate features (mean for numeric, mode for categorical)
    agg = user_data.groupby('user_id').agg({
        "tempo": "mean",
        "energy": "mean",
        "valence": "mean",
        "danceability": "mean",
        "acousticness": "mean",
        "time_of_day": lambda x: x.mode()[0],
        "emotion_tag": lambda x: x.mode()[0]
    }).reset_index(drop=True)
    
    pred_cluster = mood_pipeline.predict(agg)[0]
    mood = cluster_labels.get(pred_cluster, "Unknown")
    return mood

# Combine emotions with simple majority or priority
def combine_emotions(face_emotion, voice_emotion, mood_emotion):
    print(f"Face Emotion: {face_emotion}")
    print(f"Voice Emotion: {voice_emotion}")
    print(f"Mood Cluster: {mood_emotion}")
    
    # Simple majority vote approach
    votes = [face_emotion, voice_emotion, mood_emotion]
    votes = [v for v in votes if v is not None]
    if not votes:
        return "Unknown"
    
    # Count votes
    from collections import Counter
    vote_counts = Counter(votes)
    final_emotion = vote_counts.most_common(1)[0][0]

    return final_emotion
def recommend_playlist(emotion):
    recommendations = {
        "Happy": ["Relaxing", "Happy", "Romantic"],
        "Sad": ["Relaxing", "Happy"],
        "Angry": ["Relaxing", "Happy", "Sad"],
        "Fear": ["Relaxing"],
        "Neutral": ["Energetic"],
        "Disgusted": ["Relaxing", "Happy"],
    }
    return recommendations.get(emotion, ["Relaxing"])  # Default to Relaxing if unknown




def main():
    user_id = int(input("Enter user_id for mood prediction: "))
    
    # 1. Get face emotion
    face_emotion = get_face_emotion()
    
    # 2. Record and get voice emotion
    record_audio()
    voice_emotion, confidence = predict_voice_emotion("live_audio.wav")
    
    # 3. Get mood cluster from history
    mood_emotion = predict_mood_cluster(user_id)
    
    # 4. Combine results
    final_emotion = combine_emotions(face_emotion, voice_emotion, mood_emotion)
    print(f"\n🎯 Final Predicted Current Emotion: {final_emotion}")

    # Recommend playlist based on final emotion
    playlist_types = recommend_playlist(final_emotion)
    print(f"For current emotion '{final_emotion}', recommend playlist types: {playlist_types}")

if __name__ == "__main__":
    main()
