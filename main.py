import cv2
import sounddevice as sd
import scipy.io.wavfile as wav
import librosa
import numpy as np
import joblib
import pandas as pd
import time
from collections import Counter
from deepface import DeepFace
from keras.models import load_model

# === Load Models ===
voice_model = joblib.load("voice_emotion_model.pkl")
mood_pipeline = joblib.load("user_mood_kmeans_model.pkl")
user_pref_model = load_model("user_preference_model.h5")

# === Load Dataset ===
songs_df = pd.read_csv("mood_based_song_analysis/predicted_moods.csv")

# === Cluster to Label Mapping ===
cluster_labels = {
    0: "Happy",
    1: "Sad",
    2: "Angry",
    3: "Disgusted",
    4: "Neutral",
}

# === Helper Functions ===
def record_audio(filename="live_audio.wav", duration=3, fs=44100):
    print("🎙️ Recording audio...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    wav.write(filename, fs, audio)

def extract_voice_features(file):
    y, sr = librosa.load(file)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return np.mean(mfccs.T, axis=0)

def predict_voice_emotion(file):
    features = extract_voice_features(file).reshape(1, -1)
    pred = voice_model.predict(features)[0]
    return pred.capitalize(), voice_model.predict_proba(features).max()

def get_face_emotion():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open webcam.")
        return None

    print("📷 Press 'q' to start 15s face emotion detection.")
    emotions_list, analyzing, start_time = [], False, None

    while True:
        ret, frame = cap.read()
        if not ret: break
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') and not analyzing:
            analyzing, start_time = True, time.time()
            print("🧠 Detecting emotions for 15 seconds...")

        if analyzing:
            try:
                results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                dominant = results[0]['dominant_emotion'] if isinstance(results, list) else results['dominant_emotion']
                dominant = dominant.capitalize()
                emotions_list.append(dominant)
                cv2.putText(frame, f"{dominant}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
            except:
                cv2.putText(frame, "No face", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            elapsed = time.time() - start_time
            cv2.putText(frame, f"{15-int(elapsed)}s left", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
            if elapsed >= 15: break
        else:
            cv2.putText(frame, "Press 'q' to start", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

        cv2.imshow("Face Emotion", frame)

    cap.release()
    cv2.destroyAllWindows()

    return Counter(emotions_list).most_common(1)[0][0] if emotions_list else None

def predict_mood_cluster(user_id):
    df = pd.read_csv("dataset/user_7_day_listening_history_sample.csv")
    user_data = df[df['user_id'] == user_id]
    if user_data.empty: return None
    agg = user_data.groupby('user_id').agg({
        "tempo": "mean", "energy": "mean", "valence": "mean", "danceability": "mean",
        "acousticness": "mean", "time_of_day": lambda x: x.mode()[0], "emotion_tag": lambda x: x.mode()[0]
    }).reset_index(drop=True)
    pred_cluster = mood_pipeline.predict(agg)[0]
    return cluster_labels.get(pred_cluster, "Unknown")

def combine_emotions(face_emotion, voice_emotion, mood_emotion):
    all_emotions = [e for e in [face_emotion, voice_emotion, mood_emotion] if e]
    return Counter(all_emotions).most_common(1)[0][0] if all_emotions else "Unknown"

def recommend_playlist_types(emotion):
    doctor_recommendations = {
        "Angry": ["relaxing"],
        "Sad": ["happy", "romantic"],
        "Neutral": ["relaxing", "romantic"],
        "Happy": ["energetic", "happy"],
        "Fear": ["relaxing", "romantic"],
        "Disgust": ["relaxing", "happy"]
    }
    return doctor_recommendations.get(emotion, ["relaxing"])

def get_user_preference_vector():
    import pandas as pd
    df = pd.read_csv("dataset/predicted_user_history.csv")

    # No need for model — just return mean tempo and energy
    user_pref = {
        "tempo": df["tempo"].mean(),
        "energy": df["energy"].mean()
    }
    return user_pref

import numpy as np

def calculate_similarity(song_features, user_pref):
    song_vec = np.array([song_features['tempo'], song_features['energy']])
    user_vec = np.array([user_pref['tempo'], user_pref['energy']])
    song_norm = song_vec / np.linalg.norm(song_vec)
    user_norm = user_vec / np.linalg.norm(user_vec)
    return np.dot(song_norm, user_norm)

def recommend_song_list(emotion, songs_df, user_pref, top_n=10):
    playlist_types = recommend_playlist_types(emotion)
    print(f"🎯 Emotion: {emotion}")
    print(f"🧠 Doctor Recommendation Playlist Types: {playlist_types}")

    # Normalize and strip whitespaces
    songs_df['major_feeling'] = songs_df['major_feeling'].astype(str).str.strip().str.lower()
    songs_df['second_major_feeling'] = songs_df['second_major_feeling'].astype(str).str.strip().str.lower()

    # Convert playlist_types to lowercase
    playlist_types = [pt.lower() for pt in playlist_types]

    print("🎵 Unique major_feeling values in dataset:", songs_df['major_feeling'].unique())
    print("🎵 Unique second_major_feeling values in dataset:", songs_df['second_major_feeling'].unique())

    # Filter based on doctor's recommendation
    filtered_songs = songs_df[
        (songs_df['major_feeling'].isin(playlist_types)) |
        (songs_df['second_major_feeling'].isin(playlist_types))
    ]

    print(f"🔍 Filtered songs count: {len(filtered_songs)}")

    if filtered_songs.empty:
        print("⚠️ No songs matched the emotion-based playlist types.")
        return pd.DataFrame()

    filtered_songs = filtered_songs.copy()
    filtered_songs['similarity'] = filtered_songs.apply(
        lambda row: calculate_similarity(row, user_pref), axis=1
    )

    return filtered_songs.sort_values(by='similarity', ascending=False).head(top_n)

def generate_playlist_csv(emotion, user_pref_path, mood_song_path, output_path):
    import pandas as pd

    songs_df = pd.read_csv(mood_song_path)
    user_pref = get_user_preference_vector()

    playlist_df = recommend_song_list(emotion, songs_df, user_pref, top_n=10)

    if playlist_df.empty:
        print("⚠️ No songs matched your emotion and preferences.")
    else:
        print("\n🎧 Final Personalized Playlist:")
        print(playlist_df[['song', 'singer']].to_string(index=False))

    playlist_df.to_csv(output_path, index=False)
    print(f"\n✅ Playlist saved to {output_path}")

# === MAIN ===
def main():
    user_id = int(input("Enter user ID: "))
    
    # Step 1: Face emotion
    face_emotion = get_face_emotion()
    
    # Step 2: Voice emotion
    record_audio()
    voice_emotion, _ = predict_voice_emotion("live_audio.wav")
    
    # Step 3: Historical mood
    mood_emotion = predict_mood_cluster(user_id)
    
    # Step 4: Combine all
    final_emotion = combine_emotions(face_emotion, voice_emotion, mood_emotion)
    print(f"\n🎯 Final Emotion: {final_emotion}")
    
    # Step 5: Get user preferences
    final_emotion = final_emotion  # This should be captured from your emotion detection logic
    user_pref_path = "dataset/user_history.csv"
    mood_song_path = "mood_based_song_analysis/predicted_moods.csv"
    output_path = "dataset/personalized_playlist.csv"

    generate_playlist_csv(final_emotion, user_pref_path, mood_song_path, output_path)

if __name__ == "__main__":
    main()
