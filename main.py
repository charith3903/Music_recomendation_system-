import cv2
import sounddevice as sd
import scipy.io.wavfile as wav
import librosa
import numpy as np
import joblib
import pandas as pd
from deepface import DeepFace
from keras.models import load_model
import time
from collections import Counter

# Load pretrained models
voice_model = joblib.load("voice_emotion_model.pkl")
mood_pipeline = joblib.load("user_mood_kmeans_model.pkl")
""" user_pref_model = load_model("user_preference_model.h5")

# Load song dataset
songs_df = pd.read_csv("mood_based_song_analysis/predicted_moods.csv")
 """
# Mapping clusters to mood labels
cluster_labels = {
    0: "Happy",
    1: "Sad",
    2: "Angry",
    3: "Disgusted",
    4: "Neutral",
}

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
    return pred.capitalize(), proba



import time
from collections import Counter
import cv2
from deepface import DeepFace

def get_face_emotion():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open webcam.")
        return None

    print("📷 Showing webcam. Press 'q' to start 15-second emotion analysis...")

    analyzing = False
    emotions_list = []
    start_time = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame.")
            break

        key = cv2.waitKey(1) & 0xFF

        # Press 'q' to start analyzing
        if key == ord('q') and not analyzing:
            analyzing = True
            start_time = time.time()
            print("🧠 Starting 15-second emotion analysis...")

        # Analyzing for 15 seconds
        if analyzing:
            try:
                results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                dominant_emotion = results[0]['dominant_emotion'] if isinstance(results, list) else results['dominant_emotion']
                dominant_emotion = dominant_emotion.capitalize()
                emotions_list.append(dominant_emotion)

                # Show current emotion on frame
                cv2.putText(frame, f"Detected: {dominant_emotion}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                print(f"🟢 Frame Emotion: {dominant_emotion}")
            except Exception as e:
                print("⚠️ Frame skipped (no face or error)")
                cv2.putText(frame, "No face detected", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # Show remaining time
            elapsed = time.time() - start_time
            time_left = max(0, int(15 - elapsed))
            cv2.putText(frame, f"⏱️ {time_left}s left", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            # End after 15 seconds
            if elapsed >= 15:
                break
        else:
            # Before analysis starts, prompt user
            cv2.putText(frame, "Press 'q' to start analysis", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        # Show frame
        cv2.imshow("Live Webcam - Emotion Detector", frame)

    cap.release()
    cv2.destroyAllWindows()

    if not emotions_list:
        print("⚠️ No emotion detected during analysis.")
        return None

    # Final result: most frequent emotion
    final_emotion = Counter(emotions_list).most_common(1)[0][0]
    print(f"🎯 Final Predicted Emotion: {final_emotion}")
    return final_emotion

def predict_mood_cluster(user_id):
    df = pd.read_csv("dataset/user_7_day_listening_history_sample.csv")
    user_data = df[df['user_id'] == user_id]
    if user_data.empty:
        print(f"⚠️ No historical data for user {user_id}")
        return None
    agg = user_data.groupby('user_id').agg({
        "tempo": "mean",
        "energy": "mean",
        "valence": "mean",
        "danceability": "mean",
        "acousticness": "mean",
        "time_of_day": lambda x: x.mode()[0],
        "emotion_tag": lambda x: x.mode()[0]
    }).reset_index(drop=True)
    
    # ✅ DO NOT drop columns
    pred_cluster = mood_pipeline.predict(agg)[0]
    mood = cluster_labels.get(pred_cluster, "Unknown")
    return mood
def combine_emotions(face_emotion, voice_emotion, mood_emotion):
    print(f"\n🧠 Face Emotion: {face_emotion}")
    print(f"🗣️ Voice Emotion: {voice_emotion}")
    print(f"📊 Mood Cluster: {mood_emotion}")
    votes = [e for e in [face_emotion, voice_emotion, mood_emotion] if e]
    if not votes:
        return "Unknown"
    from collections import Counter
    final_emotion = Counter(votes).most_common(1)[0][0]
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
    return recommendations.get(emotion, ["Relaxing"])
""" 
def calculate_similarity(song_features, user_pref):
    song_vec = np.array([song_features['tempo'], song_features['energy']])
    user_vec = np.array([user_pref['tempo'], user_pref['energy']])
    song_norm = song_vec / np.linalg.norm(song_vec)
    user_norm = user_vec / np.linalg.norm(user_vec)
    return np.dot(song_norm, user_norm)

def recommend_song_list(emotion, songs_df, user_pref, top_n=5):
    playlist_types = recommend_playlist(emotion)
    filtered_songs = songs_df[
        (songs_df['major_feeling'].str.capitalize().isin(playlist_types)) |
        (songs_df['second_major_feeling'].str.capitalize().isin(playlist_types))
    ]
    filtered_songs = filtered_songs.copy()
    filtered_songs['similarity'] = filtered_songs.apply(
        lambda row: calculate_similarity(row, user_pref), axis=1
    )
    recommended = filtered_songs.sort_values(by='similarity', ascending=False)
    return recommended[['song', 'singer', 'tempo', 'energy', 'major_feeling', 'second_major_feeling', 'similarity']].head(top_n)

def get_user_pref_vector():
    df = pd.read_csv("dataset/user_history.csv")

    # Columns required for the preference model
    feature_cols = [
        'tempo', 'energy', 'danceability', 'acousticness', 'instrumentalness',
        'valence', 'liveness', 'speechiness', 'loudness',
        'genre', 'artist', 'language', 'lyrics_sentiment', 'emotion_tag'
    ]

    # Encode categorical columns
    cat_cols = ['genre', 'artist', 'language', 'lyrics_sentiment', 'emotion_tag']
    for col in cat_cols:
        df[col] = df[col].astype('category').cat.codes

    input_vector = df[feature_cols].mean().values.reshape(1, -1)
    predictions = user_pref_model.predict(input_vector)[0]

    return {"tempo": predictions[0], "energy": predictions[1]}
 """
# Main function
def main():
    user_id = int(input("Enter user ID: "))

    # Step 1: Face emotion
    face_emotion = get_face_emotion()

    # Step 2: Voice emotion
    record_audio()
    voice_emotion, _ = predict_voice_emotion("live_audio.wav")

    # Step 3: Historical mood
    mood_emotion = predict_mood_cluster(user_id)  # ✅ fixed

    # Step 4: Final emotion
    final_emotion = combine_emotions(face_emotion, voice_emotion, mood_emotion)
    print(f"\n🎯 Final Predicted Emotion: {final_emotion}")

    # Step 5: User preference (no CSV saving)
    """ user_pref = get_user_pref_vector()

    # Step 6: Recommend songs
    playlist = recommend_song_list(final_emotion, songs_df, user_pref, top_n=5)
    print("\n🎵 Recommended Songs:")
    print(playlist) """
if __name__ == "__main__":
    main()
