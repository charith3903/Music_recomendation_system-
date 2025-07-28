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
import os
from colorama import Fore, Back, Style, init

# Initialize colorama for cross-platform colored output
init(autoreset=True)

class TerminalUI:
    """Beautiful terminal UI class with colors and formatting"""
    
    # Color definitions
    HEADER = Fore.CYAN + Style.BRIGHT
    SUCCESS = Fore.GREEN + Style.BRIGHT
    WARNING = Fore.YELLOW + Style.BRIGHT
    ERROR = Fore.RED + Style.BRIGHT
    INFO = Fore.BLUE + Style.BRIGHT
    EMOTION = Fore.MAGENTA + Style.BRIGHT
    MUSIC = Fore.GREEN
    ACCENT = Fore.CYAN
    
    @staticmethod
    def clear_screen():
        os.system('cls' if os.name == 'nt' else 'clear')
    
    @staticmethod
    def print_banner():
        banner = f"""
{Fore.CYAN}{'='*80}
{Fore.MAGENTA + Style.BRIGHT}
    ♪♫♪ EMOTIONAL MUSIC RECOMMENDATION SYSTEM ♪♫♪
    
    🎵 Analyzing your emotions through face, voice & history
    🎧 Creating personalized playlists just for you
    ✨ Experience music that matches your soul
{Fore.CYAN}
{'='*80}{Style.RESET_ALL}
        """
        print(banner)
    
    @staticmethod
    def print_section_header(title, icon="🔥"):
        print(f"\n{Fore.YELLOW + Style.BRIGHT}{'─'*60}")
        print(f"{icon} {title.upper()}")
        print(f"{'─'*60}{Style.RESET_ALL}")
    
    @staticmethod
    def print_step(step_num, title, icon="⚡"):
        print(f"\n{Fore.CYAN + Style.BRIGHT}[STEP {step_num}] {icon} {title}{Style.RESET_ALL}")
    
    @staticmethod
    def print_emotion_result(emotion_type, emotion, confidence=None):
        confidence_str = f" ({confidence:.1%} confidence)" if confidence else ""
        print(f"  {Fore.GREEN}✓{Style.RESET_ALL} {emotion_type}: {Fore.MAGENTA + Style.BRIGHT}{emotion}{Style.RESET_ALL}{confidence_str}")
    
    @staticmethod
    def print_loading(text, duration=2):
        print(f"{Fore.YELLOW}⏳ {text}", end="", flush=True)
        for i in range(duration * 2):
            print(".", end="", flush=True)
            time.sleep(0.5)
        print(f" {Fore.GREEN}Done!{Style.RESET_ALL}")
    
    @staticmethod
    def print_playlist_header():
        header = f"""
{Fore.MAGENTA + Style.BRIGHT + Back.BLACK}
╔════════════════════════════════════════════════════════════════════════════╗
║                        🎵 YOUR PERSONALIZED PLAYLIST 🎵                   ║
╚════════════════════════════════════════════════════════════════════════════╝
{Style.RESET_ALL}
        """
        print(header)
    
    @staticmethod
    def print_song_item(rank, song, artist, similarity, mood):
        # Create a visual similarity bar
        bar_length = 20
        filled_length = int(bar_length * similarity)
        bar = "█" * filled_length + "░" * (bar_length - filled_length)
        mood_name = str(mood).title() if mood else "Unknown"
        print(f"""
{Fore.CYAN}#{rank:2d}{Style.RESET_ALL} │ {Fore.WHITE + Style.BRIGHT}{song[:35]:<35}{Style.RESET_ALL}
     │ {Fore.YELLOW}🎤 {artist[:30]:<30}{Style.RESET_ALL}

     │ {Fore.MAGENTA + Style.BRIGHT}Mood: {mood_name}{Style.RESET_ALL}
     └{'─'*70}""")
    
    @staticmethod
    def print_summary_box(final_emotion, total_songs, processing_time):
        summary = f"""
{Fore.GREEN + Style.BRIGHT + Back.BLACK}
╔══════════════════════════════════════════════════════════════════════════════╗
║                               📊 SESSION SUMMARY                            ║
║                                                                              ║
║  🎯 Detected Emotion: {final_emotion:<20}                                  ║
║  🎵 Songs Recommended: {total_songs:<19}                                   ║
║  ⏱️  Processing Time: {processing_time:<6.1f} seconds                              ║
║  ✨ Status: Playlist Ready!                                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝
{Style.RESET_ALL}
        """
        print(summary)
    
    @staticmethod
    def print_footer():
        footer = f"""
{Fore.CYAN}
╔════════════════════════════════════════════════════════════════════════════╗
║  🎧 Enjoy your personalized playlist!                                     ║
║  💫 Music has the power to heal, inspire, and connect souls               ║
║  🌟 Thank you for using our Emotional Music Recommendation System         ║
╚════════════════════════════════════════════════════════════════════════════╝
{Style.RESET_ALL}
        """
        print(footer)

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
    print(f"  {Fore.YELLOW}🎙️ Recording audio for {duration} seconds...{Style.RESET_ALL}")
    
    # Countdown
    for i in range(3, 0, -1):
        print(f"    {Fore.RED + Style.BRIGHT}{i}{Style.RESET_ALL}", end="", flush=True)
        time.sleep(1)
        if i > 1:
            print(" - ", end="", flush=True)
    
    print(f"\n  {Fore.GREEN}🔴 RECORDING NOW!{Style.RESET_ALL}")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    wav.write(filename, fs, audio)
    print(f"  {Fore.GREEN}✓ Audio recorded successfully!{Style.RESET_ALL}")

def extract_voice_features(file):
    y, sr = librosa.load(file)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return np.mean(mfccs.T, axis=0)

def predict_voice_emotion(file):
    print(f"  {Fore.BLUE}🧠 Analyzing voice patterns...{Style.RESET_ALL}")
    features = extract_voice_features(file).reshape(1, -1)
    pred = voice_model.predict(features)[0]
    confidence = voice_model.predict_proba(features).max()
    return pred.capitalize(), confidence

def get_face_emotion():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print(f"  {Fore.RED}❌ Cannot open webcam.{Style.RESET_ALL}")
        return None

    print(f"  {Fore.YELLOW}📷 Position yourself in front of camera and press 'q' to start{Style.RESET_ALL}")
    emotions_list, analyzing, start_time = [], False, None

    while True:
        ret, frame = cap.read()
        if not ret: break
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') and not analyzing:
            analyzing, start_time = True, time.time()
            print(f"  {Fore.GREEN}🧠 Analyzing facial emotions for 15 seconds...{Style.RESET_ALL}")

        if analyzing:
            try:
                results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                dominant = results[0]['dominant_emotion'] if isinstance(results, list) else results['dominant_emotion']
                dominant = dominant.capitalize()
                emotions_list.append(dominant)
                cv2.putText(frame, f"Emotion: {dominant}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
            except:
                cv2.putText(frame, "No face detected", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            elapsed = time.time() - start_time
            cv2.putText(frame, f"Time left: {15-int(elapsed)}s", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
            if elapsed >= 15: break
        else:
            cv2.putText(frame, "Press 'q' to start analysis", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

        cv2.imshow("Facial Emotion Detection", frame)

    cap.release()
    cv2.destroyAllWindows()

    if emotions_list:
        most_common = Counter(emotions_list).most_common(1)[0][0]
        print(f"  {Fore.GREEN}✓ Facial emotion analysis completed!{Style.RESET_ALL}")
        return most_common
    return None

def predict_mood_cluster(user_id):
    print(f"  {Fore.BLUE}📊 Analyzing your historical listening patterns...{Style.RESET_ALL}")
    df = pd.read_csv("dataset/user_7_day_listening_history_sample.csv")
    user_data = df[df['user_id'] == user_id]
    if user_data.empty: 
        print(f"  {Fore.YELLOW}⚠️ No historical data found for user {user_id}{Style.RESET_ALL}")
        return None
    
    agg = user_data.groupby('user_id').agg({
        "tempo": "mean", "energy": "mean", "valence": "mean", "danceability": "mean",
        "acousticness": "mean", "time_of_day": lambda x: x.mode()[0], "emotion_tag": lambda x: x.mode()[0]
    }).reset_index(drop=True)
    
    pred_cluster = mood_pipeline.predict(agg)[0]
    mood = cluster_labels.get(pred_cluster, "Unknown")
    print(f"  {Fore.GREEN}✓ Historical mood pattern analyzed!{Style.RESET_ALL}")
    return mood

def combine_emotions(face_emotion, voice_emotion, mood_emotion):
    print(f"\n  {Fore.CYAN}🔄 Combining all emotional indicators...{Style.RESET_ALL}")
    
    all_emotions = [e for e in [face_emotion, voice_emotion, mood_emotion] if e]
    if not all_emotions:
        return "Unknown"
    
    final = Counter(all_emotions).most_common(1)[0][0]
    
    # Show the combination process
    print(f"    • Face Emotion: {Fore.MAGENTA}{face_emotion or 'Not detected'}{Style.RESET_ALL}")
    print(f"    • Voice Emotion: {Fore.MAGENTA}{voice_emotion or 'Not detected'}{Style.RESET_ALL}")
    print(f"    • Historical Mood: {Fore.MAGENTA}{mood_emotion or 'Not available'}{Style.RESET_ALL}")
    
    print(f"  {Fore.GREEN}✓ Final emotion determined!{Style.RESET_ALL}")
    return final

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
    df = pd.read_csv("dataset/predicted_user_history.csv")
    user_pref = {
        "tempo": df["tempo"].mean(),
        "energy": df["energy"].mean()
    }
    return user_pref

def calculate_similarity(song_features, user_pref):
    song_vec = np.array([song_features['tempo'], song_features['energy']])
    user_vec = np.array([user_pref['tempo'], user_pref['energy']])
    song_norm = song_vec / np.linalg.norm(song_vec)
    user_norm = user_vec / np.linalg.norm(user_vec)
    return np.dot(song_norm, user_norm)

def recommend_song_list(emotion, songs_df, user_pref, top_n=10):
    playlist_types = recommend_playlist_types(emotion)
    
    print(f"  {Fore.BLUE}🎯 User Emotion: {Fore.RED + Style.BRIGHT}{emotion}{Style.RESET_ALL}")
    print(f"  {Fore.BLUE}💊 Doctor Recommended Song Moods: {Fore.GREEN + Style.BRIGHT}{', '.join(playlist_types)}{Style.RESET_ALL}")

    # Normalize and strip whitespaces
    songs_df['major_feeling'] = songs_df['major_feeling'].astype(str).str.strip().str.lower()
    songs_df['second_major_feeling'] = songs_df['second_major_feeling'].astype(str).str.strip().str.lower()
    playlist_types = [pt.lower() for pt in playlist_types]

    # STRICT FILTERING: Only include songs where major_feeling matches doctor's recommendations
    # This ensures ALL users get ONLY doctor-recommended mood songs
    filtered_songs = songs_df[songs_df['major_feeling'].isin(playlist_types)]

    print(f"  {Fore.BLUE}🔍 Found {len(filtered_songs)} songs matching doctor's recommendations{Style.RESET_ALL}")
    
    # Show breakdown of moods in results for verification
    if not filtered_songs.empty:
        mood_counts = filtered_songs['major_feeling'].value_counts()
        print(f"  {Fore.GREEN}✅ Playlist will contain:{Style.RESET_ALL}")
        for mood, count in mood_counts.items():
            print(f"    • {mood.title()}: {count} songs")

    if filtered_songs.empty:
        print(f"  {Fore.RED}⚠️ No songs matched the doctor's recommendations{Style.RESET_ALL}")
        return pd.DataFrame()

    filtered_songs = filtered_songs.copy()
    filtered_songs['similarity'] = filtered_songs.apply(
        lambda row: calculate_similarity(row, user_pref), axis=1
    )

    return filtered_songs.sort_values(by='similarity', ascending=False).head(top_n)

# Update your generate_playlist_csv function to use major_feeling for display:
def generate_playlist_csv(emotion, user_pref_path, mood_song_path, output_path):
    start_time = time.time()
    
    TerminalUI.print_step(5, "GENERATING PERSONALIZED PLAYLIST", "🎵")
    
    songs_df = pd.read_csv(mood_song_path)
    user_pref = get_user_preference_vector()

    print(f"  {Fore.BLUE}🔧 Calculating song compatibility...{Style.RESET_ALL}")
    playlist_df = recommend_song_list(emotion, songs_df, user_pref, top_n=10)

    if playlist_df.empty:
        print(f"  {Fore.RED}⚠️ No songs matched your emotion and preferences.{Style.RESET_ALL}")
        return 0, time.time() - start_time

    # Display beautiful playlist
    TerminalUI.print_playlist_header()
    
    for idx, (_, song) in enumerate(playlist_df.iterrows(), 1):
        TerminalUI.print_song_item(
            rank=idx,
            song=song['song'],
            artist=song['singer'],
            similarity=song['similarity'],
            mood=song['major_feeling'].title()  # Only show doctor-recommended mood
        )

    # Save to CSV
    playlist_df.to_csv(output_path, index=False)
    print(f"\n  {Fore.GREEN}✅ Playlist saved to: {Fore.CYAN}{output_path}{Style.RESET_ALL}")
    
    return len(playlist_df), time.time() - start_time
# === MAIN ===
def main():
    start_time = time.time()
    
    # Clear screen and show banner
    TerminalUI.clear_screen()
    TerminalUI.print_banner()
    
    # Get user input with style
    print(f"{Fore.YELLOW}👤 Please enter your User ID: {Style.RESET_ALL}", end="")
    user_id = int(input())
    
    print(f"\n{Fore.GREEN}✨ Welcome User #{user_id}! Let's discover your perfect playlist...{Style.RESET_ALL}")
    
    # Step 1: Face emotion detection
    TerminalUI.print_step(1, "FACIAL EMOTION DETECTION", "📸")
    face_emotion = get_face_emotion()
    if face_emotion:
        TerminalUI.print_emotion_result("Facial Emotion", face_emotion)
    
    # Step 2: Voice emotion detection  
    TerminalUI.print_step(2, "VOICE EMOTION ANALYSIS", "🎤")
    record_audio()
    voice_emotion, voice_confidence = predict_voice_emotion("live_audio.wav")
    TerminalUI.print_emotion_result("Voice Emotion", voice_emotion, voice_confidence)
    
    # Step 3: Historical mood analysis
    TerminalUI.print_step(3, "HISTORICAL MOOD ANALYSIS", "📊")
    mood_emotion = predict_mood_cluster(user_id)
    if mood_emotion:
        TerminalUI.print_emotion_result("Historical Mood", mood_emotion)
    
    # Step 4: Combine emotions
    TerminalUI.print_step(4, "EMOTION FUSION & ANALYSIS", "🧠")
    final_emotion = combine_emotions(face_emotion, voice_emotion, mood_emotion)
    
    print(f"\n  {Fore.MAGENTA + Style.BRIGHT + Back.BLACK}🎯 FINAL DETECTED EMOTION: {final_emotion.upper()} 🎯{Style.RESET_ALL}")
    
    # Step 5: Generate playlist
    user_pref_path = "dataset/user_history.csv"
    mood_song_path = "mood_based_song_analysis/predicted_moods.csv"
    output_path = "dataset/personalized_playlist.csv"

    playlist_count, processing_time = generate_playlist_csv(final_emotion, user_pref_path, mood_song_path, output_path)
    
    # Show summary
    total_time = time.time() - start_time
    TerminalUI.print_summary_box(final_emotion, playlist_count, total_time)
    
    # Footer
    TerminalUI.print_footer()

if __name__ == "__main__":
    main()