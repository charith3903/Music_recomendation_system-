import cv2
import sounddevice as sd
import scipy.io.wavfile as wav
import librosa
import numpy as np
import joblib
import pandas as pd
import time
from collections import Counter, defaultdict
from deepface import DeepFace
from keras.models import load_model
import os
from colorama import Fore, Back, Style, init
import matplotlib.pyplot as plt
from scipy.special import softmax
import warnings
warnings.filterwarnings('ignore')

# Initialize colorama for cross-platform colored output
init(autoreset=True)

class WeightedProbabilisticFusion:
    """Advanced emotion fusion using weighted probabilistic approach"""
    
    def __init__(self):
        self.emotion_classes = ['Happy', 'Sad', 'Angry', 'Fear', 'Surprise', 'Disgust', 'Neutral']
        self.model_weights = {
            'face': 0.35,      # Face detection weight
            'voice': 0.40,     # Voice analysis weight  
            'history': 0.25    # Historical pattern weight
        }
        self.confidence_threshold = 0.3  # Minimum confidence to consider
        
    def normalize_emotion(self, emotion):
        """Normalize different emotion naming conventions"""
        emotion_mapping = {
            'happiness': 'Happy', 'happy': 'Happy',
            'sadness': 'Sad', 'sad': 'Sad',
            'anger': 'Angry', 'angry': 'Angry',
            'fear': 'Fear', 'scared': 'Fear',
            'surprise': 'Surprise', 'surprised': 'Surprise',
            'disgust': 'Disgust', 'disgusted': 'Disgust',
            'neutral': 'Neutral'
        }
        return emotion_mapping.get(emotion.lower(), emotion.capitalize())
    
    def create_probability_vector(self, emotion, confidence):
        """Create probability distribution for given emotion and confidence"""
        prob_vector = np.zeros(len(self.emotion_classes))
        
        if emotion and confidence > self.confidence_threshold:
            normalized_emotion = self.normalize_emotion(emotion)
            if normalized_emotion in self.emotion_classes:
                idx = self.emotion_classes.index(normalized_emotion)
                # Primary emotion gets most probability
                prob_vector[idx] = confidence
                # Distribute remaining probability among other emotions
                remaining_prob = (1 - confidence) / (len(self.emotion_classes) - 1)
                for i in range(len(self.emotion_classes)):
                    if i != idx:
                        prob_vector[i] = remaining_prob
        else:
            # Uniform distribution if low confidence
            prob_vector = np.ones(len(self.emotion_classes)) / len(self.emotion_classes)
            
        return prob_vector
    
    def fuse_emotions(self, face_result, voice_result, history_result):
        """
        Weighted probabilistic fusion of multiple emotion detection results
        
        Args:
            face_result: tuple (emotion, confidence, probability_distribution)
            voice_result: tuple (emotion, confidence, probability_distribution) 
            history_result: tuple (emotion, confidence, probability_distribution)
        """
        
        # Extract results
        face_emotion, face_conf, face_probs = face_result if face_result else (None, 0.0, None)
        voice_emotion, voice_conf, voice_probs = voice_result if voice_result else (None, 0.0, None)
        history_emotion, history_conf, history_probs = history_result if history_result else (None, 0.0, None)
        
        # Create probability vectors
        face_prob_vec = face_probs if face_probs is not None else self.create_probability_vector(face_emotion, face_conf)
        voice_prob_vec = voice_probs if voice_probs is not None else self.create_probability_vector(voice_emotion, voice_conf)
        history_prob_vec = history_probs if history_probs is not None else self.create_probability_vector(history_emotion, history_conf)
        
        # Adjust weights based on confidence
        dynamic_weights = self.calculate_dynamic_weights(face_conf, voice_conf, history_conf)
        
        # Weighted fusion
        fused_probabilities = (
            dynamic_weights['face'] * face_prob_vec +
            dynamic_weights['voice'] * voice_prob_vec +
            dynamic_weights['history'] * history_prob_vec
        )
        
        # Normalize probabilities
        fused_probabilities = fused_probabilities / np.sum(fused_probabilities)
        
        # Get final emotion and confidence
        final_emotion_idx = np.argmax(fused_probabilities)
        final_emotion = self.emotion_classes[final_emotion_idx]
        final_confidence = fused_probabilities[final_emotion_idx]
        
        fusion_details = {
            'final_emotion': final_emotion,
            'final_confidence': final_confidence,
            'probability_distribution': fused_probabilities,
            'individual_results': {
                'face': {'emotion': face_emotion, 'confidence': face_conf},
                'voice': {'emotion': voice_emotion, 'confidence': voice_conf},
                'history': {'emotion': history_emotion, 'confidence': history_conf}
            },
            'weights_used': dynamic_weights
        }
        
        return fusion_details
    
    def calculate_dynamic_weights(self, face_conf, voice_conf, history_conf):
        """Calculate dynamic weights based on individual model confidences"""
        
        # Base weights
        weights = self.model_weights.copy()
        
        # Boost weights for high-confidence predictions
        confidence_boost = 0.2
        
        if face_conf > 0.8:
            weights['face'] += confidence_boost
        if voice_conf > 0.8:
            weights['voice'] += confidence_boost
        if history_conf > 0.8:
            weights['history'] += confidence_boost
            
        # Reduce weights for low-confidence predictions
        if face_conf < 0.4:
            weights['face'] *= 0.7
        if voice_conf < 0.4:
            weights['voice'] *= 0.7
        if history_conf < 0.4:
            weights['history'] *= 0.7
        
        # Normalize weights to sum to 1
        total_weight = sum(weights.values())
        weights = {k: v/total_weight for k, v in weights.items()}
        
        return weights

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
{Fore.CYAN}{'='*85}
{Fore.MAGENTA + Style.BRIGHT}
    ♪♫♪ ADVANCED EMOTION-BASED MUSIC RECOMMENDATION SYSTEM ♪♫♪
    
    🎵 Multi-Modal Emotion Detection: Face + Voice + History
    🧠 Weighted Probabilistic Fusion Technology  
    🎧 AI-Powered Personalized Playlist Generation
    ✨ Experience music that truly matches your emotional state
{Fore.CYAN}
{'='*85}{Style.RESET_ALL}
        """
        print(banner)
    
    @staticmethod
    def print_section_header(title, icon="🔥"):
        print(f"\n{Fore.YELLOW + Style.BRIGHT}{'─'*70}")
        print(f"{icon} {title.upper()}")
        print(f"{'─'*70}{Style.RESET_ALL}")
    
    @staticmethod
    def print_step(step_num, title, icon="⚡"):
        print(f"\n{Fore.CYAN + Style.BRIGHT}[STEP {step_num}] {icon} {title}{Style.RESET_ALL}")
    
    @staticmethod
    def print_emotion_result(emotion_type, emotion, confidence=None):
        confidence_str = f" ({confidence:.1%} confidence)" if confidence else ""
        print(f"  {Fore.GREEN}✓{Style.RESET_ALL} {emotion_type}: {Fore.MAGENTA + Style.BRIGHT}{emotion}{Style.RESET_ALL}{confidence_str}")
    
    @staticmethod
    def print_fusion_results(fusion_details):
        """Display detailed fusion analysis results"""
        print(f"\n{Fore.YELLOW + Style.BRIGHT}🧠 WEIGHTED PROBABILISTIC FUSION ANALYSIS{Style.RESET_ALL}")
        print("─" * 70)
        
        # Individual model results
        individual = fusion_details['individual_results']
        weights = fusion_details['weights_used']
        
        print(f"\n{Fore.CYAN}📊 Individual Model Results:{Style.RESET_ALL}")
        for model, result in individual.items():
            emotion = result['emotion'] or 'Not detected'
            conf = result['confidence']
            weight = weights[model]
            print(f"  {model.upper():<8} │ {emotion:<10} │ Confidence: {conf:.1%} │ Weight: {weight:.1%}")
        
        # Probability distribution visualization
        print(f"\n{Fore.CYAN}🎯 Final Probability Distribution:{Style.RESET_ALL}")
        emotions = ['Happy', 'Sad', 'Angry', 'Fear', 'Surprise', 'Disgust', 'Neutral']
        probs = fusion_details['probability_distribution']
        
        for emotion, prob in zip(emotions, probs):
            bar_length = 20
            filled_length = int(bar_length * prob)
            bar = "█" * filled_length + "░" * (bar_length - filled_length)
            print(f"  {emotion:<10} │ {bar} {prob:.1%}")
        
        # Final result
        final_emotion = fusion_details['final_emotion']
        final_confidence = fusion_details['final_confidence']
        
        print(f"\n{Fore.GREEN + Style.BRIGHT + Back.BLACK}🎯 FINAL FUSED EMOTION: {final_emotion.upper()} ({final_confidence:.1%} confidence) 🎯{Style.RESET_ALL}")
    
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
        
        # Handle missing values with defaults
        song_name = str(song)[:35] if song else "Unknown Song"
        artist_name = str(artist)[:30] if artist else "Unknown Artist"
        mood_name = str(mood).title() if mood else "Unknown"
        similarity_val = float(similarity) if similarity else 0.0
        
        print(f"""
{Fore.CYAN}#{rank:2d}{Style.RESET_ALL} │ {Fore.WHITE + Style.BRIGHT}{song_name:<35}{Style.RESET_ALL}
     │ {Fore.YELLOW}🎤 {artist_name:<30}{Style.RESET_ALL}
     │ {Fore.GREEN}📊 Match: {bar} {similarity_val:.1%}{Style.RESET_ALL}
     │ {Fore.MAGENTA}💭 Mood: {mood_name}{Style.RESET_ALL}
     └{'─'*70}""")
    
    @staticmethod
    def print_summary_box(final_emotion, total_songs, processing_time, fusion_confidence):
        summary = f"""
{Fore.GREEN + Style.BRIGHT + Back.BLACK}
╔══════════════════════════════════════════════════════════════════════════════╗
║                               📊 SESSION SUMMARY                            ║
║                                                                              ║
║  🎯 Fused Emotion: {final_emotion:<20} (Confidence: {fusion_confidence:.1%})        ║
║  🎵 Songs Recommended: {total_songs:<19}                                   ║
║  ⏱️  Processing Time: {processing_time:<6.1f} seconds                              ║
║  🧠 Fusion Method: Weighted Probabilistic                                   ║
║  ✨ Status: Advanced Playlist Ready!                                        ║
╚══════════════════════════════════════════════════════════════════════════════╝
{Style.RESET_ALL}
        """
        print(summary)
    
    @staticmethod
    def print_footer():
        footer = f"""
{Fore.CYAN}
╔════════════════════════════════════════════════════════════════════════════╗
║  🎧 Enjoy your AI-generated personalized playlist!                        ║
║  💫 Advanced multi-modal emotion detection at your service                ║
║  🌟 Thank you for using our cutting-edge music recommendation system      ║
╚════════════════════════════════════════════════════════════════════════════╝
{Style.RESET_ALL}
        """
        print(footer)

# === Load Models ===
print(f"{Fore.CYAN}🔄 Loading AI models...{Style.RESET_ALL}")

try:
    voice_model = joblib.load("voice_emotion_model.pkl")
    print(f"{Fore.GREEN}✓ Voice emotion model loaded{Style.RESET_ALL}")
except Exception as e:
    print(f"{Fore.YELLOW}⚠️ Voice model not found, using fallback: {e}{Style.RESET_ALL}")
    voice_model = None

try:
    mood_pipeline = joblib.load("user_mood_kmeans_model.pkl")
    print(f"{Fore.GREEN}✓ Mood clustering model loaded{Style.RESET_ALL}")
except Exception as e:
    print(f"{Fore.YELLOW}⚠️ Mood model not found, using fallback: {e}{Style.RESET_ALL}")
    mood_pipeline = None

try:
    user_pref_model = load_model("user_preference_model.h5")
    print(f"{Fore.GREEN}✓ User preference model loaded{Style.RESET_ALL}")
except Exception as e:
    print(f"{Fore.YELLOW}⚠️ Preference model not found, using fallback: {e}{Style.RESET_ALL}")
    user_pref_model = None

# === Load Dataset ===
try:
    songs_df = pd.read_csv("mood_based_song_analysis/predicted_moods.csv")
    print(f"{Fore.GREEN}✓ Songs database loaded ({len(songs_df)} songs){Style.RESET_ALL}")
except Exception as e:
    print(f"{Fore.YELLOW}⚠️ Songs database not found, creating sample data: {e}{Style.RESET_ALL}")
    # Create sample songs data for testing
    songs_df = pd.DataFrame({
        'song': ['Song 1', 'Song 2', 'Song 3', 'Song 4', 'Song 5'],
        'singer': ['Artist 1', 'Artist 2', 'Artist 3', 'Artist 4', 'Artist 5'],
        'major_feeling': ['happy', 'sad', 'energetic', 'relaxing', 'romantic'],
        'second_major_feeling': ['energetic', 'relaxing', 'happy', 'ambient', 'happy'],
        'tempo': [120, 80, 140, 60, 100],
        'energy': [0.8, 0.3, 0.9, 0.2, 0.6]
    })

# === Cluster to Label Mapping ===
cluster_labels = {
    0: "Happy",
    1: "Sad", 
    2: "Angry",
    3: "Disgust",
    4: "Neutral",
    5: "Fear",
    6: "Surprise"
}

# Initialize fusion system
fusion_system = WeightedProbabilisticFusion()

# === Enhanced Helper Functions ===
def record_audio(filename="live_audio.wav", duration=5, fs=44100):
    """Record audio with enhanced feedback"""
    print(f"  {Fore.YELLOW}🎙️ Recording audio for {duration} seconds...{Style.RESET_ALL}")
    
    # Countdown
    for i in range(3, 0, -1):
        print(f"    {Fore.RED + Style.BRIGHT}{i}{Style.RESET_ALL}", end="", flush=True)
        time.sleep(1)
        if i > 1:
            print(" - ", end="", flush=True)
    
    print(f"\n  {Fore.GREEN}🔴 RECORDING NOW! Speak clearly...{Style.RESET_ALL}")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    wav.write(filename, fs, audio)
    print(f"  {Fore.GREEN}✓ Audio recorded successfully!{Style.RESET_ALL}")

def extract_voice_features(file):
    """Extract comprehensive voice features with error handling"""
    try:
        y, sr = librosa.load(file)
        
        # Extract MFCC features (most reliable)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        features = [np.mean(mfccs.T, axis=0)]
        
        # Try to extract additional features with error handling
        try:
            chroma = librosa.feature.chroma(y=y, sr=sr)
            features.append(np.mean(chroma.T, axis=0))
        except AttributeError:
            # Use chromagram if chroma doesn't exist
            try:
                chroma = librosa.feature.chromagram(y=y, sr=sr)
                features.append(np.mean(chroma.T, axis=0))
            except:
                # Skip chroma features if not available
                pass
        
        try:
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
            features.append(np.mean(spectral_centroid.T, axis=0))
        except:
            pass
            
        try:
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
            features.append(np.mean(zero_crossing_rate.T, axis=0))
        except:
            pass
        
        # Combine all available features
        if len(features) > 1:
            combined_features = np.concatenate(features)
        else:
            combined_features = features[0]
            
        return combined_features
        
    except Exception as e:
        print(f"    {Fore.YELLOW}⚠️ Feature extraction error: {e}{Style.RESET_ALL}")
        # Return dummy features if extraction fails
        return np.random.rand(40)  # Basic MFCC size

def predict_voice_emotion(file):
    """Enhanced voice emotion prediction with probability distribution"""
    print(f"  {Fore.BLUE}🧠 Analyzing voice patterns and acoustic features...{Style.RESET_ALL}")
    
    if voice_model is None:
        print(f"  {Fore.YELLOW}⚠️ Voice model not available, using random prediction{Style.RESET_ALL}")
        # Return random emotion for testing
        emotions = ['Happy', 'Sad', 'Angry', 'Fear', 'Surprise', 'Disgust', 'Neutral']
        emotion = np.random.choice(emotions)
        confidence = np.random.uniform(0.6, 0.9)
        probs = np.random.dirichlet(np.ones(len(fusion_system.emotion_classes)))
        return emotion, confidence, probs
    
    try:
        features = extract_voice_features(file).reshape(1, -1)
        
        # Get prediction and probabilities
        pred = voice_model.predict(features)[0]
        probabilities = voice_model.predict_proba(features)[0]
        confidence = probabilities.max()
        
        # Create probability distribution for all emotion classes
        emotion_probs = np.zeros(len(fusion_system.emotion_classes))
        
        # Map model output to fusion system emotions
        model_emotions = voice_model.classes_
        for i, model_emotion in enumerate(model_emotions):
            normalized_emotion = fusion_system.normalize_emotion(model_emotion)
            if normalized_emotion in fusion_system.emotion_classes:
                idx = fusion_system.emotion_classes.index(normalized_emotion)
                emotion_probs[idx] = probabilities[i]
        
        return pred.capitalize(), confidence, emotion_probs
        
    except Exception as e:
        print(f"  {Fore.RED}❌ Voice analysis failed: {e}{Style.RESET_ALL}")
        return None, 0.0, None

def get_face_emotion_advanced():
    """Advanced face emotion detection with multiple frames analysis"""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print(f"  {Fore.RED}❌ Cannot open webcam.{Style.RESET_ALL}")
        return None, 0.0, None

    print(f"  {Fore.YELLOW}📷 Position yourself in front of camera and press 'q' to start{Style.RESET_ALL}")
    
    emotion_history = []
    confidence_history = []
    analyzing = False
    start_time = None
    analysis_duration = 20  # Increased for better accuracy

    while True:
        ret, frame = cap.read()
        if not ret: 
            break
            
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') and not analyzing:
            analyzing = True
            start_time = time.time()
            print(f"  {Fore.GREEN}🧠 Analyzing facial emotions for {analysis_duration} seconds...{Style.RESET_ALL}")

        if analyzing:
            try:
                # Analyze current frame
                results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                
                if isinstance(results, list):
                    emotions = results[0]['emotion']
                    dominant = results[0]['dominant_emotion']
                else:
                    emotions = results['emotion']
                    dominant = results['dominant_emotion']
                
                # Store emotion data
                emotion_history.append(emotions)
                confidence_history.append(emotions[dominant])
                
                # Display on frame
                cv2.putText(frame, f"Emotion: {dominant.capitalize()}", (20, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
                cv2.putText(frame, f"Confidence: {emotions[dominant]:.1%}", (20, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                
            except Exception as e:
                cv2.putText(frame, "No face detected", (20, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            elapsed = time.time() - start_time
            remaining = analysis_duration - int(elapsed)
            cv2.putText(frame, f"Time left: {remaining}s", (20, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
            
            if elapsed >= analysis_duration:
                break
        else:
            cv2.putText(frame, "Press 'q' to start analysis", (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

        cv2.imshow("Advanced Facial Emotion Detection", frame)

    cap.release()
    cv2.destroyAllWindows()

    if not emotion_history:
        return None, 0.0, None

    # Advanced emotion aggregation
    aggregated_emotions = defaultdict(list)
    
    for frame_emotions in emotion_history:
        for emotion, score in frame_emotions.items():
            aggregated_emotions[emotion].append(score)
    
    # Calculate average scores and confidence
    final_emotions = {}
    for emotion, scores in aggregated_emotions.items():
        final_emotions[emotion] = np.mean(scores)
    
    # Get dominant emotion and confidence
    dominant_emotion = max(final_emotions.keys(), key=lambda k: final_emotions[k])
    confidence = final_emotions[dominant_emotion] / 100.0  # Convert to 0-1 range
    
    # Create probability distribution
    emotion_probs = np.zeros(len(fusion_system.emotion_classes))
    for emotion, score in final_emotions.items():
        normalized_emotion = fusion_system.normalize_emotion(emotion)
        if normalized_emotion in fusion_system.emotion_classes:
            idx = fusion_system.emotion_classes.index(normalized_emotion)
            emotion_probs[idx] = score / 100.0
    
    # Normalize probabilities
    emotion_probs = emotion_probs / np.sum(emotion_probs)
    
    print(f"  {Fore.GREEN}✓ Facial emotion analysis completed! Analyzed {len(emotion_history)} frames{Style.RESET_ALL}")
    
    return dominant_emotion.capitalize(), confidence, emotion_probs

def predict_mood_cluster_advanced(user_id):
    """Enhanced historical mood analysis with confidence scoring"""
    print(f"  {Fore.BLUE}📊 Analyzing historical listening patterns and behavioral data...{Style.RESET_ALL}")
    
    if mood_pipeline is None:
        print(f"  {Fore.YELLOW}⚠️ Mood model not available, using random prediction{Style.RESET_ALL}")
        # Return random mood for testing
        emotions = ['Happy', 'Sad', 'Angry', 'Fear', 'Surprise', 'Disgust', 'Neutral']
        emotion = np.random.choice(emotions)
        confidence = np.random.uniform(0.5, 0.8)
        probs = np.random.dirichlet(np.ones(len(fusion_system.emotion_classes)))
        return emotion, confidence, probs
    
    try:
        # Try to load user data with better error handling
        try:
            df = pd.read_csv("dataset/user_7_day_listening_history_sample.csv")
        except FileNotFoundError:
            print(f"  {Fore.YELLOW}⚠️ Historical data file not found, creating sample data{Style.RESET_ALL}")
            # Create sample historical data
            sample_data = {
                'user_id': [user_id] * 10,
                'tempo': np.random.uniform(60, 180, 10),
                'energy': np.random.uniform(0.1, 1.0, 10),
                'valence': np.random.uniform(0.1, 1.0, 10),
                'danceability': np.random.uniform(0.1, 1.0, 10),
                'acousticness': np.random.uniform(0.1, 1.0, 10),
                'time_of_day': np.random.choice(['morning', 'afternoon', 'evening', 'night'], 10),
                'emotion_tag': np.random.choice(['happy', 'sad', 'energetic', 'calm', 'neutral'], 10)
            }
            df = pd.DataFrame(sample_data)
        
        user_data = df[df['user_id'] == user_id]
        
        if user_data.empty:
            print(f"  {Fore.YELLOW}⚠️ No historical data found for user {user_id}, using default pattern{Style.RESET_ALL}")
            # Return default mood
            default_emotions = ['Neutral', 'Happy']
            emotion = np.random.choice(default_emotions)
            confidence = 0.6
            probs = np.random.dirichlet(np.ones(len(fusion_system.emotion_classes)))
            return emotion, confidence, probs
        
        # Simplified feature aggregation to avoid column issues
        try:
            agg_data = {
                'tempo_mean': user_data['tempo'].mean(),
                'tempo_std': user_data['tempo'].std(),
                'energy_mean': user_data['energy'].mean(),
                'energy_std': user_data['energy'].std(),
                'valence_mean': user_data['valence'].mean() if 'valence' in user_data.columns else 0.5,
                'valence_std': user_data['valence'].std() if 'valence' in user_data.columns else 0.1,
                'danceability_mean': user_data['danceability'].mean() if 'danceability' in user_data.columns else 0.5,
                'danceability_std': user_data['danceability'].std() if 'danceability' in user_data.columns else 0.1,
                'acousticness_mean': user_data['acousticness'].mean() if 'acousticness' in user_data.columns else 0.5,
                'acousticness_std': user_data['acousticness'].std() if 'acousticness' in user_data.columns else 0.1,
            }
            
            # Use only mean features for prediction
            prediction_features = np.array([
                agg_data['tempo_mean'],
                agg_data['energy_mean'],
                agg_data['valence_mean'],
                agg_data['danceability_mean'],
                agg_data['acousticness_mean']
            ]).reshape(1, -1)
            
            # Make prediction
            pred_cluster = mood_pipeline.predict(prediction_features)[0]
            mood = cluster_labels.get(pred_cluster, "Neutral")
            
            # Calculate confidence based on data consistency
            consistency_scores = []
            for feature in ['tempo', 'energy', 'valence', 'danceability', 'acousticness']:
                if f'{feature}_std' in agg_data:
                    std_val = agg_data[f'{feature}_std']
                    # Lower std deviation indicates more consistent patterns
                    consistency = 1.0 / (1.0 + std_val) if std_val > 0 else 0.8
                    consistency_scores.append(consistency)
            
            confidence = np.mean(consistency_scores) if consistency_scores else 0.6
            
            # Create probability distribution
            emotion_probs = np.zeros(len(fusion_system.emotion_classes))
            if mood in fusion_system.emotion_classes:
                idx = fusion_system.emotion_classes.index(mood)
                emotion_probs[idx] = confidence
                # Distribute remaining probability
                remaining_prob = (1 - confidence) / (len(fusion_system.emotion_classes) - 1)
                for i in range(len(fusion_system.emotion_classes)):
                    if i != idx:
                        emotion_probs[i] = remaining_prob
            
            print(f"  {Fore.GREEN}✓ Historical analysis completed! Processed {len(user_data)} listening sessions{Style.RESET_ALL}")
            
            return mood, confidence, emotion_probs
            
        except Exception as model_error:
            print(f"  {Fore.YELLOW}⚠️ Model prediction failed: {model_error}{Style.RESET_ALL}")
            # Fallback based on user data patterns
            if not user_data.empty:
                # Simple heuristic based on energy and valence
                avg_energy = user_data['energy'].mean() if 'energy' in user_data.columns else 0.5
                if avg_energy > 0.7:
                    mood = "Happy"
                elif avg_energy < 0.3:
                    mood = "Sad"
                else:
                    mood = "Neutral"
                confidence = 0.6
            else:
                mood = "Neutral"
                confidence = 0.5
                
            probs = np.random.dirichlet(np.ones(len(fusion_system.emotion_classes)))
            return mood, confidence, probs
        
    except Exception as e:
        print(f"  {Fore.YELLOW}⚠️ Historical analysis failed, using fallback: {e}{Style.RESET_ALL}")
        # Fallback prediction
        emotions = ['Neutral', 'Happy', 'Sad']
        emotion = np.random.choice(emotions)
        confidence = 0.5
        probs = np.random.dirichlet(np.ones(len(fusion_system.emotion_classes)))
        return emotion, confidence, probs

def recommend_playlist_types(emotion):
    """Enhanced playlist recommendation based on psychological research"""
    doctor_recommendations = {
        "Angry": ["relaxing", "ambient"],
        "Sad": ["uplifting", "happy", "motivational"],
        "Neutral": ["relaxing", "romantic", "ambient"],
        "Happy": ["energetic", "happy", "upbeat"],
        "Fear": ["calming", "relaxing", "peaceful"],
        "Disgust": ["cleansing", "happy", "uplifting"],
        "Surprise": ["energetic", "upbeat", "dynamic"]
    }
    return doctor_recommendations.get(emotion, ["relaxing", "neutral"])

def get_user_preference_vector():
    """Enhanced user preference analysis"""
    try:
        df = pd.read_csv("dataset/predicted_user_history.csv")
        user_pref = {
            "tempo": df["tempo"].mean(),
            "energy": df["energy"].mean(),
            "valence": df["valence"].mean() if "valence" in df.columns else 0.5,
            "danceability": df["danceability"].mean() if "danceability" in df.columns else 0.5
        }
        return user_pref
    except Exception:
        # Default preferences
        return {"tempo": 120, "energy": 0.6, "valence": 0.5, "danceability": 0.5}

def calculate_similarity(song_features, user_pref):
    """Enhanced similarity calculation with multiple features"""
    try:
        song_vec = np.array([
            song_features.get('tempo', 120),
            song_features.get('energy', 0.5),
            song_features.get('valence', 0.5) if 'valence' in song_features else 0.5,
            song_features.get('danceability', 0.5) if 'danceability' in song_features else 0.5
        ])
        
        user_vec = np.array([
            user_pref['tempo'],
            user_pref['energy'],
            user_pref.get('valence', 0.5),
            user_pref.get('danceability', 0.5)
        ])
        
        # Normalize vectors
        song_norm = song_vec / (np.linalg.norm(song_vec) + 1e-8)
        user_norm = user_vec / (np.linalg.norm(user_vec) + 1e-8)
        
        # Calculate cosine similarity
        similarity = np.dot(song_norm, user_norm)
        return max(0, similarity)  # Ensure non-negative
        
    except Exception:
        return 0.5  # Default similarity

def recommend_song_list(emotion, songs_df, user_pref, top_n=10):
    """Enhanced song recommendation with better filtering"""
    playlist_types = recommend_playlist_types(emotion)
    
    print(f"  {Fore.BLUE}🎯 Target Emotion: {Fore.MAGENTA + Style.BRIGHT}{emotion}{Style.RESET_ALL}")
    print(f"  {Fore.BLUE}🧠 Recommended Playlist Types: {Fore.CYAN}{', '.join(playlist_types)}{Style.RESET_ALL}")

    # Normalize and strip whitespaces
    songs_df['major_feeling'] = songs_df['major_feeling'].astype(str).str.strip().str.lower()
    songs_df['second_major_feeling'] = songs_df['second_major_feeling'].astype(str).str.strip().str.lower()
    playlist_types = [pt.lower() for pt in playlist_types]

    # Filter based on doctor's recommendation
    filtered_songs = songs_df[
        (songs_df['major_feeling'].isin(playlist_types)) |
        (songs_df['second_major_feeling'].isin(playlist_types))
    ]

    print(f"  {Fore.BLUE}🔍 Found {len(filtered_songs)} matching songs{Style.RESET_ALL}")

    if filtered_songs.empty:
        print(f"  {Fore.YELLOW}⚠️ No exact matches found, using broader criteria...{Style.RESET_ALL}")
        # Fallback to all songs if no matches
        filtered_songs = songs_df.head(50)  # Use top 50 songs as fallback

    if filtered_songs.empty:
        print(f"  {Fore.RED}❌ No songs available in database{Style.RESET_ALL}")
        return pd.DataFrame()

    # Calculate similarity scores
    filtered_songs = filtered_songs.copy()
    filtered_songs['similarity'] = filtered_songs.apply(
        lambda row: calculate_similarity(row, user_pref), axis=1
    )

    # Sort by similarity and return top N
    result = filtered_songs.sort_values(by='similarity', ascending=False).head(top_n)
    
    print(f"  {Fore.GREEN}✓ Generated {len(result)} personalized recommendations{Style.RESET_ALL}")
    return result

def generate_playlist_csv(emotion, fusion_confidence, user_pref_path, mood_song_path, output_path):
    """Generate playlist with enhanced emotion-based filtering"""
    start_time = time.time()
    
    TerminalUI.print_step(5, "GENERATING AI-POWERED PERSONALIZED PLAYLIST", "🎵")
    
    try:
        songs_df = pd.read_csv(mood_song_path)
        user_pref = get_user_preference_vector()

        print(f"  {Fore.BLUE}🔧 Calculating song compatibility using AI algorithms...{Style.RESET_ALL}")
        playlist_df = recommend_song_list(emotion, songs_df, user_pref, top_n=12)

        if playlist_df.empty:
            print(f"  {Fore.RED}⚠️ No songs matched your emotion and preferences.{Style.RESET_ALL}")
            return 0, time.time() - start_time

        # Display beautiful playlist
        TerminalUI.print_playlist_header()
        
        for idx, (_, song) in enumerate(playlist_df.iterrows(), 1):
            TerminalUI.print_song_item(
                rank=idx,
                song=song.get('song', 'Unknown Song'),
                artist=song.get('singer', 'Unknown Artist'),
                similarity=song.get('similarity', 0.0),
                mood=song.get('major_feeling', 'Unknown').title()
            )

        # Save to CSV with additional metadata
        playlist_df['fusion_confidence'] = fusion_confidence
        playlist_df['generated_timestamp'] = pd.Timestamp.now()
        playlist_df['target_emotion'] = emotion
        
        playlist_df.to_csv(output_path, index=False)
        print(f"\n  {Fore.GREEN}✅ Advanced playlist saved to: {Fore.CYAN}{output_path}{Style.RESET_ALL}")
        
        return len(playlist_df), time.time() - start_time
        
    except Exception as e:
        print(f"  {Fore.RED}❌ Playlist generation failed: {e}{Style.RESET_ALL}")
        return 0, time.time() - start_time

# === ENHANCED MAIN FUNCTION ===
def main():
    """Main function with advanced weighted probabilistic fusion"""
    start_time = time.time()
    
    # Clear screen and show banner
    TerminalUI.clear_screen()
    TerminalUI.print_banner()
    
    # Get user input with enhanced validation
    while True:
        try:
            print(f"{Fore.YELLOW}👤 Please enter your User ID (1-1000): {Style.RESET_ALL}", end="")
            user_id = int(input())
            if 1 <= user_id <= 1000:
                break
            else:
                print(f"{Fore.RED}❌ Please enter a valid User ID between 1 and 1000{Style.RESET_ALL}")
        except ValueError:
            print(f"{Fore.RED}❌ Please enter a valid number{Style.RESET_ALL}")
    
    print(f"\n{Fore.GREEN}✨ Welcome User #{user_id}! Initializing advanced emotion detection...{Style.RESET_ALL}")
    print(f"{Fore.CYAN}🧠 Using Weighted Probabilistic Fusion for maximum accuracy{Style.RESET_ALL}")
    
    # Step 1: Advanced Face emotion detection
    TerminalUI.print_step(1, "ADVANCED FACIAL EMOTION DETECTION", "📸")
    print(f"  {Fore.BLUE}ℹ️ Multi-frame analysis for enhanced accuracy{Style.RESET_ALL}")
    
    face_result = get_face_emotion_advanced()
    if face_result[0]:
        TerminalUI.print_emotion_result("Facial Emotion", face_result[0], face_result[1])
    else:
        print(f"  {Fore.YELLOW}⚠️ Facial emotion detection failed{Style.RESET_ALL}")
    
    # Step 2: Enhanced Voice emotion detection  
    TerminalUI.print_step(2, "ADVANCED VOICE EMOTION ANALYSIS", "🎤")
    print(f"  {Fore.BLUE}ℹ️ Multi-feature acoustic analysis (MFCC, Chroma, Spectral){Style.RESET_ALL}")
    
    record_audio(duration=5)  # Longer recording for better accuracy
    voice_result = predict_voice_emotion("live_audio.wav")
    if voice_result[0]:
        TerminalUI.print_emotion_result("Voice Emotion", voice_result[0], voice_result[1])
    else:
        print(f"  {Fore.YELLOW}⚠️ Voice emotion detection failed{Style.RESET_ALL}")
    
    # Step 3: Enhanced Historical mood analysis
    TerminalUI.print_step(3, "ADVANCED HISTORICAL PATTERN ANALYSIS", "📊")
    print(f"  {Fore.BLUE}ℹ️ Behavioral pattern recognition and consistency analysis{Style.RESET_ALL}")
    
    history_result = predict_mood_cluster_advanced(user_id)
    if history_result[0]:
        TerminalUI.print_emotion_result("Historical Pattern", history_result[0], history_result[1])
    else:
        print(f"  {Fore.YELLOW}⚠️ Historical analysis unavailable{Style.RESET_ALL}")
    
    # Step 4: Weighted Probabilistic Fusion
    TerminalUI.print_step(4, "WEIGHTED PROBABILISTIC FUSION", "🧠")
    print(f"  {Fore.BLUE}ℹ️ Combining multi-modal data using advanced AI fusion algorithms{Style.RESET_ALL}")
    
    # Perform fusion
    fusion_details = fusion_system.fuse_emotions(face_result, voice_result, history_result)
    
    # Display detailed fusion results
    TerminalUI.print_fusion_results(fusion_details)
    
    final_emotion = fusion_details['final_emotion']
    fusion_confidence = fusion_details['final_confidence']
    
    # Step 5: Generate AI-powered playlist
    user_pref_path = "dataset/user_history.csv"
    mood_song_path = "mood_based_song_analysis/predicted_moods.csv"
    output_path = "dataset/personalized_advanced_playlist.csv"

    playlist_count, processing_time = generate_playlist_csv(
        final_emotion, fusion_confidence, user_pref_path, mood_song_path, output_path
    )
    
    # Show enhanced summary
    total_time = time.time() - start_time
    TerminalUI.print_summary_box(final_emotion, playlist_count, total_time, fusion_confidence)
    
    # Additional fusion statistics
    print(f"\n{Fore.CYAN + Style.BRIGHT}📈 FUSION STATISTICS:{Style.RESET_ALL}")
    weights = fusion_details['weights_used']
    print(f"  • Face Model Weight: {weights['face']:.1%}")
    print(f"  • Voice Model Weight: {weights['voice']:.1%}")  
    print(f"  • History Model Weight: {weights['history']:.1%}")
    print(f"  • Overall Fusion Confidence: {fusion_confidence:.1%}")
    
    # Footer
    TerminalUI.print_footer()
    
    # Optional: Save fusion details for analysis
    fusion_summary = {
        'user_id': user_id,
        'timestamp': pd.Timestamp.now(),
        'final_emotion': final_emotion,
        'fusion_confidence': fusion_confidence,
        'face_emotion': face_result[0] if face_result else None,
        'face_confidence': face_result[1] if face_result else 0.0,
        'voice_emotion': voice_result[0] if voice_result else None,
        'voice_confidence': voice_result[1] if voice_result else 0.0,
        'history_emotion': history_result[0] if history_result else None,
        'history_confidence': history_result[1] if history_result else 0.0,
        'weights_used': weights,
        'playlist_count': playlist_count,
        'processing_time': total_time
    }
    
    # Save fusion log
    fusion_log_path = "logs/fusion_analysis_log.csv"
    try:
        os.makedirs("logs", exist_ok=True)
        fusion_df = pd.DataFrame([fusion_summary])
        
        if os.path.exists(fusion_log_path):
            fusion_df.to_csv(fusion_log_path, mode='a', header=False, index=False)
        else:
            fusion_df.to_csv(fusion_log_path, index=False)
            
        print(f"\n{Fore.GREEN}📝 Fusion analysis saved to: {fusion_log_path}{Style.RESET_ALL}")
    except Exception as e:
        print(f"\n{Fore.YELLOW}⚠️ Could not save fusion log: {e}{Style.RESET_ALL}")

if __name__ == "__main__":
    main()