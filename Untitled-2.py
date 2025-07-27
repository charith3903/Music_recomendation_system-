# %%
pip install kagglehub librosa sounddevice scikit-learn joblib numpy tqdm


# %%
import os
import librosa
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import joblib
import kagglehub

# %%
# Step 1: Download Dataset
print("[INFO] Downloading dataset from Kaggle...")
path = kagglehub.dataset_download("uwrfkaggler/ravdess-emotional-speech-audio")
print("[INFO] Dataset downloaded to:", path)


# %%
# Step 2: Emotion labels
emotion_map = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fear",
    "07": "disgust",
    "08": "surprise"
}

# %%
def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        mfccs_scaled = np.mean(mfccs.T, axis=0)
        return mfccs_scaled
    except Exception as e:
        print(f"[WARN] Could not process {file_path}: {e}")
        return None

features = []
labels = []

# %%

pip install librosa==0.10.1 resampy==0.4.2 numpy scipy soundfile audioread


# %%
# Step 3: Traverse audio files and extract features

import os
import librosa
from tqdm import tqdm
import numpy as np
print("[INFO] Extracting features...")
for root, dirs, files in os.walk(path):
    for file in tqdm(files):
        if file.endswith(".wav"):
            emotion_code = file.split("-")[2]
            emotion = emotion_map.get(emotion_code)
            if emotion:
                file_path = os.path.join(root, file)
                mfccs = extract_features(file_path)
                if mfccs is not None:
                    features.append(mfccs)
                    labels.append(emotion)

# %%
# Step 4: Train/test split
print("[INFO] Training model...")
X = np.array(features)
y = np.array(labels)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# %%
# Step 5: Train classifier (SVM or Random Forest)
model = SVC(kernel='linear', probability=True)
model.fit(X_train, y_train)


# %%
# Step 6: Evaluate
print("[INFO] Model evaluation:")
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))


# %%
joblib.dump(model, "voice_emotion_model.pkl")
print("[INFO] Model saved as voice_emotion_model.pkl")

# %%
import sounddevice as sd
import librosa
import numpy as np
import scipy.io.wavfile as wav
import joblib

def record_audio(filename="live_audio.wav", duration=3, fs=44100):
    print("🎙️ Recording...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    wav.write(filename, fs, audio)
    print("✅ Recording saved.")

def extract_features(file):
    y, sr = librosa.load(file)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return np.mean(mfccs.T, axis=0)

def predict_emotion(file):
    model = joblib.load("voice_emotion_model.pkl")
    features = extract_features(file).reshape(1, -1)
    prediction = model.predict(features)
    proba = model.predict_proba(features).max()
    return prediction[0], round(proba * 100, 2)

# Run detection
record_audio("live_audio.wav", duration=3)
emotion, confidence = predict_emotion("live_audio.wav")
print(f"🎭 Detected Emotion: {emotion} ({confidence}%)")


# %%



