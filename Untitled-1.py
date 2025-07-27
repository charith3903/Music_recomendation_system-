# %%
pip install pandas scikit-learn tensorflow


# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# %%
# Load the dataset
df = pd.read_csv("dataset/personalized_music_recommendation_dataset.csv")

# %%
features = [
    "genre", "artist", "language", "tempo", "energy", "danceability", 
    "acousticness", "instrumentalness", "valence", "liveness", 
    "speechiness", "loudness", "lyrics_sentiment", "emotion_tag"
]

# %%
target = "liked"

# %%
# Drop rows with missing values in selected columns
df = df[features + [target]].dropna()


# %%
# Encode categorical features
label_encoders = {}
categorical_cols = ["genre", "artist", "language", "lyrics_sentiment", "emotion_tag"]

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# %%
# Separate features and labels
X = df[features]
y = df[target]

# Standardize numerical features
numerical_cols = list(set(X.columns) - set(categorical_cols))
scaler = StandardScaler()
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

# %%
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# %%
# Define the deep learning model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')  # Binary output: liked or not
])

# %%
# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# %%
# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.2, verbose=1)


# %%
# Evaluate the model
y_pred = (model.predict(X_test) > 0.5).astype("int32")
print(classification_report(y_test, y_pred))

# %%
model.save("user_preference_model.h5")

# %%
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import load_model

# === Load saved model ===
model = load_model("user_preference_model.h5")  # Make sure you save your model first

# === Load encoders and scalers ===
# These should match what you used during training
label_columns = ["genre", "artist", "language", "lyrics_sentiment", "emotion_tag"]
numerical_columns = ["tempo", "energy", "danceability", "acousticness", "instrumentalness", 
                     "valence", "liveness", "speechiness", "loudness"]

# Load new data (user listening history)
df_new = pd.read_csv("dataset/user_history.csv")

# Apply label encoding (same as training)
label_encoders = {}
for col in label_columns:
    le = LabelEncoder()
    df_new[col] = le.fit_transform(df_new[col])
    label_encoders[col] = le

# Standardize numerical columns
scaler = StandardScaler()
df_new[numerical_columns] = scaler.fit_transform(df_new[numerical_columns])

# Predict using the trained model
predictions = model.predict(df_new)
df_new["predicted_like"] = (predictions > 0.5).astype(int)

# Save result
df_new.to_csv("dataset/predicted_user_history.csv", index=False)
print("Prediction saved to predicted_user_history.csv")

# %%
# Re-import necessary library after code execution state reset
import pandas as pd

# Create a sample user_history DataFrame
sample_data = {
    "genre": ["Pop", "Rock", "Jazz"],
    "artist": ["Ed Sheeran", "Imagine Dragons", "Norah Jones"],
    "language": ["English", "English", "English"],
    "tempo": [120, 135, 90],
    "energy": [0.75, 0.85, 0.4],
    "danceability": [0.8, 0.7, 0.3],
    "acousticness": [0.2, 0.1, 0.6],
    "instrumentalness": [0.0, 0.0, 0.0],
    "valence": [0.9, 0.8, 0.5],
    "liveness": [0.1, 0.3, 0.2],
    "speechiness": [0.05, 0.06, 0.04],
    "loudness": [-5.0, -4.5, -7.0],
    "lyrics_sentiment": ["Positive", "Neutral", "Sad"],
    "emotion_tag": ["Happy", "Energetic", "Sad"]
}

user_history_df = pd.DataFrame(sample_data)

# Save to CSV
user_history_csv_path = "dataset/user_history.csv"
user_history_df.to_csv(user_history_csv_path, index=False)

user_history_csv_path

# %%
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Predict probabilities
y_probs = model.predict(X_test).flatten()

# Predict classes
y_pred = (y_probs > 0.5).astype("int32")

# Classification report (already included)
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
auc_score = roc_auc_score(y_test, y_probs)

plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
plt.show()

# Print additional metrics
print(f"AUC Score: {auc_score:.4f}")


# %%



