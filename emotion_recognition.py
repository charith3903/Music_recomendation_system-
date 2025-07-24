import cv2
from deepface import DeepFace
import numpy as np

# rest of your code ...

# Emotion categories you're interested in
TARGET_EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad']

# Load webcam
cap = cv2.VideoCapture(0)

# Set frame width/height (optional)
cap.set(3, 640)
cap.set(4, 480)

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Detect faces and emotions
    try:
        results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

        if isinstance(results, list):
            for res in results:
                dominant_emotion = res['dominant_emotion']
                if dominant_emotion in TARGET_EMOTIONS:
                    x, y, w, h = res['region']['x'], res['region']['y'], res['region']['w'], res['region']['h']
                    # Draw rectangle and emotion
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, dominant_emotion, (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            res = results
            dominant_emotion = res['dominant_emotion']
            if dominant_emotion in TARGET_EMOTIONS:
                x, y, w, h = res['region']['x'], res['region']['y'], res['region']['w'], res['region']['h']
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, dominant_emotion, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    except Exception as e:
        print(f"Error: {str(e)}")

    cv2.imshow("Emotion Recognition", frame)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
