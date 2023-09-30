import cv2
import numpy as np
import tensorflow as tf

# Load the pre-trained emotion detection model
model = tf.keras.models.load_model("C:\\Users\\Mohan\\Pictures\\1.jpg.h5")

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define a list of emotions
emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# Initialize the webcam (you can also use a video file)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face_roi = gray[y:y + h, x:x + w]
        resized_face = cv2.resize(face_roi, (48, 48))
        resized_face = resized_face / 255.0  # Normalize the image

        # Predict the emotion
        emotion_pred = model.predict(np.expand_dims(np.expand_dims(resized_face, -1), axis=0))
        emotion_label = emotions[np.argmax(emotion_pred)]

        # Draw a rectangle and label the emotion
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame with emotions
    cv2.imshow('Facial Emotion Detection', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the window
cap.release()
cv2.destroyAllWindows()
