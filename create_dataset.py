import os
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt
import pickle

# Initialize MediaPipe hands module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize MediaPipe Hands object
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Directory containing the dataset
DATA_DIR = './data'

# Lists to store data and corresponding labels
data = []
labels = []

# Loop through each class directory
for dir_ in os.listdir(DATA_DIR):
    # Loop through each image in the class directory
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        # List to store hand landmarks for the current image
        data_aux = []
        
        # Read the image
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process the image to detect hand landmarks
        results = hands.process(img_rgb)
        
        # Check if hand landmarks are detected
        if results.multi_hand_landmarks:
            # Iterate through detected hands
            for hand_landmarks in results.multi_hand_landmarks:
                # Iterate through landmarks of each hand
                for i in range(len(hand_landmarks.landmark)):
                    # Extract x and y coordinates of each landmark
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x)
                    data_aux.append(y)

            # Append hand landmarks data and corresponding label to lists
            data.append(data_aux)
            labels.append(dir_)

# Save the data and labels to a pickle file
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)
