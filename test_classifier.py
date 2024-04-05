import cv2
import pickle
import mediapipe as mp
import numpy as np

#Load the trained model from the pickle file
model_dict=pickle.load(open('./RF_model.p','rb'))
model=model_dict['RF_model']

#Open the camera for real-time
cam=cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands=mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Dictionary mapping numeric labels to corresponding characters
labels_dict = {1:'A',2:'B',3:'C',7:'Thank you',4:'Hi',8:'Yes',6:'No',5:'Love You'}

while True:
    data_aux=[]
    x_dict,y_dict=[],[]
    # Read frame from the webcam
    ret,frame=cam.read()
    Horizontal, Width, Height = frame.shape

    #Converting it to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    #Drawing landmarks and connections if hand is detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
             mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style() ) 
        
        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x=hand_landmarks.landmark[i].x
                y=hand_landmarks.landmark[i].y
                data_aux.append(x)
                data_aux.append(y)
                x_dict.append(x)
                y_dict.append(y)
        
        x1=int(min(x_dict)*Width)-10
        y1=int(min(y_dict)*Horizontal)-10
        x2=int(max(x_dict)*Width)-10
        y2=int(max(y_dict)*Horizontal)-10

        #Make prediction using model
        prediction = model.predict([np.asarray(data_aux)])
        predicted_character = labels_dict[int(prediction[0])]
        
        #Drawing the box and predicted text
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,0),4)
        cv2.putText(frame, predicted_character, (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                        cv2.LINE_AA)
    
    cv2.imshow('frame',frame)
    cv2.waitKey(1)

cam.release()
cv2.destroyAllWindows()