import cv2 as cv
import numpy as np
from collections import deque
import mediapipe as mp


mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap =cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

with mp_hands.Hands(min_detection_confidence=0.7,min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret,frame=cap.read()
    
    ####
        image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        
        print(results.multi_hand_landmarks)
        if results.multi_hand_landmarks:
            for num,hand in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(frame,hand,mp_hands.HAND_CONNECTIONS)
            
    ####
    
    
    
    
    
    
    
        cv.imshow("frame",frame)
        if cv.waitKey(10) & 0xFF == ord('q'):
            break
cap.release()