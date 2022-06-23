import cv2
import numpy as np
import mediapipe as mp

vid = cv2.VideoCapture(0)

pose = mp.solutions.pose.Pose()
hands = mp.solutions.hands.Hands()

draw_fn = mp.solutions.drawing_utils

while vid.isOpened():
    
    ret , img = vid.read()
    
    if ret == False :
        print('Can not continue the video or webcam.')
        break
    
    results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    if results.pose_landmarks:
            
        draw_fn.draw_landmarks(img, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
    
    results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    if results.multi_hand_landmarks:
        
        for landmarks in results.multi_hand_landmarks:
            
            draw_fn.draw_landmarks(img, landmarks, mp.solutions.hands.HAND_CONNECTIONS)
    
    
    cv2.imshow('Pose and Hand Tracking',img)
    
    if cv2.waitKey(1) in [27,13,ord('q')]:
        break
        
vid.release()
cv2.destroyAllWindows()
