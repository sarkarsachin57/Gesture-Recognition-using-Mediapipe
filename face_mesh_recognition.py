import cv2
import numpy as np
import mediapipe as mp


vid = cv2.VideoCapture(0)

face_mesh = mp.solutions.face_mesh.FaceMesh()
draw_fn = mp.solutions.drawing_utils

while vid.isOpened():
    
    ret , img = vid.read()
    
    if ret == False :
        print('Can not continue the video or webcam.')
        break
    
    results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    
    if results.multi_face_landmarks:
        
        for landmarks in results.multi_face_landmarks:
            
            draw_fn.draw_landmarks(img, landmarks, mp.solutions.face_mesh_connections.FACEMESH_TESSELATION)
    
    
    cv2.imshow('Face Mesh Recognition',img)
    
    if cv2.waitKey(1) in [27,13,ord('q')]:
        break
        
vid.release()
cv2.destroyAllWindows()
