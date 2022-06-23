import cv2, autopy, time
import numpy as np
import mediapipe as mp
from sklearn.metrics import euclidean_distances

hand = mp.solutions.hands.Hands()
draw = mp.solutions.drawing_utils.draw_landmarks

video = cv2.VideoCapture(0) # returns frame resolution of (640, 480)

frames_after_clicked = -50

while video.isOpened():
    
    ret , img = video.read()
    img = cv2.flip(img,1)
    
    if not ret :
        print('Cant continue video or webcam.')
        break
        
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    results = hand.process(rgb_img)
    
    #x8, x12 = False, False
    distance = 51
    
    if results.multi_hand_landmarks:
        
        for landmarks in results.multi_hand_landmarks:
            
            for i,lm in enumerate(landmarks.landmark):
                
                if i == 4:
                    x4, y4 = int(lm.x * img.shape[1]), int(lm.y * img.shape[0])
                if i == 7:
                    x7, y7 = int(lm.x * img.shape[1]), int(lm.y * img.shape[0]) 
                if i == 8:
                    x8, y8 = int(lm.x * img.shape[1]), int(lm.y * img.shape[0]) 
                if i == 12:
                    x12, y12 = int(lm.x * img.shape[1]), int(lm.y * img.shape[0]) 
                    
                    distance = euclidean_distances([(x8,y8),(x12,y12)])[0,1]
                
                if distance < 30 and y8 < y7:
                    
                    cv2.circle(img, (x8,y8), 15, (200,100,250),-1)
                    
                    mouse_x = np.interp(x8, (0+75,640-75),(0,autopy.screen.size()[0]-1))
                    mouse_y = np.interp(y8, (0+75,480-75),(0,autopy.screen.size()[1]-1))
                    
                    autopy.mouse.move(mouse_x,mouse_y)
                    
                if euclidean_distances([(x4,y4),(x8,y8)])[0,1] < 30:
                    cv2.circle(img, (x8,y8), 15, (100,250,200),-1)
                    if frames_after_clicked > 20:
                        autopy.mouse.click()
                        frames_after_clicked = 0
            
            draw(img, landmarks, mp.solutions.hands.HAND_CONNECTIONS)
    
    cv2.putText(img,'('+str(int(autopy.mouse.location()[0]))+','+str(int(autopy.mouse.location()[1]))+')',(20,20),1,1,(0,0,255),2)
    cv2.imshow('AI Hand Mouse Controller',img)
    
    frames_after_clicked += 1
    
    if cv2.waitKey(1) in [13, 27, ord('q')]:
        break
        
video.release()
cv2.destroyAllWindows()
