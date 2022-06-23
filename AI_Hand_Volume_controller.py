import cv2
import mediapipe as mp
import numpy as np
from sklearn.metrics import euclidean_distances 
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume


vid = cv2.VideoCapture(0)
vid.set(3,580)
vid.set(4,360)

hands = mp.solutions.hands.Hands()

draw_fn = mp.solutions.drawing_utils

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

distance = 0

while vid.isOpened():
    
    ret , img = vid.read()
    
    if ret == False :
        print('Can not continue the video or webcam.')
        break
    
    results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    if results.multi_hand_landmarks:
        
        for landmarks in results.multi_hand_landmarks:
            
            for i, lm in enumerate(landmarks.landmark):
                
                if i == 4:
                    x4, y4 = int(lm.x * img.shape[1]), int(lm.y * img.shape[0])  
                
                if i == 8:
                    x8, y8 = int(lm.x * img.shape[1]), int(lm.y * img.shape[0])  
                    
            cv2.circle(img,(x4,y4),15,(200,100,150),2)
            cv2.circle(img,(x8,y8),15,(200,100,150),2)
            cv2.line(img,(x4,y4),(x8,y8),(150,100,200),2)
            
            #distance = euclidean_distances([(x4,y4),(x8,y8)])[0,1]
            
            if abs(x8 - x4) < 50 :
                distance = abs(y8 - y4)
            
            minvol, maxvol, _ = volume.GetVolumeRange()
            
            newvol = np.interp(distance, (20, 120),(minvol, maxvol))
            barpos = np.interp(distance, (20, 120),(300,150))
            percent = int(np.interp(distance, (20, 120),(0,100)))
            
            cv2.rectangle(img,(20,150),(40,300),(0,255,0),1)
            cv2.rectangle(img,(20,int(barpos)),(40,300),(0,255,0),-1)
            cv2.putText(img,f'{str(percent)} %',(20,330),1,1,(0,0,255),2)
            
            volume.SetMasterVolumeLevel(newvol, None)
            
            draw_fn.draw_landmarks(img, landmarks, mp.solutions.hands.HAND_CONNECTIONS)
    
    cv2.imshow('Hand Tracking',img)
    
    if cv2.waitKey(1) in [27,13,ord('q')]:
        break
        
vid.release()
cv2.destroyAllWindows()
