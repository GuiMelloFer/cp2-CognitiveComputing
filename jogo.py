import cv2
import mediapipe
import time

ctime=0
ptime=0

cap=cv2.VideoCapture("pedra-papel-tesoura.mp4")

medhands=mediapipe.solutions.hands
hands=medhands.Hands(max_num_hands=1,min_detection_confidence=0.7)
draw=mediapipe.solutions.drawing_utils

while True:
    success, img=cap.read() #pega um frame da imagem
    
    img = cv2.flip(img,1) # inverte a imagem
    
    imgrgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
    
    
    #realiza a detecção da mão na imagem
    res = hands.process(imgrgb)
    
    lmlist=[]
    tipids=[4,8,12,16,20] # lista com as pontas dos dedos
    
    #desenha no canto da tela um retangulo, os numeros vão aparecer aqui
    cv2.rectangle(img,(20,350),(90,440),(0,255,204),cv2.FILLED)
    cv2.rectangle(img,(20,350),(90,440),(0,0,0),5)
    
    ## se detectar alguma mão entra no if 
    if res.multi_hand_landmarks:
        for handlms in res.multi_hand_landmarks:
            for id,lm in enumerate(handlms.landmark):
                
                h,w,c= img.shape
                cx,cy=int(lm.x * w) , int(lm.y * h)
                lmlist.append([id,cx,cy])
                if len(lmlist) != 0 and len(lmlist)==21:
                    fingerlist=[]
                    
                    #thumb and dealing with flipping of hands
                    if lmlist[12][1] > lmlist[20][1]:
                        if lmlist[tipids[0]][1] > lmlist[tipids[0]-1][1]:
                            fingerlist.append(1)
                        else:
                            fingerlist.append(0)
                    else:
                        if lmlist[tipids[0]][1] < lmlist[tipids[0]-1][1]:
                            fingerlist.append(1)
                        else:
                            fingerlist.append(0)
                    
                    #others
                    for id in range (1,5):
                        if lmlist[tipids[id]][2] < lmlist[tipids[id]-2][2]:
                            fingerlist.append(1)
                        else:
                            fingerlist.append(0)
                    
                    
                    if len(fingerlist)!=0:  # se a lista for diferente de zero então 
                        fingercount=fingerlist.count(1) # conta quantidade de dedos
                    
                    # escreve na tela a quantidade detectada
                    cv2.putText(img,str(fingercount),(25,430),cv2.FONT_HERSHEY_PLAIN,6,(0,0,0),5)
                    
                #change color of points and lines
                draw.draw_landmarks(img,handlms,medhands.HAND_CONNECTIONS,draw.DrawingSpec(color=(0,255,204),thickness=2,circle_radius=2),draw.DrawingSpec(color=(0,0,0),thickness=2,circle_radius=3))
                
    cv2.imshow("hand gestures",img)
    
    #press q to quit
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()   
cv2.destroyAllWindows()