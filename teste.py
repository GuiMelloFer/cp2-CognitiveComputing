import cv2
import mediapipe
import time
import numpy as np

# templates
paper = cv2.imread('paper-template.png',0)
scissors = cv2.imread('scissors-template.png',0)
rock = cv2.imread('rock-template.png',0)
                 


def processaVideo(cap):
    success, img=cap.read() #pega um frame da imagem
    
    #img = cv2.flip(img,1) # inverte a imagem
    
    imgrgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Definição dos valores minimo e max da mascara
    # o magenta tem h=300 mais ou menos ou 150 para a OpenCV
    image_lower_hsv = np.array([0, 18, 10])  
    image_upper_hsv = np.array([180, 255, 255])


    mask_hsv = cv2.inRange(img_hsv, image_lower_hsv, image_upper_hsv)

    contornos, _ = cv2.findContours(mask_hsv, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) 
    mask_rgb = cv2.cvtColor(mask_hsv, cv2.COLOR_GRAY2RGB) 
    contornos_img = mask_rgb.copy() # Cópia da máscara para ser desenhada "por cima"

    #cv2.drawContours(contornos_img, contornos, -1, [255, 0, 0], 5);
    
    ##busca das 2 maiores areas
    max_area = -1
    second_max_area = -1

    for i in range(len(contornos)):
        area = cv2.contourArea(contornos[i])
        if area>max_area:
            aux = contornos[i]
            max_area = area


    for i in range(len(contornos)):
        area2 = cv2.contourArea(contornos[i])    
        if area2>second_max_area and area2<max_area:
            aux2 = contornos[i]
            second_max_area = area2

    cv2.drawContours(contornos_img, aux, -1, [255, 0, 0], 5);
    cv2.drawContours(contornos_img, aux2, -1, [0, 255, 0], 5);

    M1 = cv2.moments(aux)
    M2 = cv2.moments(aux2)
    #print( M )

    cx1 = int(M1['m10']/M1['m00'])
    cy1 = int(M1['m01']/M1['m00'])

    cx2 = int(M2['m10']/M2['m00'])
    cy2 = int(M2['m01']/M2['m00'])

   
    # inicializa com o construtor ORB
    orb = cv2.ORB_create(nfeatures=100)
    # cria o objeto bf (Brute-force descriptor matcher)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

   
    if (cx1 < cx2):
        mao1 = calculaMatches(aux, bf, orb, img) 
        mao2 = calculaMatches(aux2, bf, orb, img) 
    else:
        mao1 = calculaMatches(aux2, bf, orb, img) 
        mao2 = calculaMatches(aux, bf, orb, img)  
    
    

#     #print(mao1, mao2) 

    cv2.imshow("hand gestures",contornos_img)



def calculaMatches(mao, bf, orb, img):

    M = cv2.moments(mao)
    
    cx1 = int(M['m10']/M['m00'])
    cy1 = int(M['m01']/M['m00'])
    print(cx1, cy1 )
    print("---------------")
    # Cropping an image
    cropped_image = img[cy1 - 200:cy1 + 200, cx1 - 500:cy1+500]

    
    


    #Podemos usar uma função que calcula os keypoints e Descritores
    kp1, des1 = orb.detectAndCompute(paper,None)
    kp2, des2 = orb.detectAndCompute(scissors,None)
    kp3, des3 = orb.detectAndCompute(rock,None)
    kpFrame, desFrame = orb.detectAndCompute(cropped_image,None)


    # a função match devolve os matches encontrados
    matches_paper = bf.match(des1, desFrame)
    matches_scissors = bf.match(des2, desFrame)
    matches_rock = bf.match(des3, desFrame)


    if(len(matches_paper) > len(matches_scissors) and len(matches_paper) > len(matches_rock)):
        return "papel"
    elif(len(matches_scissors) >len(matches_paper) and len(matches_scissors) > len(matches_rock)):
         return "tesoura"
    else:
        return "pedra"    
    



ctime=0
ptime=0

cap=cv2.VideoCapture("pedra-papel-tesoura.mp4")

medhands=mediapipe.solutions.hands
hands=medhands.Hands(max_num_hands=1,min_detection_confidence=0.7)
draw=mediapipe.solutions.drawing_utils

while True:

    processaVideo(cap)
   
    #realiza a detecção da mão na imagem
    # res = hands.process(imgrgb)
    
    # lmlist=[]
    # tipids=[4,8,12,16,20] # lista com as pontas dos dedos
    
    #desenha no canto da tela um retangulo, os numeros vão aparecer aqui
    # cv2.rectangle(img,(20,350),(90,440),(0,255,204),cv2.FILLED)
    # cv2.rectangle(img,(20,350),(90,440),(0,0,0),5)
    
    ## se detectar alguma mão entra no if 
    # if res.multi_hand_landmarks:
    #     for handlms in res.multi_hand_landmarks:
    #         for id,lm in enumerate(handlms.landmark):
                
    #             h,w,c= img.shape
    #             cx,cy=int(lm.x * w) , int(lm.y * h)
    #             lmlist.append([id,cx,cy])
    #             if len(lmlist) != 0 and len(lmlist)==21:
    #                 fingerlist=[]
                    
    #                 #thumb and dealing with flipping of hands
    #                 if lmlist[12][1] > lmlist[20][1]:
    #                     if lmlist[tipids[0]][1] > lmlist[tipids[0]-1][1]:
    #                         fingerlist.append(1)
    #                     else:
    #                         fingerlist.append(0)
    #                 else:
    #                     if lmlist[tipids[0]][1] < lmlist[tipids[0]-1][1]:
    #                         fingerlist.append(1)
    #                     else:
    #                         fingerlist.append(0)
                    
    #                 #others
    #                 for id in range (1,5):
    #                     if lmlist[tipids[id]][2] < lmlist[tipids[id]-2][2]:
    #                         fingerlist.append(1)
    #                     else:
    #                         fingerlist.append(0)
                    
                    
    #                 if len(fingerlist)!=0:  # se a lista for diferente de zero então 
    #                     fingercount=fingerlist.count(1) # conta quantidade de dedos
                    
    #                 # escreve na tela a quantidade detectada
    #                 cv2.putText(img,str(fingercount),(25,430),cv2.FONT_HERSHEY_PLAIN,6,(0,0,0),5)
                    
    #             #change color of points and lines
    #             draw.draw_landmarks(img,handlms,medhands.HAND_CONNECTIONS,draw.DrawingSpec(color=(0,255,204),thickness=2,circle_radius=2),draw.DrawingSpec(color=(0,0,0),thickness=2,circle_radius=3))
          
    
    
    #press q to quit
    if cv2.waitKey(1) == ord('q'):
        break


cap.release()   
cv2.destroyAllWindows()

