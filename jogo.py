import cv2
#import mediapipe
import time
import numpy as np

# templates
paper = cv2.imread('paper-template.png',0)
scissors = cv2.imread('scissors-template.png',0)
rock = cv2.imread('rock-template.png',0)

#pontuação
pontosMao1 = 0
pontosMao2 = 0

#globals
anterior1 = ""
anterior2 = ""
vencedor = ""                 


def processaVideo(cap):
    success, img=cap.read() #pega um frame da imagem
    
    
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Definição dos valores minimo e max da mascara
    image_lower_hsv = np.array([0, 18, 10])  
    image_upper_hsv = np.array([180, 255, 255])

    #Cria mascara com filtros
    mask_hsv = cv2.inRange(img_hsv, image_lower_hsv, image_upper_hsv)

    #acha os contornos
    contornos, _ = cv2.findContours(mask_hsv, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) 
    
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

    #Recebe os moments para descobrir o X e o Y das duas maiores areas        
    M1 = cv2.moments(aux)
    M2 = cv2.moments(aux2)

    cx1 = int(M1['m10']/M1['m00'])
    cy1 = int(M1['m01']/M1['m00'])

    cx2 = int(M2['m10']/M2['m00'])
    cy2 = int(M2['m01']/M2['m00'])

   
   #Associa a mao 1 para mão mais proxima do eixo x
    if (cx1 < cx2):
        mao1 = verificaMao(aux) 
        mao2 = verificaMao(aux2) 
    else:
        mao1 = verificaMao(aux2) 
        mao2 = verificaMao(aux)  
    
    

    validaRodada(mao1, mao2, img)

    #Desenha na tela 
    text1 = pontosMao1
    text2 = pontosMao2
    text3 = vencedor
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, str(text1), (700,50), font,1,(0,0,0),2,cv2.LINE_AA)
    cv2.putText(img, "X", (750,50), font,1,(0,0,0),2,cv2.LINE_AA)
    cv2.putText(img, str(text2), (800,50), font,1,(0,0,0),2,cv2.LINE_AA)
    cv2.putText(img, str(text3), (600,100), font,1,(0,0,0),2,cv2.LINE_AA)


    cv2.imshow("hand gestures",img)


#Retorna jogada baseada na area do contorno
def verificaMao(mao):

    area = cv2.contourArea(mao)

    if(area > 63000 and area < 70000):
        return "papel"
    elif area < 51000 and area > 48000:
         return "tesoura"
    else:
        return "pedra"    
    

def validaRodada(mao1, mao2, img):
    font = cv2.FONT_HERSHEY_SIMPLEX
    text1 = 'JOGADOR 1: ', (mao1)
    text2 = 'JOGADOR 2: ', (mao2)

    global anterior1, anterior2, pontosMao1, pontosMao2, vencedor

    cv2.putText(img, str(text1), (10,50), font,1,(0,0,0),2,cv2.LINE_AA)
    cv2.putText(img, str(text2), (10,150), font,1,(0,0,0),2,cv2.LINE_AA)

    #Caso tenha alteração na jogada.. valida quem ganha e soma os pontos
    if mao1 != anterior1 or mao2 != anterior2:
        if(mao1 == "tesoura" and mao2 == "papel"):
            pontosMao1 += 1
            vencedor = "Jogador 1 Venceu"
        elif(mao1 == "tesoura" and mao2 == "pedra"):  
            pontosMao2 += 1
            vencedor = "Jogador 2 Venceu"
        elif(mao1 == "papel" and mao2 == "tesoura"):  
            pontosMao2 += 1
            vencedor = "Jogador 2 Venceu"
        elif(mao1 == "papel" and mao2 == "pedra"):  
            pontosMao1 += 1
            vencedor = "Jogador 1 Venceu"
        elif(mao1 == "pedra" and mao2 == "papel"):  
            pontosMao2 += 1
            vencedor = "Jogador 2 Venceu"
        elif(mao1 == "pedra" and mao2 == "tesoura"):  
            pontosMao1 += 1   
            vencedor = "Jogador 1 Venceu"
        else:
            vencedor = "Empate"  
        
        anterior1 = mao1
        anterior2 = mao2   

ctime=0
ptime=0

cap=cv2.VideoCapture("pedra-papel-tesoura.mp4")

while True:

    processaVideo(cap)
   
    
    #press q to quit
    if cv2.waitKey(1) == ord('q'):
        break


cap.release()   
cv2.destroyAllWindows()

