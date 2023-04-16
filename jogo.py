import cv2 
import numpy as np
import math
#import funcao

cv2.namedWindow("preview")

vc = cv2.VideoCapture('pedra-papel-tesoura.mp4') 

def processa_img(img):

    # carrega as imagens.
    template = cv2.imread('paper-template.png', 0)  ##template

   
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Aplica template Matching
    res = cv2.matchTemplate(img_gray,template,cv2.TM_SQDIFF)

    # res é uma imagem e podemos plotar seu shape e imagem

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    # Resolução
    #Extrai o shape (dimensão da imagem)
    largura, altura = template.shape[::-1]
    
    # ajusta o bounbox da imagem lembra que é uma tupla ()
    bottom_right = (min_loc[0] + largura, min_loc[1] + altura)
    
    # desenha o retangulo na imagem original
    cv2.rectangle(img,min_loc, bottom_right, (127,255,255), 4)



    return img

if vc.isOpened():
    rval, frame = vc.read()
else:
    rval = False

while rval:
    
   # cv2.imshow("preview", frame)
    
    res_processa = processa_img(frame)
    
    cv2.imshow("result", res_processa)
    rval, frame = vc.read()

    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break

cv2.destroyAllWindows()
vc.release()