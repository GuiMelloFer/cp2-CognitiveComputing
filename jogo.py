import mediapipe
import cv2
import numpy as np
import uuid
import os
#import funcao

cv2.namedWindow("preview")

vc = cv2.VideoCapture('pedra-papel-tesoura.mp4') 

def processa_img(img):

    
    
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