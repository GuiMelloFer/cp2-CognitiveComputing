
import cv2
from matplotlib import pyplot as plt
import numpy as np

# carrega as imagens
img1 = cv2.imread('paper-template.png',0)
img2 = cv2.imread('game.png',0)

# inicializa com o construtor ORB
orb = cv2.ORB_create(nfeatures=100)

#Podemos usar uma função que calcula os keypoints e Descritores
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)


gray1 = cv2.drawKeypoints(img1, kp1, outImage=np.array([]), flags=0)
gray2 = cv2.drawKeypoints(img2, kp2, outImage=np.array([]), flags=0)

# cria o objeto bf (Brute-force descriptor matcher)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# a função match devolve os matches encontrados
matches = bf.match(des1,des2)

print("Foram encontrados: {} matches".format(len(matches)))

img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches,None, flags=2)


plt.figure(figsize = (20,10))
plt.imshow(img3); plt.show();

#template
# plt.figure(figsize = (10,10))
# plt.imshow(gray1); plt.show();

# # Imagem espaço de busca
# plt.figure(figsize = (10,10))
# plt.imshow(gray2); plt.show();