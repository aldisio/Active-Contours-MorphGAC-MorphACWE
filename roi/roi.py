import numpy as np
import matplotlib.pyplot as plt
import cv2
 
countImg=1 
img = cv2.imread('../testimages/CasoAS.bmp',0) 
img_blur = cv2.GaussianBlur(img,(5,5),0)
# ------------------------------
# Binarizacao da imagem
# ------------------------------
ret, img_bin = cv2.threshold(img_blur,0,255,cv2.THRESH_BINARY)

# ------------------------------
# Identifica a maior componente
# ------------------------------
connectivity = 4  
# Perform the operation
output = cv2.connectedComponentsWithStats(img_bin, connectivity, cv2.CV_8U)

# Rotulos das componentes identificadas
num_labels = output[0]
# Matriz com os pixels identificados com os rotulos
labels = output[1]

# Matriz Estatisticas calculadas para cada componente conexa detectada
# stats[label, COLUMN]
# cv2.CC_STAT_LEFT The leftmost (x) coordinate which is the inclusive start of the bounding box in the horizontal direction.
# cv2.CC_STAT_TOP The topmost (y) coordinate which is the inclusive start of the bounding box in the vertical direction.
# cv2.CC_STAT_WIDTH The horizontal size of the bounding box
# cv2.CC_STAT_HEIGHT The vertical size of the bounding box
# cv2.CC_STAT_AREA The total area (in pixels) of the connected component
stats = output[2]
# Matrix com os centroides das componentes
centroids = output[3]

for i in np.arange(0, np.size(stats, 0)):
    print(str(i) +": "+str(stats[i, cv2.CC_STAT_AREA]))

print(stats.shape)

# ------------------------------
# Detecta e separa a maior componente
# ------------------------------
img_max = np.zeros_like(labels) 
idx_maxcomp = 1+stats[1:, cv2.CC_STAT_AREA].argmax()

img_max[labels != idx_maxcomp] = 0
img_max[labels == idx_maxcomp] = 255

# ------------------------------
# Desenha circulo - ROI
# ------------------------------
roi_cx = int(centroids[idx_maxcomp,0])
roi_cy = int(centroids[idx_maxcomp,1])
img_roi = np.zeros_like(img)
cv2.circle(img_roi, (roi_cx, roi_cy), 120, (1,0,0), -1)
img_roicrop = img_roi*img
# ------------------------------
# Imprime imagens
# ------------------------------
numPrintImgs = 3
plt.subplot(1,numPrintImgs,countImg), plt.imshow(img,'gray'); countImg+=1
plt.subplot(1,numPrintImgs,countImg), plt.imshow(img_bin,'gray'); countImg+=1
plt.subplot(1,numPrintImgs,countImg), plt.imshow(img_roicrop,'gray'); countImg+=1

plt.show()