from imageStitching import *
import cv2


image1 = cv2.imread('vol1.jpeg')
image2 = cv2.imread('vol2.jpeg')
image3 = cv2.imread('vol3.jpeg')
image4 = cv2.imread('vol4.jpeg')

images = [image1, image2, image3, image4]
panorama = images[0]
for i in range(len(images) - 1):
    panorama = stitchImage(panorama, images[i+1])

cv2.imwrite('Panorama.jpg', panorama)