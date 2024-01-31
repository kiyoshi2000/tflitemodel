import cv2
from cvzone.HandTrackingModule import HandDetector

import numpy as np
import math
import time

#Captures from webcam
cap = cv2.VideoCapture(0)

#Instance the handdector from czone 
detector = HandDetector(maxHands=1)

#Size of image that will be the input of our neural network
imgSize =244


offset =20

#Private folDer to save our dataset images
folder = "Data/V"

# image counter 
count =0

while True:
        
    success , img = cap.read()
    hands, img = detector.findHands(img)

    img_h=img
    
    #img_h = cv2.flip(img, 1)


    if hands:

        #The hand that we are going to track will be the first one that we found
        hand = hands[0]
        #Dimensions between the bounded box that fits the tracked hand
        x,y,w,h = hand['bbox']

        #white box that will fits the hand
        imgWhite = np.ones((imgSize,imgSize,3), np.uint8)*255

        #Cropped image that will be automatically adjust to fits inside the given limits
        imgCrop = img_h[y - offset:y + h + offset, x - offset:x + w +offset]
        
        #Will return the heigth and the weigth
        imgCropShape = imgCrop.shape
        
        #Ratio 
        aspecRatio =h/w

        if aspecRatio > 1:

            k = imgSize/h
            wCal = math.ceil(k * w)
            
            imgRezise = cv2.resize(imgCrop, (wCal, imgSize))
            imgReziseShape = imgRezise.shape

            wGap = math.ceil((imgSize-wCal)/2)
        
            imgWhite[:, wGap:wCal+wGap] = imgRezise

        else:

            k = imgSize/w
            hCal = math.ceil(k * h)
            
            imgRezise = cv2.resize(imgCrop, (imgSize, hCal))
            imgReziseShape = imgRezise.shape

            hGap = math.ceil((imgSize-hCal)/2)
        
            imgWhite[hGap:hCal + hGap,:] = imgRezise


        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("imageWhite", imgWhite)
    
    cv2.imshow("Image", img_h)
    key = cv2.waitKey(1)

    if key == ord("s"):
        count += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(count)