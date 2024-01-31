import cv2
import numpy as np
import math
import time
import mediapipe as mp

# Captures from webcam
cap = cv2.VideoCapture(0)

# Instance the hand detector from mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Size of image that will be the input of our neural network
imgSize = 244

offset = 20

# Private folder to save our dataset images
folder = "Dataset/A"

# Image counter
count = 0

while True:
    success, img = cap.read()
    img_h = img.copy()
    imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255  # Initialize imgWhite outside the loop

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = [(int(lm.x * img.shape[1]), int(lm.y * img.shape[0])) for lm in hand_landmarks.landmark]

            # The hand that we are going to track will be the first one that we found
            x, y, w, h = cv2.boundingRect(np.array(landmarks))
            imgCrop = img_h[y - offset:y + h + offset, x - offset:x + w + offset]

            # Calculate aspect ratio
            aspect_ratio = h / w

            if aspect_ratio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize

            # Display images
            cv2.imshow("ImageCrop", imgCrop)
    
    cv2.imshow("imageWhite", imgWhite)
    cv2.imshow("Image", img_h)

    key = cv2.waitKey(1)

    if key == ord("s"):
        count += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(count)
