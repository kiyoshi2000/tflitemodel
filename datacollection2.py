import cv2
import mediapipe as mp
import numpy as np
import math
import time

# Captures from webcam
cap = cv2.VideoCapture(0)

# Initialize Mediapipe Hand module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Size of image that will be the input of our neural network
img_size = 244

# Offset for cropping
offset = 20

# Private folder to save our dataset images
folder = "Data/C"

# Image counter
count = 0

while True:
    success, img = cap.read()

    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process image with Mediapipe Hand module
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        # Take the first detected hand
        hand_landmarks = results.multi_hand_landmarks[0].landmark

        # Extract bounding box coordinates
        x, y, w, h = (
            min(hand_landmarks, key=lambda point: point.x).x * img.shape[1],
            min(hand_landmarks, key=lambda point: point.y).y * img.shape[0],
            (max(hand_landmarks, key=lambda point: point.x).x - min(hand_landmarks, key=lambda point: point.x).x) * img.shape[1],
            (max(hand_landmarks, key=lambda point: point.y).y - min(hand_landmarks, key=lambda point: point.y).y) * img.shape[0]
        )

        # White box that will fit the hand
        img_white = np.ones((img_size, img_size, 3), np.uint8) * 255

        # Cropped image that will be automatically adjusted to fit inside the given limits
        img_crop = img[int(y) - offset:int(y) + int(h) + offset, int(x) - offset:int(x) + int(w) + offset]

        # Will return the height and the width
        img_crop_shape = img_crop.shape

        # Ratio
        aspect_ratio = h / w

        if aspect_ratio > 1:
            k = img_size / h
            w_cal = math.ceil(k * w)
            img_resize = cv2.resize(img_crop, (w_cal, img_size))
            w_gap = math.ceil((img_size - w_cal) / 2)
            img_white[:, w_gap:w_cal + w_gap] = img_resize
        else:
            k = img_size / w
            h_cal = math.ceil(k * h)
            img_resize = cv2.resize(img_crop, (img_size, h_cal))
            h_gap = math.ceil((img_size - h_cal) / 2)
            img_white[h_gap:h_cal + h_gap, :] = img_resize

        cv2.imshow("ImageCrop", img_crop)
        cv2.imshow("ImageWhite", img_white)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)

    if key == ord("s"):
        count += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', img_white)
        print(count)

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
