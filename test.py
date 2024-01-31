import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

# Classifier

#Add to your folder
classifier = Classifier("Model_4/keras_model.h5", "Model_4/labels.txt")

offset = 20
imgSize = 300

# Mapeamento de índices para letras
index_to_letter = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G", 7: "H", 8: "I", 9: "K", 10: "L", 11: "O", 12: "Y", 13: "W", 14: "V"}



# Lista para armazenar as letras detectadas
detected_letters_list = []
start_time = time.time()  # Inicializa start_time com o tempo atual

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)

    img_h = img
    imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255  # Inicializa com branco

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgCrop = img_h[y - offset:y + h + offset, x - offset:x + w + offset]

        imgCropShape = imgCrop.shape
        aspecRatio = h / w

        if aspecRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgRezise = cv2.resize(imgCrop, (wCal, imgSize))
            imgWhite[:, math.ceil((imgSize - wCal) / 2):wCal + math.ceil((imgSize - wCal) / 2)] = imgRezise
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgRezise = cv2.resize(imgCrop, (imgSize, hCal))
            imgWhite[math.ceil((imgSize - hCal) / 2):hCal + math.ceil((imgSize - hCal) / 2), :] = imgRezise

        # Obter a previsão da letra
        prediction, index = classifier.getPrediction(imgWhite)

        # Se a letra A, B ou C for detectada e ainda não tiver aparecido, adicione-a à lista
        if index in index_to_letter:
            letter = index_to_letter[index]
            if start_time is None or time.time() - start_time > 2:
                detected_letters_list.append(letter)
                start_time = time.time()

    # Posiciona o texto no canto esquerdo da imagem e usa a cor branca
    text_position = (10, 30)
    cv2.putText(img_h, "Sentence: " + "".join(detected_letters_list), text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    cv2.imshow("imageWhite", imgWhite)
    cv2.imshow("Image", img_h)
    cv2.waitKey(1)
