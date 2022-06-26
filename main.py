import cv2
import numpy as np
import os
from cvzone.HandTrackingModule import HandDetector
# import mediapipe as mp

#variables
width, height = 1280, 720
folderpath = "presentation"
imgNumber = 0
hs, ws = int(120 * 1), int(213 * 1)
gestureThreshold = 300
buttonPressed = False
buttonCounter = 0
buttonDelay = 20

#camera setup
cap =   cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

#list of presentatin images
pathimages = sorted(os.listdir(folderpath), key = len)
print(pathimages)


#Hand Detector
detector = HandDetector(detectionCon=0.8, maxHands=1) # consider it as a hand only if you are 80 percent confident. and mx number of hands are 1

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1) # 1 - for horizontal direction
    #import Images
    pathFullImage = os.path.join(folderpath, pathimages[imgNumber])
    imgCurrent = cv2.imread(pathFullImage)

    # Adding webcam image on the slides

    h, w, _ = imgCurrent.shape

    hands, img = detector.findHands(img) # left, right -- , flipType=False
    cv2.line(img, (0, gestureThreshold), (width,gestureThreshold), (0, 255, 0), 10)
    imgSmall = cv2.resize(img, (ws, hs))
    if hands and not buttonPressed:
        hand = hands[0]
        lmlist = hand['lmList']
        cx, cy = hand['center']
        xVal = int(np.interp(lmlist[8][0], [width//2, w], [0, width]))
        yVal = int(np.interp(lmlist[8][1], [150, height - 150], [0, height]))
        indexFinger = [xVal, yVal]
        fingers = detector.fingersUp(hand)
        if cy <= gestureThreshold: #if hand is at the height of the threshold then we will consider the gestures
            #gesture 1 - move left
            if fingers == [1,0,0,0,0]:

                print("left")
                if imgNumber > 0:
                    imgNumber -= 1
                    buttonPressed = True
            # gesture 2 - move right
            if fingers == [0, 0, 0, 0, 1]:

                print("right")
                if imgNumber < len(pathimages) - 1:
                    imgNumber += 1
                    buttonPressed = True
        # gesture 3 - show pointer
        if fingers == [0, 1, 1, 0, 0]:
            cv2.circle(imgCurrent, indexFinger, 12, (0,0,255), cv2.FILLED)
        # gesture 4 - draw
        if fingers == [0, 1, 0, 0, 0]:
            cv2.circle(imgCurrent, indexFinger, 12, (0, 0, 255), cv2.FILLED)
    if buttonPressed:
        buttonCounter += 1
        if buttonCounter >= buttonDelay:
            buttonCounter = 0
            buttonPressed = False



    imgCurrent[h - hs: h, w - ws:w] = imgSmall
    cv2.imshow("Image", imgSmall)
    cv2.imshow("slides", imgCurrent)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

