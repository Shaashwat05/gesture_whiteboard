from cvzone.HandTrackingModule import HandDetector
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

model = load_model('mp_hand_gesture')

# Load class names
f = open('gesture.names', 'r')
classNames = f.read().split('\n')
f.close()
print(classNames)

cap = cv2.VideoCapture(0)
detector = HandDetector(detectionCon=0.8, maxHands=1)
while True:
    # Get image frame
    success, img = cap.read()
    # Find the hand and its landmarks
    hands, img = detector.findHands(img)  # with draw
    # hands = detector.findHands(img, draw=False)  # without draw
    

    if hands:
        # Hand 1
        hand1 = hands[0]
        lmList1 = hand1["lmList"]  # List of 21 Landmark points
        bbox1 = hand1["bbox"]  # Bounding box info x,y,w,h
        centerPoint1 = hand1['center']  # center of the hand cx,cy
        handType1 = hand1["type"]  # Handtype Left or Right

        prediction = model.predict([np.array(lmList1)[:, :2].tolist()], verbose=0)
        classID = np.argmax(prediction)
        className = classNames[classID]
        #img = cv2.putText(img, className, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
   

        fingers1 = detector.fingersUp(hand1)
        #print(fingers1.count(1))
        if fingers1.count(1) == 1:
            # mode = write
            coors = (lmList1[8][0], lmList1[8][1])
        if fingers1.count(1) == 5:
            # mode = erase
            coors = (lmList1[12][0], lmList1[12][1])
        
        if fingers1.count(1) == 0 or className == 'Fist':
            # model = erase everything 
            pass
        

    # Display
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()