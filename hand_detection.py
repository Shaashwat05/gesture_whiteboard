from cvzone.HandTrackingModule import HandDetector
import cv2
import numpy as np

cap = cv2.VideoCapture(0)
detector = HandDetector(detectionCon=0.8, maxHands=1)
x1, y1 = 0,0
canvas = None
while True:
    # Get image frame
    success, img = cap.read()
    img = cv2.flip( img, 1 )
    if canvas is None: 
        canvas = np.zeros_like(img)

    # Find the hand and its landmarks
    hands, img = detector.findHands(img)  # with draw
    # hands = detector.findHands(img, draw=False)  # without draw
    
    if hands:
        # Hand 1
        hand1 = hands[0]
        for hand in hands:
            lmList1 = hand["lmList"]  # List of 21 Landmark points
            bbox1 = hand["bbox"]  # Bounding box info x,y,w,h
            centerPoint1 = hand['center']  # center of the hand cx,cy
            handType1 = hand["type"]  # Handtype Left or Right

            fingers1 = detector.fingersUp(hand)
            #print(fingers1.count(1))
            if fingers1.count(1) == 1:
                # mode = write
                print("writing mode")
                coors = (lmList1[8][0], lmList1[8][1])
                if x1 == 0 and y1 == 0:
                    x1,y1= coors
                    print('coors', coors)
                else:
                    # Draw the line on the canvas

                    canvas = cv2.line(canvas, (x1,y1),coors, [255,0,0], 10)
                
                # After the line is drawn the new points become the previous points.
                x1,y1= coors
            if fingers1.count(1) == 5:
                # mode = erase
                coors = (lmList1[12][0], lmList1[12][1])
    img = cv2.add(img,canvas)
        

    # Display
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()