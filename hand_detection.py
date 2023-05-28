from cvzone.HandTrackingModule import HandDetector
import cv2
import numpy as np

cap = cv2.VideoCapture(0)
detector = HandDetector(detectionCon=0.8, maxHands=2)
x1, y1 = [0,0], [0,0]
coors = [[0,0], [0,0]]
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
        for i in range(len(hands)):
            hand = hands[i]
            lmList1 = hand["lmList"]  # List of 21 Landmark points
            bbox1 = hand["bbox"]  # Bounding box info x,y,w,h
            centerPoint1 = hand['center']  # center of the hand cx,cy
            handType1 = hand["type"]  # Handtype Left or Right
            print(lmList1)
            fingers1 = detector.fingersUp(hand)
            #print(fingers1.count(1))
            # img = cv2.circle(img, (lmList1[12][0], lmList1[12][1]), 10, (0, 0, 255), 10)
            if fingers1.count(1) == 1:
                # mode = write
                print("writing mode")
                coors[i] = [lmList1[8][0], lmList1[8][1]]
                if x1[i] == 0 and y1[i] == 0:
                    x1[i],y1[i]= coors[i][0], coors[i][1]
                    # print('coors', coors)
                    # print('coors', (x1[i],y1[i]))

                else:
                    # Draw the line on the canvas
                    print((x1[i],y1[i]),coors[i])
                    canvas = cv2.line(canvas, (x1[i],y1[i]),(coors[i][0], coors[i][1]), [0,0,255], 10)
                
                # After the line is drawn the new points become the previous points.
                x1[i],y1[i]= coors[i][0], coors[i][1]
            if fingers1.count(1) >3:
                # mode = erase
                coords = [lmList1[12][0], lmList1[12][1]]
            #     print('erase coors', coords)
                # print(canvas.shape, img.shape)
                erase_size = 30
                canvas[lmList1[12][1]-erase_size-1:lmList1[12][1]+erase_size, lmList1[12][0]-erase_size-1:lmList1[12][0]+erase_size] = 0
    img = cv2.add(img,canvas)
        

    # Display
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()