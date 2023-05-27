from cvzone.HandTrackingModule import HandDetector
import cv2
import numpy as np

cap = cv2.VideoCapture(0)
detector = HandDetector(detectionCon=0.8, maxHands=2)
x1, y1 = [0,0], [0,0]
coors = [[0,0], [0,0]]
canvas = None
color = [0,0,255]
while True:
    # Get image frame
    success, img = cap.read()
    img = cv2.flip( img, 1 )
    # color_picker = cv2.imread('color_picker.png')
    # img[:color_picker.shape[0],:color_picker.shape[1]] = color_picker
    color_options = [(000,000,000),(255,255,255),(127,127,127),(195,195,195),(136,000,21),(185,122,87),(237,28,36),(255,174,201),(255,127,39),(255,201,14),(255,242,000),(239,228,176),(34,177,76),(181,230,29),(000,162,232),(153,217,234),(63,72,204),(112,146,190),(163,73,164),(200,191,231)]
    color_size,row = 80, 0
    for i in range(len(color_options)):
        img = cv2.rectangle(img, (i*color_size, row), (i*color_size+color_size, color_size),color_options[i], -1)
    color_picker_shape = [ i* color_size,color_size]
    # print(color_picker_shape)

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
            handType1 = hand["type"]
          # Handtype Left or Right
            # print(lmList1)
            fingers1 = detector.fingersUp(hand)
            #print(fingers1.count(1))
            # img = cv2.circle(img, (lmList1[12][0], lmList1[12][1]), 10, (0, 0, 255), 10)
            if fingers1.count(1) == 1 and handType1 == 'Right':
                print('color select', color_picker_shape, img.shape)
                select_coord = [lmList1[8][0], lmList1[8][1]]
                print(select_coord)
                if select_coord[1]< color_picker_shape[1] and select_coord[0]< color_picker_shape[0]:
                    color = color_options[select_coord[0]//color_size]#img[select_coord[1]][select_coord[0]]
                    print('color selected',color)

            if fingers1.count(1) == 1 and handType1 == 'Left':
                # mode = write
                print("writing mode")
                coors[i] = [lmList1[8][0], lmList1[8][1]]
                if x1[i] == 0 and y1[i] == 0:
                    x1[i],y1[i]= coors[i][0], coors[i][1]
                    # print('coors', coors)
                    # print('coors', (x1[i],y1[i]))

                else:
                    # Draw the line on the canvas
                    # print((x1[i],y1[i]),coors[i])
                    print('write with color' , color)
                    canvas = cv2.line(canvas, (x1[i],y1[i]),(coors[i][0], coors[i][1]), (int(color[0]), int(color[1]), int(color[2])), 10)
                
                # After the line is drawn the new points become the previous points.
                x1[i],y1[i]= coors[i][0], coors[i][1]
            if fingers1.count(1) >3 and handType1 == 'Left':
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