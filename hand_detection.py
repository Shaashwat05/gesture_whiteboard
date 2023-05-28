from cvzone.HandTrackingModule import HandDetector
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np


model = load_model('mp_hand_gesture')
model_predict = load_model('QuickDraw.h5')
prediction_class_names = ["Apple","Bowtie","Candle","Door","Envelope","Fish","Guitar","Ice Cream","Lightning","Moon","Mountain","Star","Tent","Toothbrush","Wristwatch"]
# Load class names
f = open('gesture.names', 'r')
classNames = f.read().split('\n')
f.close()
print(classNames)

def keras_predict(model_predict, image):
    processed = keras_process_image(image)
    # print("processed: " + str(processed.shape))
    # print(model_predict.summary())
    pred_probab = model_predict.predict(processed, verbose = 0)[0]
    pred_class = list(pred_probab).index(max(pred_probab))
    return max(pred_probab), pred_class


def keras_process_image(img):
    image_x = 28
    image_y = 28
    img = cv2.resize(img, (image_x, image_y))
    print(img.shape)
    img = np.array(img, dtype=np.float32)
    img = np.reshape(img, (-1, image_x, image_y, 1))
    return img

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
            if handType1 == 'Right':
                prediction = model.predict([np.array(lmList1)[:, :2].tolist()], verbose=0)
                classID = np.argmax(prediction)
                className = classNames[classID]
                # print(className)
                if fingers1.count(1) == 0 and (className == 'fist' or className == 'rock'):
                    canvas = np.zeros_like(img)
                    x1, y1 = [0,0], [0,0]
                    coors = [[0,0], [0,0]]
                    break

            if fingers1.count(1) == 1 and handType1 == 'Right':
                # print('color select', color_picker_shape, img.shape)
                select_coord = [lmList1[8][0], lmList1[8][1]]
                print(select_coord)
                if select_coord[1]< color_picker_shape[1] and select_coord[0]< color_picker_shape[0]:
                    color = color_options[select_coord[0]//color_size]#img[select_coord[1]][select_coord[0]]
                    print('color selected',color)
                break

            if fingers1.count(1) == 1 and handType1 == 'Left':
                # mode = write
                # print("writing mode")
                print(coors)
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
    prediction_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    blur1 = cv2.medianBlur(prediction_canvas, 15)
    blur1 = cv2.GaussianBlur(blur1, (5, 5), 0)
    thresh1 = cv2.threshold(blur1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    blackboard_cnts = cv2.findContours(thresh1.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
    if len(blackboard_cnts) >= 1:
        cnt = max(blackboard_cnts, key=cv2.contourArea)
        print(cv2.contourArea(cnt))
        if cv2.contourArea(cnt) > 10000:
            x, y, w, h = cv2.boundingRect(cnt)
            digit = prediction_canvas[y:y + h, x:x + w]
            print('digit shape', digit.shape)
            pred_probab, pred_class = keras_predict(model_predict, digit)
            print(pred_class, pred_probab)
            print(pred_class, prediction_class_names[pred_class])
            cv2.putText(img, prediction_class_names[pred_class], (img.shape[1]//2, img.shape[0] - 50) , cv2.FONT_HERSHEY_SIMPLEX, 4, (255,255,255), 4)


    # prob, idx = keras_predict(model_predict, canvas)
    # print(idx, prediction_class_names[idx])
    # cv2.putText(img, prediction_class_names[idx], (1000, 50) , cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    img = cv2.add(canvas,img)
    stacked = np.hstack((canvas,img))    

    # Display
    # cv2.imshow("Image", img)
    cv2.imshow("Image", stacked)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

