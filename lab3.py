import cv2
import pytesseract
import numpy as np   


cap=cv2.VideoCapture(0)

while True:
    ret, frame =cap.read()

    imgH,imgW,_ =frame.shape
    x1,y1,w1,h1=0,0,imgH,imgW

    pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
    imgchar = pytesseract.image_to_string(frame)
    imgboxes= pytesseract.image_to_boxes(frame)

    #draw boxes around like contour analysis
    for boxes in imgboxes.splitlines():
        boxes=boxes.split(' ')
        x,y,w,h = int(boxes[1]),int(boxes[2]),int(boxes[3]),int(boxes[4])
        cv2.rectangle(frame, (x,imgH-y),(w,imgH-h),(0,0,255),3)

    #add text
    cv2.putText(frame,imgchar,(x1+int(w1/50),y1+int(h1/50)),cv2.FONT_HERSHEY_COMPLEX,0.7,(255,0,0),2)
    print(imgchar)

    cv2.imshow('Text Detection with Based Contour Analysis',frame)


    if cv2.waitKey(1)== ord('q'): #wait up to 1 milisecond (if we press q then we break)
        break

cap.release()
cv2.destroyAllWindows()
        