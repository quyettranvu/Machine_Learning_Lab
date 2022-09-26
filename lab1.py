import numpy as np
import cv2

cap=cv2.VideoCapture(0) #put 0 means access to 1 webcam

#CascadeClassifier will load a classifier from a file
face_cascade=cv2.CascadeClassifier('C:\\Python310\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml')
eye_cascade=cv2.CascadeClassifier('C:\\Python310\\Lib\\site-packages\\cv2\\data\\haarcascade_eye.xml')

#Get a frame from our webcam
while True:
    ret,frame = cap.read()

    gray=cv2.cvtColor(frame,cv2.COLOR_RGBA2GRAY) #gray scale for image

    # detect faces: 1/scaleFactor 2/value for resizing 3/minNeighbors: specify how many neighbors each rectangle should have to retain it (better from 3-6)
    #4/minSize of object 5/maxSize of object
    faces=face_cascade.detectMultiScale(gray,1.3, 5)

    #draw rectangle
    for(x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

        #region of interest for scale gray image and color
        roi_gray=gray[y:y+h,x:x+w]
        roi_color=frame[y:y+h,x:x+w]

        eyes=eye_cascade.detectMultiScale(roi_gray, 1.3,5)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    cv2.imshow('frame',frame)

    if cv2.waitKey(1)== ord('q'): #wait up to 1 milisecond (if we press q then we break)
        break

cap.release()
cv2.destroyAllWindows()


