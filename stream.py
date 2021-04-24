
import urllib.request as urllib
import cv2
import numpy as np
import time

url='http://192.168.18.3:8080/shot.jpg'

while True:

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # Use urllib to get the image and convert into a cv2 usable format
    imgResp=urllib.urlopen(url)
    imgNp=np.array(bytearray(imgResp.read()),dtype=np.uint8)
    img=cv2.imdecode(imgNp,-1)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x -20, y -20), (x + w +40, y + h +40), (255, 0, 255), 2)
    
    cv2.imshow('Frame', img)

    #time.sleep(0.1) 

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#https://www.facebook.com/mrlunk