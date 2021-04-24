
import cv2
import numpy as np
import urllib.request as urllib
import time



# Goal: from a stream or just a video, cut the silent parts of the input and center the frame if it finds a face

phone_stream_url = 'http://192.168.18.3:8080/shot.jpg'

def main():
    # Load trained face recognition classifier
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # Input, for now it is an image instead of a video
    #input_image = cv2.imread('people.jpg')
    #video_stream = cv2.VideoCapture('person_dancing.mp4')
    imgResp = urllib.urlopen(phone_stream_url)
    imgNp = np.array(bytearray(imgResp.read()),dtype=np.uint8)
    img = cv2.imdecode(imgNp,-1)
    
    video_stream = cv2.VideoCapture(0)

    while True:
        _, input_image = video_stream.read()
        # Convert the input into a grayscale image
        grayscaled_input = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

        # Detect the faces
        faces = face_cascade.detectMultiScale(grayscaled_input, 1.1, 4)

        # Just drawing a rectangle around it
        for (x, y, w, h) in faces:
            cv2.rectangle(input_image, (x, y), (x+w, y+h), (255, 0, 0), 2)

            #center_coordinates = (x + int(w/2), y + int(y/2))
            #cv2.circle(input_image, center_coordinates, w, (255, 0, 0), 2)
            #cv2.imshow('Cropped frame', input_image[(y-20):(y+h+20), (x-20):(x+w+20)])
        

        cv2.imshow('Frame', input_image)
        time.sleep(.1)
        
        # Stop if escape key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_stream.release()


if __name__ == "__main__":
    main()
