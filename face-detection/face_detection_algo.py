import cv2
from random import randrange

#loading pre-trained data of faces from opencv
trained_face_data=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#choose image to detect faces in
#img=cv2.imread('group.jpg')

#to capture face in a video
webcam = cv2.VideoCapture(0)

while True:
    successful_fram_read, frame = webcam.read()

    #change image to greyscale
    grayscaleimg=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    #detect face in the image
    face_corordinates = trained_face_data.detectMultiScale(grayscaleimg)
    #print(face_corordinates)

    for (x,y,w,h) in face_corordinates:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(randrange(256),randrange(256),randrange(256)),10)

    #display the image
    cv2.imshow('facedetector',frame)
    key=cv2.waitKey(1)

    if key==65 or key==97:
        break
webcam.release()
#draw a rectangle around the face thats detected
#for (x,y,w,h) in face_corordinates:
  #  cv2.rectangle(frame,(x,y),(x+w,y+h),(randrange(256),randrange(256),randrange(256)),10)

#display the image
#cv2.imshow('facedetector',frame)
#cv2.waitKey()

print("project completed")