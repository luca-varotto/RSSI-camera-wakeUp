import cv2
import numpy as np
import time
import os 
from datetime import datetime
from matplotlib import pyplot as plt
import csv

############################################################################################################################


############################################################################################################################

#  COLLECT CAMERA DATA

############################################################################################################################

start_time = datetime.now()

dataset_type = "test"
wantSave = True 
wantRec = True

prefix = datetime.now().strftime("%m%d%Y")
output_path = './data/' + prefix 

# cascade classifier face
face_cascade = cv2.CascadeClassifier('./models/haarcascade_frontalface_default.xml') # haarcascade_frontalface_default,haarcascade_eye

# acquire and save video stream from ...
# ... IP camera
# url = "http://192.168.43.1:8080/" 
# cap = cv2.VideoCapture(url+"video") # resolution: 320x240
# ... PC webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)

# video recording
if wantRec:
    width = 640# int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
    height = 480# int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
    size = (width, height)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path+'-video.avi', fourcc, 20.0, size)

frames_counter = 0
timeStmp_list = []
detection_list = []
while(True):

    # ***** FRAME ACQUISITION ***** 
    ret, frame = cap.read()
    dt = datetime.now() - start_time
    timeStmp_list.append( (dt.days * 24 * 60 * 60 + dt.seconds)* 1000 + dt.microseconds / 1000.0 ) # store time stamp of current frame
    frames_counter += 1 
    # cv2.imshow('frame',frame)

    # ***** CASCADE CLASSIFIER *****
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
    try:
        out.write(frame)
    except NameError:
        pass
    cv2.imshow('Detection',frame)

    if len(faces) > 0:
        detection_list.append(1)
    else:
        detection_list.append(0)

    # close the stream video
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
try:
    out.release()
except NameError:
    pass

# create csv with timestamps and detection list
if wantSave:
    with open(output_path+"-camera-"+dataset_type+".csv", 'w', newline='') as csvfile:
        fieldnames = ['timestmp', 'detection']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i, j in zip(timeStmp_list, detection_list):
            writer.writerow({'timestmp': i, 'detection': j})
