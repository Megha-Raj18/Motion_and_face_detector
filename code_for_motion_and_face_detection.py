import cv2

import csv

import datetime as dt

import pandas as pd

import altair as alt

first_frame=None

face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

eyes_cascade=cv2.CascadeClassifier("haarcascade_eye.xml")

def find_face_and_eyes(img):
    
    global face_cascade,eyes_cascade
    faces=face_cascade.detectMultiScale(img,scaleFactor=1.05,minNeighbors=5)
    
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.equalizeHist(img_gray)
    
    for l,b,w,h in faces:
        img=cv2.rectangle(img,(l,b),(l+w,b+h),(255,0,0),2)
        faceROI = img_gray[b:b+h,l:l+w]
        eyes = eyes_cascade.detectMultiScale(faceROI)
        
        for (l2,b2,w2,h2) in eyes:
            eye_center = (l + l2 + w2//2, b + b2 + h2//2)
            radius = int(round((w2 + h2)*0.25))
            img = cv2.circle(img, eye_center, radius, (0, 0, 255 ), 2)
        
    cv2.imshow('Colour Frame',img)
    return
    

video=cv2.VideoCapture(0)

motion_list=[2,2]

motion_list_graph=[0]

timestamps=[]

timestamps_graph=[]

while True:
    
    chk,frame=video.read()
    motion=0
    
    gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray_frame=cv2.GaussianBlur(gray_frame,(21,21),0)
    
    if first_frame is None:
        first_frame=gray_frame
        timestamps_graph.append(dt.datetime.now())
        continue
        
    absdiff_frame=cv2.absdiff(first_frame,gray_frame)
    threshold_frame=cv2.threshold(absdiff_frame,50,255,cv2.THRESH_BINARY)[1]
    threshold_frame=cv2.dilate(threshold_frame,None,iterations=5)
    
    (cntrs,_)=cv2.findContours(threshold_frame.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in cntrs:
        
        if cv2.contourArea(contour)<10000:
            continue
        motion=1
        
        (l,b,w,h)=cv2.boundingRect(contour)
        cv2.rectangle(frame,(l,b),(l+w,b+h),(0,255,0),2)
        
    motion_list.append(motion) 
    
    if motion_list[-1]==1 and motion_list[-2]==0:
        motion_list_graph.append(1)
        timestamps.append(dt.datetime.now())
        
    if motion_list[-1]==0 and motion_list[-2]==1:
        motion_list_graph.append(0)
        timestamps.append(dt.datetime.now())
    
    cv2.imshow("Gray Frame",gray_frame)
    cv2.imshow("AbsDifference Frame",absdiff_frame)
    cv2.imshow("Threshold Frame",threshold_frame)
    find_face_and_eyes(frame)
    
    key=cv2.waitKey(1)
    if key==ord('q'):
        
        if motion==1:
            timestamps.append(dt.datetime.now())
            motion_list_graph.append(0)
            
        break
        
video.release()
cv2.destroyAllWindows()
    
    
with open('Time_Stamps.csv','w') as file_handle:
    writer_obj=csv.writer(file_handle)
    writer_obj.writerow(["START","END"])
    for i in range(0,len(timestamps),2):
        writer_obj.writerow([timestamps[i],timestamps[i+1]])

        
timestamps_graph.extend(timestamps)  
df_timestamps=pd.read_csv('Time_Stamps.csv',parse_dates=True)
