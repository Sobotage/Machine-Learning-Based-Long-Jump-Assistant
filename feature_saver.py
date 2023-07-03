#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 08:25:20 2023

@author: jacobsobota
"""

import cv2
import mediapipe as mp
import time
import numpy as np
import math
from csv import writer

feature_list_normalized = []
feature_list_scaled = []

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose

movie = "/Users/jacobsobota/Desktop/Long Jump Videos/LJ48.mp4"

cap = cv2.VideoCapture(movie)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

def angle3pt(a, b, c):
    """Counterclockwise angle in degrees by turning from a to c around b
        Returns a float between 0.0 and 360.0"""
    ang = math.degrees(
        math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
    return ang
#las,lan,lks,lkn,lhs,lhn,lss,lsn,ras,ran,rks,rkn,rhs,rhn,rss,rsn = 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0

def repeat(x):
    pose_static = mpPose.Pose(static_image_mode=(True),model_complexity=2,smooth_landmarks=(False),min_tracking_confidence=.65)
    frame_number = int(cv2.getTrackbarPos('frame_number','one_frame'))
    cap = cv2.VideoCapture(movie) #video_name is the video being called
    cap.set(1,frame_number); # Where frame_no is the frame you want
    ret, frame = cap.read() # Read the frame
    RGBimage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    drawings = pose_static.process(RGBimage)
    mpDraw.draw_landmarks(frame, drawings.pose_landmarks, mpPose.POSE_CONNECTIONS)
    cv2.putText(frame, "Frame #: " + str(cap.get(cv2.CAP_PROP_POS_FRAMES)), (20,50), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0),3)

#If you want to hold the window, until you press exit:
    if drawings.pose_landmarks:
        for id, lm in enumerate(drawings.pose_landmarks.landmark):
            h, w, c = frame.shape
            cx, cy, cz = int(lm.x * w), int(lm.y * h), int(lm.z * w)
            nx, ny = lm.x, lm.y
            if id == 27: #left ankle
                las = (cx,cy)
                lan = (nx,ny)
            if id == 25: #left knee
                lks = (cx,cy)
                lkn = (nx,ny)
            if id == 23: #left hip
                lhs = (cx,cy)
                lhn = (nx,ny)
            if id == 11: #left shoulder
                lss = (cx,cy)
                lsn = (nx,ny)
            if id == 28: #right ankle
                ras = (cx,cy)
                ran = (nx,ny)
            if id == 26: #right knee
                rks = (cx,cy)
                rkn = (nx,ny)
            if id == 24: #right hip
                rhs = (cx,cy)
                rhn = (nx,ny)
            if id == 12: #right shoulder
                rss = (cx,cy)
                rsn = (nx,ny)
            if id == 16: #right wrist
                rws = (cx,cy)
                rwn = (nx,ny)
            if id == 15: #left wrist
                lws = (cx,cy)
                lwn = (nx,ny)

        for i in [las,lks,lhs,lss,lws]:
            cv2.circle(frame, i ,3,(0,255,0),cv2.FILLED)
        for i in [ras,rks,rhs,rss,rws]:
            cv2.circle(frame, i ,3,(255,0,0),cv2.FILLED)
            
    
    
        rkas = int(angle3pt(rhs, rks, ras)) #right knee angle scaled
        rkan = int(angle3pt(rhn, rkn, ran)) #right knee angle normalized
        rhas = int(360 - angle3pt(rss,rhs,rks)) #right hip angle scaled
        rhan = int(360 - angle3pt(rsn,rhn,rkn)) #right hip angle normalized
        lkas = int(angle3pt(lhs, lks, las)) #left knee angle scaled
        lkan = int(angle3pt(lhn, lkn, lan)) #left knee angle normalized
        lhas = int(360 - angle3pt(lss,lhs,lks)) #left hip angle scaled
        lhan = int(360 - angle3pt(lsn,lhn,lkn)) #left hip angle normalized
        
        horizontal_right_scaled = (ras[0]+1,ras[1]) 
        horizontal_right_normalized = (ran[0]+1,ran[1])
        horizontal_left_scaled = (las[0]+1,las[1])
        horizontal_left_normalized = (lan[0]+1,lan[1])
    
        gars = int(angle3pt(rks, ras, horizontal_right_scaled)) #ground angle right scaled
        garn = int(angle3pt(rkn, ran, horizontal_right_normalized)) #ground angle right normalized
        gals = int(angle3pt(lks, las, horizontal_left_scaled)) #ground angle left scaled
        galn = int(angle3pt(lkn, lan, horizontal_left_normalized)) #ground angle left normalized
            
        ads = float((math.dist(ras,las))/(abs(math.dist(ras,rks))+abs(math.dist(rks,rhs)))) #ankle distance divided by leg length scaled
        adn = float((math.dist(ran,lan))/(abs(math.dist(ran,rkn))+abs(math.dist(rkn,rhn)))) #ankle distance divided by leg length normalized
        
        rhds = float((rhs[0]-ras[0])/(abs(math.dist(ras,rks))+abs(math.dist(rks,rhs)))) #x distance between right ankle and hip divided by leg
        rhdn = float((rhn[0]-ran[0])/(abs(math.dist(ran,rkn))+abs(math.dist(rkn,rhn))))
        
        lhds = float((lhs[0]-las[0])/(abs(math.dist(las,lks))+abs(math.dist(lks,lhs))))#x distance between left ankle and hip divided by leg length
        lhdn = float((lhn[0]-lan[0])/(abs(math.dist(lan,lkn))+abs(math.dist(lkn,lhn))))
        
        hds = float(math.dist(lhs, rhs)/(math.dist(las,lks) + math.dist(lks,lhs))) #Distance between hip joints divided by left leg length. Useful to know if person is in proper camera position
        hdn = float(math.dist(lhn,rhn)/(math.dist(lan,lkn) + math.dist(lkn,lhn)))
        
        rhhds = float(math.dist(rws,rhs)/(math.dist(las,lks) + math.dist(lks,lhs))) #distance between righ wrist and right hip/left leg length
        rhhdn = float(math.dist(rwn, rhn)/(math.dist(lan,lkn) + math.dist(lkn,lhn)))
        
        lhhds = float(math.dist(lws, lhs)/(math.dist(las,lks) + math.dist(lks,lhs))) #distance between left wrist and left hip/left leg length
        lhhdn = float(math.dist(lwn, lhn)/(math.dist(lan,lkn) + math.dist(lkn,lhn)))
        
        forward_foot = None
        is_takeoff = 0 #is_takeoff value of zero means it is not a takeoff
        
    
        if (las[0]-ras[0]) > 0:
            forward_foot = 'R'
        else:
            forward_foot = 'L'
    
        if cv2.getTrackbarPos('Is_Takeoff','Switches') > 0:
            is_takeoff = 1
        
        scaled_values = [rkas,rhas,lkas,lhas,gars,gals,ads,rhds,lhds,hds,rhhds,lhhds,forward_foot,is_takeoff]
        normalized_values = [rkan,rhan,lkan,lhan,garn,galn,adn,rhdn,lhdn,hdn,rhhdn,lhhdn,forward_foot,is_takeoff]
        
        if cv2.getTrackbarPos('Save_Data','Switches') > 0:
            #feature_list_scaled.append(scaled_values)
            #feature_list_normalized.append(normalized_values)
            
            with open("/Users/jacobsobota/Desktop/Master's Project/scaled_final.csv", 'a') as s_object:
                writer_object_scaled = writer(s_object)
                writer_object_scaled.writerow(scaled_values)
                s_object.close()
            with open("/Users/jacobsobota/Desktop/Master's Project/normalized_final.csv", 'a') as n_object:
                writer_object_scaled = writer(n_object)
                writer_object_scaled.writerow(normalized_values)
                n_object.close()
    cv2.imshow('one_frame', frame) # show frame on window
    
cv2.namedWindow('one_frame')
cv2.namedWindow('Switches')
cv2.createTrackbar('frame_number','one_frame',0,total, repeat)
cv2.createTrackbar('Save_Data','Switches',0,1,repeat)
cv2.createTrackbar('Is_Takeoff','Switches',0,1,repeat)  
repeat(0)

while True:
    ch = 0xFF & cv2.waitKey(1) # Wait for a second
    if ch == 27:
        break
