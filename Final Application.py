import cv2
import mediapipe as mp
import time
import numpy as np
import math
from csv import writer
import joblib
import pandas as pd
import random 
import tensorflow.keras.models as models


mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose(model_complexity=2,smooth_landmarks=(False),min_tracking_confidence=.75)


movie = "/Users/jacobsobota/Desktop/Long Jump Videos/test_video.mp4"

cap = cv2.VideoCapture(movie)



def angle3pt(a, b, c):
    """Counterclockwise angle in degrees by turning from a to c around b
        Returns a float between 0.0 and 360.0"""
    ang = math.degrees(
        math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
    return ang

feature_list_scaled = []
feature_list_normalized = []
full_list_scaled = []
full_list_normalized = []

zero_tuple = []
for i in range(33):
    zero_tuple.append((random.randint(1,10),random.randint(1,10)))  

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    #print(results)
    if results.pose_landmarks:
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            feature_list_scaled.append([cx,cy])
            feature_list_normalized.append([lm.x,lm.y])
            
            if len(feature_list_scaled)==33:
                full_list_scaled.append(feature_list_scaled)
                feature_list_scaled = []
            if len(feature_list_normalized)==33:
                full_list_normalized.append(feature_list_normalized)
                feature_list_normalized = []
    else:
        full_list_scaled.append(zero_tuple)



sx_data = pd.read_csv("/Users/jacobsobota/Desktop/Master's Project/touchdown_frame_feature_data.csv")


sx_data.columns = ['right_knee_angle', 'right_hip_angle', 'left_knee_angle',
       'left_hip_angle', 'right_ground_angle', 'left_ground_angle',
       'ankle_distance', 'right_hip_distance', 'left_hip_distance',
       'hip_distance','right_hand_hip_dist','left_hand_hip_dist','forward_foot','is_touchdown']


sx_data["forward_foot"] = sx_data["forward_foot"].map({"R": 1, "L": 0})
sx_data.fillna(0, inplace=True)


rkas_mean = sx_data['right_knee_angle'].mean()
rkas_sd = sx_data['right_knee_angle'].std()
rhas_mean = sx_data['right_hip_angle'].mean()
rhas_sd = sx_data['right_hip_angle'].std()
lkas_mean = sx_data['left_knee_angle'].mean()
lkas_sd = sx_data['left_knee_angle'].std()
lhas_mean = sx_data['left_hip_angle'].mean()
lhas_sd = sx_data['left_hip_angle'].std()
gars_mean = sx_data['right_ground_angle'].mean()
gars_sd = sx_data['right_ground_angle'].std()
gals_mean = sx_data['left_ground_angle'].mean()
gals_sd = sx_data['left_ground_angle'].std()


frame_list = []
sequence_list = []

for frame_number in range(len(full_list_scaled)):
    las = tuple(full_list_scaled[frame_number][27]) #left ankle
    lks = tuple(full_list_scaled[frame_number][25]) #left knee
    lhs = tuple(full_list_scaled[frame_number][23]) #left hip
    lss = tuple(full_list_scaled[frame_number][11]) #left shoulder
    lws = tuple(full_list_scaled[frame_number][15]) #left wrist
    ras = tuple(full_list_scaled[frame_number][28]) #right ankle
    rks = tuple(full_list_scaled[frame_number][26]) #right knee
    rhs = tuple(full_list_scaled[frame_number][24]) #right hip
    rss = tuple(full_list_scaled[frame_number][12]) #right shoulder
    rws = tuple(full_list_scaled[frame_number][16]) #right wrist
    
    rkas = int(angle3pt(rhs, rks, ras)) #right knee angle scaled
    rhas = int(360 - angle3pt(rss,rhs,rks)) #right hip angle scaled
    lkas = int(angle3pt(lhs, lks, las)) #left knee angle scaled
    lhas = int(360 - angle3pt(lss,lhs,lks)) #left hip angle scaled
    
    horizontal_right_scaled = (ras[0]+1,ras[1]) 
    horizontal_left_scaled = (las[0]+1,las[1])

    gars = int(angle3pt(rks, ras, horizontal_right_scaled)) #ground angle right scaled
    gals = int(angle3pt(lks, las, horizontal_left_scaled)) #ground angle left scaled
        
    ads = float((math.dist(ras,las))/(abs(math.dist(ras,rks))+abs(math.dist(rks,rhs)))) #ankle distance divided by leg length scaled

    
    rhds = float((rhs[0]-ras[0])/(abs(math.dist(ras,rks))+abs(math.dist(rks,rhs)))) #x distance between right ankle and hip divided by leg

    
    lhds = float((lhs[0]-las[0])/(abs(math.dist(las,lks))+abs(math.dist(lks,lhs))))#x distance between left ankle and hip divided by leg length

    
    hds = float(math.dist(lhs, rhs)/(math.dist(las,lks) + math.dist(lks,lhs))) #Distance between hip joints divided by left leg length. Useful to know if person is in proper camera position

    
    rhhds = float(math.dist(rws,rhs)/(math.dist(las,lks) + math.dist(lks,lhs))) #distance between righ wrist and right hip/left leg length

    
    lhhds = float(math.dist(lws, lhs)/(math.dist(las,lks) + math.dist(lks,lhs))) #distance between left wrist and left hip/left leg length

    
    if (las[0]-ras[0]) > 0:
        forward_foot = 1
    else:
        forward_foot = 0
    
    if (las[1]-ras[1]) > 0:
        takeoff_foot = 0
    else:
        takeoff_foot = 1

    rkasn = (rkas - rkas_mean)/rkas_sd
    rhasn = (rhas - rhas_mean)/rhas_sd
    lkasn = (lkas - lkas_mean)/lkas_sd
    lhasn = (lhas - lhas_mean)/lhas_sd
    garsn = (gars - gars_mean)/gars_sd
    galsn = (gals - gals_mean)/gals_sd

    scaled_values_frame = [rkasn,rhasn,lkasn,lhasn,garsn,galsn,ads,rhds,lhds,hds,rhhds,lhhds,forward_foot]
    scaled_values_sequence = [rkas,rhas,lkas,lhas,gars,gals,ads,rhds,lhds,hds,rhhds,lhhds,forward_foot,takeoff_foot]

    frame_list.append(scaled_values_frame)
    sequence_list.append(scaled_values_sequence)
    
frame_model = joblib.load("/Users/jacobsobota/Desktop/Master's Project/touchdown_recognition_model.pkl")

touchdown_position = 0
highest_pred = 0
for i in range(len(frame_list)):
    probability = frame_model.predict_proba([frame_list[i]])
    prediction_frame = probability[0][1]
    
    if prediction_frame > highest_pred:
        highest_pred = prediction_frame
        frame_number_touchdown = i
        if frame_list[i][12] == 1:
            ground_angle = sequence_list[i][4]
            touchdown_knee_angle = sequence_list[i][0]
        else:
            ground_angle = sequence_list[i][5]
            touchdown_knee_angle = sequence_list[i][2]

sequence_model = models.load_model("/Users/jacobsobota/Desktop/Master's Project/sequence_recognition_model5.h5")

max_len = 5
step_size = 1

# Initialize a list to store the output
output_list = []

# Initialize variables to store the current label, start frame, and class1 probability
current_label = -1
start_frame = 0
class1_probability = 0

# Loop over the video frames, breaking the video into sequences of length max_len with step size step_size
for i in range(0, len(sequence_list), step_size):
    # Check if there are enough frames left in the video to form a complete sequence of length max_len
    if i + max_len <= len(sequence_list):
        # Get the sequence of frames
        sequence = sequence_list[i:i+max_len]

        # Pad the sequence if necessary
        if len(sequence) < max_len:
            sequence = sequence + [[0] * len(sequence[0]) for _ in range(max_len - len(sequence))]

        # Reshape the sequence into a 3D tensor with shape (1, max_len, num_features)
        sequence = np.array(sequence).reshape(1, max_len, -1)

        # Predict the label and probability for the sequence
        label = sequence_model.predict(sequence)
        class1_probability = probability[0][1]
        predicted_label = np.argmax(label)

        # If the predicted label is different from the previous label, record the start and end frames and append to output_list
        if predicted_label != current_label:
            end_frame = i - step_size
            output_list.append([start_frame, end_frame, current_label])
            start_frame = i
            current_label = predicted_label

# If the last sequence does not extend to the end of the video, record the end frame and append to output_list
if current_label != -1 and end_frame != len(sequence_list) - 1:
    end_frame = len(sequence_list) - 1
    output_list.append([start_frame, end_frame, current_label])

for i in range(len(output_list)):
    if output_list[i][2] == 1 and (abs(output_list[i][0]-frame_number_touchdown) <= 5) :
        takeoff_prediction = output_list[i]

minimum_knee = 360
for i in range(takeoff_prediction[0],takeoff_prediction[1]+1):
    if sequence_list[i][13] == 1 and sequence_list[i][0] <= minimum_knee:
        minimum_knee = sequence_list[i][0]
    if sequence_list[i][13] == 0 and sequence_list[i][2] <= minimum_knee:
       minimum_knee = sequence_list[i][2] 
   


def repeat(x):

    frame_number = int(cv2.getTrackbarPos('frame_number','one_frame'))
    cap.set(1,frame_number); # Where frame_no is the frame you want
    ret, frame = cap.read() # Read the frame
    cv2.rectangle(frame, (10,5), (500, 190), (128, 128, 128,0), -1)
    cv2.putText(frame, "Frame #: " + str(frame_number), (20,30), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0),2)
    
    
    las = tuple(full_list_scaled[frame_number][27]) #left ankle
    lan = tuple(full_list_normalized[frame_number][27])
    lks = tuple(full_list_scaled[frame_number][25]) #left knee
    lkn = tuple(full_list_normalized[frame_number][25])
    lhs = tuple(full_list_scaled[frame_number][23]) #left hip
    lhn = tuple(full_list_normalized[frame_number][23])
    lss = tuple(full_list_scaled[frame_number][11]) #left shoulder
    lsn = tuple(full_list_normalized[frame_number][11])
    lws = tuple(full_list_scaled[frame_number][15]) #left wrist
    lwn = tuple(full_list_normalized[frame_number][15])
    ras = tuple(full_list_scaled[frame_number][28]) #right ankle
    ran = tuple(full_list_normalized[frame_number][28])
    rks = tuple(full_list_scaled[frame_number][26]) #right knee
    rkn = tuple(full_list_normalized[frame_number][26])
    rhs = tuple(full_list_scaled[frame_number][24]) #right hip
    rhn = tuple(full_list_normalized[frame_number][24])
    rss = tuple(full_list_scaled[frame_number][12]) #right shoulder
    rsn = tuple(full_list_normalized[frame_number][12])
    rws = tuple(full_list_scaled[frame_number][16]) #right wrist
    rwn = tuple(full_list_normalized[frame_number][16])

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
    
    if (las[0]-ras[0]) > 0:
        forward_foot = 1
    else:
        forward_foot = 0

    rkass = (rkas - rkas_mean)/rkas_sd
    rhass = (rhas - rhas_mean)/rhas_sd
    lkass = (lkas - lkas_mean)/lkas_sd
    lhass = (lhas - lhas_mean)/lhas_sd
    garss = (gars - gars_mean)/gars_sd
    galss = (gals - gals_mean)/gals_sd

    scaled_values = [[rkass,rhass,lkass,lhass,garss,galss,ads,rhds,lhds,hds,rhhds,lhhds,forward_foot]]
    normalized_values = [rkan,rhan,lkan,lhan,garn,galn,adn,rhdn,lhdn,hdn,rhhdn,lhhdn,forward_foot]
    

    if 60 <= ground_angle <= 70:
        color = (125,255,0)
    else:
        color = (0,0,255)
    
    if 165 <= touchdown_knee_angle <= 170:
        color2 = (125,255,0)
    else:
        color2 = (0,0,255)
        
    if minimum_knee >= 135:
        color3 = (125,255,0)
    else:
        color3 = (0,0,255)
    
    cv2.putText(frame, "Ground Angle Opimal Range: " + str("60-70") + " degrees" , (20,50), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0),2)
    cv2.putText(frame, "Touchdown Ground Angle: " + str(ground_angle), (20,70), cv2.FONT_HERSHEY_PLAIN, 1, color ,2)
    
    cv2.putText(frame, "Optimal Touchdown Knee Angle: " + str("165-170") + " degrees" , (20,100), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0),2)
    cv2.putText(frame, "Touchdown Knee Angle: " + str(touchdown_knee_angle), (20,120), cv2.FONT_HERSHEY_PLAIN, 1, color2,2)
    cv2.putText(frame, "Right Knee Angle: " + str(rkas), (20,135), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0),2)
    cv2.putText(frame, "Left Knee Angle: " + str(lkas), (250,135), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0),2)
    
    cv2.putText(frame, "Minimum Recommended Knee Angle: " + str("135") + " degrees" , (20,160), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0),2)
    cv2.putText(frame, "Minimum Knee Angle: " + str(minimum_knee), (20,180), cv2.FONT_HERSHEY_PLAIN, 1, color3,2)
    
    if frame_number == frame_number_touchdown:
        cv2.putText(frame, "Touchdown Frame" , (200,30), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0),2)    
    
    
    
    
 
        
    cv2.imshow('one_frame', frame) # show frame on window

cv2.namedWindow('one_frame')

cv2.createTrackbar('frame_number','one_frame',0,200, repeat)
repeat(0)

while True:
    ch = 0xFF & cv2.waitKey(1) # Wait for a second
    if ch == 27:
        break

        
        
   
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        