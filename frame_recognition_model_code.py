#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 23:41:38 2023

@author: jacobsobota
"""
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import joblib

data = pd.read_csv("/Users/jacobsobota/Desktop/Master's Project/scaled_final.csv")

data.columns = ['right_knee_angle', 'right_hip_angle', 'left_knee_angle',
       'left_hip_angle', 'right_ground_angle', 'left_ground_angle',
       'ankle_distance', 'right_hip_distance', 'left_hip_distance','hip_distance','right_hand_hip_dist','left_hand_hip_dist','forward_foot','is_touchdown']

data["forward_foot"] = data["forward_foot"].map({"R": 1, "L": 0})
data.fillna(0, inplace=True)

data.to_csv("/Users/jacobsobota/Desktop/Master's Project/frame_data_labeled.csv")


data_x = data[['right_knee_angle', 'right_hip_angle', 'left_knee_angle',
       'left_hip_angle', 'right_ground_angle', 'left_ground_angle',
       'ankle_distance', 'right_hip_distance', 'left_hip_distance','hip_distance','right_hand_hip_dist','left_hand_hip_dist','forward_foot']]

standardize_columns = ['right_knee_angle', 'right_hip_angle', 'left_knee_angle',
       'left_hip_angle', 'right_ground_angle', 'left_ground_angle']




for i in standardize_columns:
    data_x[i] = ( data_x[i] - data_x[i].mean() ) / data_x[i].std()

print(data_x.head())


data_y = data[['is_touchdown']]
data_y_arr = data_y.to_numpy().ravel()


X_train, X_test, y_train, y_test = train_test_split(data_x, data_y_arr, test_size=.3, random_state = 4)
gnb = GaussianNB()
clf = LogisticRegression()
gnb.fit(X_train,y_train)
clf.fit(X_train,y_train)
y_pred_gnb = gnb.fit(X_train, y_train).predict(X_test)
y_pred_clf = clf.fit(X_train, y_train).predict(X_test)


print("Number of mislabeled naive bayes points out of a total %d points : %d"
     % (X_test.shape[0], (y_test != y_pred_gnb).sum()))
print("Number of mislabeled logit regression points out of a total %d points : %d"
     % (X_test.shape[0], (y_test != y_pred_clf).sum()))


'''
joblib.dump(gnb, "/Users/jacobsobota/Desktop/Master's Project/scaled_bayes_model.pkl")
joblib.dump(clf, "/Users/jacobsobota/Desktop/Master's Project/scaled_logit_model.pkl")
'''




#Increase the number of iterations (max_iter) or scale the data as shown in:. Gotta test scaling data for performance
    















'''
data = pd.read_csv("/Users/jacobsobota/Desktop/Master's Project/normalized_data.csv")

scaled_data = data[['right_knee_angle', 'right_hip_angle', 'left_knee_angle',
       'left_hip_angle', 'right_ground_angle', 'left_ground_angle',
       'ankle_distance', 'right_hip_distance', 'left_hip_distance',
       'takeoff_foot', 'is_takeoff125']]

scaled_data["takeoff_foot"] = scaled_data["takeoff_foot"].map({"R": 1, "L": 2})
scaled_data = scaled_data.fillna(0)

scaled_x = scaled_data[['right_knee_angle', 'right_hip_angle', 'left_knee_angle',
       'left_hip_angle', 'right_ground_angle', 'left_ground_angle',
       'ankle_distance', 'right_hip_distance', 'left_hip_distance',
       'takeoff_foot']]
scaled_y = scaled_data[['is_takeoff125']]
scaled_y_arr = scaled_y.to_numpy().ravel()

X_train, X_test, y_train, y_test = train_test_split(scaled_x, scaled_y_arr, test_size=0.15, random_state=0)
gnb = GaussianNB()
gnb.fit(X_train,y_train)
y_pred = gnb.fit(X_train, y_train).predict(X_test)

print("Number of mislabeled points out of a total %d points : %d"
     % (X_test.shape[0], (y_test != y_pred).sum()))

gnb.fit(X_train, y_train)

# Predict the class probabilities for the testing data
y_proba_gnb = gnb.predict_proba(X_test)
print(y_proba_gnb)

'''
#joblib.dump(gnb, "/Users/jacobsobota/Desktop/Master's Project/scaled_data_model.pkl")


'''
data = pd.read_csv("/Users/jacobsobota/Desktop/Master's Project/normalized_data.csv")

scaled_data = data[['right_knee_angle', 'right_hip_angle', 'left_knee_angle',
       'left_hip_angle', 'right_ground_angle', 'left_ground_angle',
       'ankle_distance', 'right_hip_distance', 'left_hip_distance', 'is_takeoff125']]


scaled_x = scaled_data[['right_knee_angle', 'right_hip_angle', 'left_knee_angle',
       'left_hip_angle', 'right_ground_angle', 'left_ground_angle',
       'ankle_distance', 'right_hip_distance', 'left_hip_distance']]
scaled_y = scaled_data[['is_takeoff125']]
scaled_y_arr = scaled_y.to_numpy().ravel()

X_train, X_test, y_train, y_test = train_test_split(scaled_x, scaled_y_arr, test_size=0.01, random_state=4)
gnb = GaussianNB()
gnb.fit(X_train,y_train)
y_pred = gnb.fit(X_train, y_train).predict(X_test)

print("Number of mislabeled points out of a total %d points : %d"
     % (X_test.shape[0], (y_test != y_pred).sum()))


clf = LogisticRegression()

# Fit the classifier on the training data
clf.fit(X_train, y_train)

# Predict the classes of the testing data
y_pred = clf.predict(X_test)

# Compute the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

y_proba = clf.predict_proba(X_test)

print(y_proba)

#joblib.dump(gnb, "/Users/jacobsobota/Desktop/Master's Project/normalized_data_model_full_train.pkl")
'''













































