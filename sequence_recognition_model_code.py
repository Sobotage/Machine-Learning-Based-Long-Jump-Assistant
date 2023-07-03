#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 00:02:14 2023

@author: jacobsobota
"""


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
import ast
from keras.utils import to_categorical

# Load the dataset
df = pd.read_csv("/Users/jacobsobota/Desktop/Master's Project/scaled_sequence_final.csv", header=None)
df.columns = ['features','video_number','class']
df['features'] = df['features'].apply(lambda x: ast.literal_eval(x))
df_grouped = df.groupby(['video_number', 'class'])['features'].apply(list).reset_index()
df_grouped["class"] = df_grouped["class"].map({"Running": 0, "Takeoff": 1, "Flight": 0})

#df_grouped.to_csv("/Users/jacobsobota/Desktop/Master's Project/sequence_grouped.csv")

X = df_grouped['features'].values
y = df_grouped['class'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

max_len = 8 # maximum sequence length
X_train = pad_sequences(X_train, maxlen=max_len, dtype='float32')
X_test = pad_sequences(X_test, maxlen=max_len, dtype='float32')

input_shape = (max_len, 14) # shape of each sequence

model = Sequential([
    LSTM(64, return_sequences=True, input_shape=input_shape),
    LSTM(16),
    Dense(2, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=25, batch_size=8)


loss, acc = model.evaluate(X_test, y_test)
print(f'Test loss: {loss:.3f}, test accuracy: {acc:.3f}')


#model.save("/Users/jacobsobota/Desktop/Master's Project/sequence_scaled_model8.h5")
'''
new_sequences = [...] # a list of sequences of features
new_sequences = pad_sequences(new_sequences, maxlen=max_len, dtype='float32')
predictions = model.predict(new_sequences)
'''




