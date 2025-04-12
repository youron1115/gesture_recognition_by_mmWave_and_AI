import os
import numpy as np
import pandas as pd
import h5py

from sklearn.model_selection import train_test_split

import tensorflow as tf
from keras import layers, models

def fit_model(data,labels,model_path):
    
    
    time_steps = 100
    height = 32
    width = 32
    input_shape = (width, height, time_steps, 1)
    num_classes= 1  #可設定num_classes 種手勢
    
    model = models.Sequential()
    
    model.add(layers.TimeDistributed(layers.Conv2D(32, (3, 3), activation='relu'),input_shape=input_shape))
    model.add(layers.TimeDistributed(layers.MaxPooling2D((2, 2))))
    model.add(layers.Dropout(0.3)) 
    
    model.add(layers.TimeDistributed(layers.Conv2D(64, (3, 3), activation='relu')))
    model.add(layers.TimeDistributed(layers.MaxPooling2D((2, 2))))
    model.add(layers.Dropout(0.3))
    
    model.add(layers.TimeDistributed(layers.Flatten()))
    
    #model.add(layers.Dropout(0.3)) 
    
    model.add(layers.LSTM(64))#64:LSTM units
    model.add(layers.Dense(32, activation='relu'))
    
    model.add(layers.Dense(num_classes, activation='sigmoid')) 

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    model.fit(data, labels, epochs=20, batch_size=14, validation_split=0.3, shuffle=True)
    print("\nTraining complete")
    
    model.save(os.path.join(model_path, 'gesture_model_RDI_data.h5'))
    print("\nsave complete")

current_path = os.path.dirname(os.path.abspath(__file__))

processed_data_path=os.path.join(current_path, "processed_data")
train_data = np.load(os.path.join(processed_data_path, 'train.npz'))
train_labels = train_data['labels']
train_data = train_data['data']

model_dir =os.path.join(current_path, "model")
fit_model(train_data, train_labels, model_dir)
