import os
import numpy as np
import pandas as pd

from keras import layers, models, losses
import tensorflow as tf

def fit_model(train_data, train_labels, valid_data, valid_labels, model_path):   
    
    time_steps = 100
    height = 32
    width = 32
    input_shape = (width, height, time_steps, 1)
    num_classes= 3 
    
    model = models.Sequential()
    
    model.add(layers.TimeDistributed(layers.Conv2D(32, (3, 3), padding='same', activation='relu'),input_shape=input_shape))
    model.add(layers.TimeDistributed(layers.MaxPooling2D((3, 3))))
    #model.add(layers.Dropout(0.3))
    
    model.add(layers.TimeDistributed(layers.Conv2D(32, (3, 3), padding='same', activation='relu')))
    model.add(layers.TimeDistributed(layers.MaxPooling2D((2, 2))))
    model.add(layers.Dropout(0.3)) 
    
    model.add(layers.TimeDistributed(layers.Conv2D(64, (3, 3), padding='same', activation='relu')))
    model.add(layers.TimeDistributed(layers.MaxPooling2D((2, 2))))
    model.add(layers.Dropout(0.3))
    
    model.add(layers.TimeDistributed(layers.Flatten()))
        
    model.add(layers.LSTM(64))#64:LSTM units
    model.add(layers.Dense(32, activation='relu'))
    
    #直接輸出logit
    model.add(layers.Dense(num_classes))
    model.compile(optimizer='adam', loss=losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    
    model.fit(train_data, train_labels, validation_data=(valid_data, valid_labels), epochs=70, batch_size=int(train_data.shape[0]/5.0), shuffle=True)
    print("\nTraining complete")
    model.save(os.path.join(model_path, 'gesture_model_RDI_data_KD_teacher.h5'))
    print("\nsave complete, model saved to: ", os.path.join(model_path, 'gesture_model_RDI_data_KD_teacher.h5'))


current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(current_path)

processed_data_path=os.path.join(parent_path, "processed_data")
train_data = np.load(os.path.join(processed_data_path, 'train.npz'))
train_labels = train_data['labels']
train_data = train_data['data']

valid_data = np.load(os.path.join(processed_data_path, 'val.npz'))
valid_labels = valid_data['labels']
valid_data = valid_data['data']

teacher_model_path = os.path.join(current_path, "teacher_model")
fit_model(train_data, train_labels, valid_data, valid_labels, teacher_model_path)
