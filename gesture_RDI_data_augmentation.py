import os
import numpy as np
import h5py

from sklearn.model_selection import train_test_split

def load_data(data_dir):
    data=[]
    labels=[]
    
    for gesture in os.listdir(data_dir):
        
        for file in os.listdir(os.path.join(data_dir, gesture)):
            if file.endswith(".h5"):
                h5_file_path = os.path.join(data_dir, gesture, file)
                with h5py.File(h5_file_path, 'r') as h5_file:
                    data_label='DS1'
                    
                    #print("整理RDI資料")
                    RDI_data = h5_file[data_label][0]
                    data.append(RDI_data)
                    labels.append(gesture)
        
    return data, labels

def preprocess_label(labels):
    
    label_dict={'pipi': 0, 'left': 1}  
    
    for i in range(len(labels)):
        labels[i]=label_dict[labels[i]]
     
    return labels

def split_data(data, labels, test_size=0.3, validation_size=0.3):
    
    data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=test_size, random_state=42)
    data_train, data_val, labels_train, labels_val = train_test_split(data_train, labels_train, test_size=validation_size, random_state=42) 
    
    print("\nData split successfully!")
    
    return data_train, labels_train, data_val, labels_val, data_test, labels_test

def data_augmentation(data, labels):
    print("\nData augmentation started!")
    # just augmentiing training but not validating and test
    augmented_data = []
    augmented_labels = []

    ## Add Gaussian noise to the data
    for i in range(len(data)):
        noise = np.random.normal(0, 0.1, data.shape)
        noisy_image = data + noise
        augmented_data.append(noisy_image[i])
        augmented_labels.append(labels[i])

    print("\nData augmentation completed! returning...")
    return np.array(augmented_data), np.array(augmented_labels)

def normalize_min_max(data,max_val=-999,min_val=999):
    
    if max_val == -999 or min_val == 999:
        #calculating max and min value in def (for train data)
        max_val = np.max(data)
        min_val = np.min(data)  
        return data / max_val, max_val, min_val
    else:
        #normalizing test data with train data max and min value
        return (data - min_val) / (max_val - min_val)

def save_data(train_data, train_labels, val_data, val_labels, test_data, test_labels, output_dir):
    
    os.makedirs(output_dir, exist_ok=True)
    np.savez_compressed(os.path.join(output_dir, 'train_aug.npz'), data=train_data, labels=train_labels)
    np.savez_compressed(os.path.join(output_dir, 'val_aug.npz'), data=val_data, labels=val_labels)
    np.savez_compressed(os.path.join(output_dir, 'test_aug.npz'), data=test_data, labels=test_labels)
    
    print("\nData saved successfully!")

current_path = os.path.dirname(os.path.abspath(__file__))
load_data_path = os.path.join(current_path, 'recorder')
data, labels = load_data(load_data_path)

labels = preprocess_label(labels)
#print("labels:",labels)

data = np.array(data)
labels = np.array(labels)

#split
data_train_set, labels_train_set, data_val_set, labels_val_set, data_test_set, labels_test_set = split_data(data, labels, test_size=0.3, validation_size=0.3)

#train aug
data_train_set, labels_train_set = data_augmentation(data_train_set, labels_train_set)
"""
data_train_set, max_value, min_value = normalize_min_max(data_train_set)
data_val_set = normalize_min_max(data_val_set,max_value,min_value)
data_test_set = normalize_min_max(data_test_set,max_value,min_value)
"""

#save
save_data(data_train_set, labels_train_set, data_val_set, labels_val_set, data_test_set, labels_test_set, os.path.join(current_path, 'processed_data'))
