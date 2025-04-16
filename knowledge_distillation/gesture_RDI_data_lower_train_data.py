import os
import numpy as np

from sklearn.model_selection import train_test_split

"""
def load_data(data_dir):
    data=[]
    labels=[]
    
    for gesture in os.listdir(data_dir):
        if gesture == 'pipi' or gesture == 'left':
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

def normalize_min_max(data):
    
    max_value = 0
    if np.max(data) > 0:
        max_value=np.max(data) 
    else :
        max_value=1.0
    
    return data / max_value

def split_data_into_files(data, labels, output_dir,test_size=0.6):
    # 確保輸出目錄存在
    os.makedirs(output_dir, exist_ok=True)

    data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=test_size, random_state=42)
    
    #print("data_train:",len(data_train))
    #print("data_test:",len(data_test))
    
    np.savez_compressed(os.path.join(output_dir, 'train_0.4.npz'), data=data_train, labels=labels_train)
    np.savez_compressed(os.path.join(output_dir, 'test_0.6.npz'), data=data_test, labels=labels_test)

    print("\nData split and saved successfully!")

current_path = os.path.dirname(os.path.abspath(__file__))
load_data_path = os.path.join(current_path, 'recorder')
data, labels = load_data(load_data_path)

labels = preprocess_label(labels)
#print("labels:",labels)

data=normalize_min_max(data)

data = np.array(data)
labels = np.array(labels)

process_data_path=os.path.join(current_path, 'processed_data')
split_data_into_files(data, labels, process_data_path)

"""


#origin_test_data = os.path.join(os.path.dirname(os.path.abspath(__file__)),  'processed_data', 'test.npz')
origin_test_data=r"D:\gesture_recognition_by_mmWave_and_AI\processed_data\test.npz"
record = np.load(origin_test_data)
data=record['data']
labels=record['labels']

data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.6, random_state=42)

output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'processed_data')

np.savez_compressed(os.path.join(output_dir, 'KD_train_0.4.npz'), data=data_train, labels=labels_train)
np.savez_compressed(os.path.join(output_dir, 'KD_test_0.6.npz'), data=data_test, labels=labels_test)

print("\nKD Data split and saved successfully!")