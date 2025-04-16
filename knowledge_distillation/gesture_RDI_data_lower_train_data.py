import os
import numpy as np

from sklearn.model_selection import train_test_split

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