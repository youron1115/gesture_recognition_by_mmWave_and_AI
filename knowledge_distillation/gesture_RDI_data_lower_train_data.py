import os
import numpy as np

from sklearn.model_selection import train_test_split

current_file = os.path.abspath(__file__)
parent_dir =os.path.dirname(os.path.dirname(current_file))
origin_train_path = os.path.join(parent_dir,  'processed_data', 'train.npz')
print("Loading data from:", origin_train_path)#, "and", origin_valid_path)

origin_train_data = np.load(origin_train_path)

data_train=origin_train_data['data']
labels_train=origin_train_data['labels']

"""
#comcatenate train and valid data
concate_data = np.concatenate((data_train, data_valid), axis=0)
concate_labels = np.concatenate((labels_train, labels_valid), axis=0)
"""

train_size = 0.5
data_train,_, labels_train, _ = train_test_split(data_train, labels_train, test_size=1-train_size, random_state=42)


print("used {} data for training".format(train_size))

output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'processed_data')
print("Saving data to:", output_dir)


np.savez_compressed(os.path.join(output_dir, 'KD_train_{}.npz'.format(train_size)), data=data_train, labels=labels_train)
#
print("\nKD Data split and saved successfully!")
"""

np.savez_compressed(os.path.join(output_dir, 'KD_train.npz'), data=concate_data, labels=concate_labels)
print("\nKD data concate and save successfully!")
"""
