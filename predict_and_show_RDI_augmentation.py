import pandas as pd
import numpy as np
import os

from keras import models


def load_and_predict(model, data):

     predictions = model.predict(data)
     return predictions

current_path = os.path.dirname(os.path.abspath(__file__))
processed_data_path=os.path.join(current_path, "processed_data")
train_or_test_data='test_aug.npz'
test_data = np.load(os.path.join(processed_data_path, train_or_test_data))
test_labels = test_data['labels']
test_data = test_data['data']

model_path =os.path.join(current_path , "model/gesture_model_RDI_data_augmentation.h5")
load_model=models.load_model(model_path)

predicted_value = load_and_predict(load_model, test_data)


output=pd.DataFrame(columns=['Predicted','Actual','Right or False','Predicted Probability Value'])

pred_labels = np.argmax(predicted_value, axis=1)
output['Predicted'] = pred_labels
print("Predicted labels")

true_labels = np.array(test_labels)
output['Actual'] = true_labels
print("Actual labels")

right_or_false=[]
for pred, true in zip(pred_labels, true_labels):
    right_or_false.append('O' if pred == true else 'X')
right_or_false = np.array(right_or_false)
output['Right or False'] = right_or_false
print("Right or False")

predicted_value = np.array(predicted_value)
output['Predicted Probability Value'] = predicted_value.tolist()
print("Predicted Probability Value")

output_path=os.path.join(current_path, "test_output/predicted_labels_RDI_augmentation.csv")
output.to_csv(output_path, index=False)
print("Predicted {} labels saved to {} CSV file.".format(train_or_test_data,output_path))
