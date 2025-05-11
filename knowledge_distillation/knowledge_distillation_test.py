import os
import numpy as np
import pandas as pd

from keras import optimizers, losses

from keras import models

def predict_labels(model, data):
    predictions = model.predict(data)
    return predictions


current_path = os.path.dirname(os.path.abspath(__file__))

#model_path=os.path.join(current_path,'model',"KD_model_lowdata.h5")
model_path=r"D:\mmWave_gesture\data_and_model\model\diff_ep_KD_model\KD_model_120.h5"
print("Loading model from: ", model_path)
model=models.load_model(model_path)

test_data_path=os.path.join(os.path.dirname(current_path),'processed_data',"test.npz")#be careful. Do not use part of train or val
print("Loading test data from: ", test_data_path)

test_data=np.load(test_data_path)['data']
test_labels=np.load(test_data_path)['labels']

model.compile(optimizer=optimizers.Adam())
predicted_value = predict_labels(model, test_data)

#print("Predicted value : ", predicted_value)
output=pd.DataFrame(columns=['Predicted','Actual','Right or False','Predicted Value'])

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
output['Predicted Value'] = predicted_value.tolist()
print("Predicted Value")


output_path=os.path.join(current_path,'test_output',"predicted_labels_RDI_distillation_logit_lowdata.csv")
if not os.path.exists(os.path.dirname(output_path)):#if the csv not exists, terminal will raise an error
    os.makedirs(os.path.dirname(output_path))
output.to_csv(output_path, index=False)
print("Predicted labels saved to {} CSV file.".format(output_path))