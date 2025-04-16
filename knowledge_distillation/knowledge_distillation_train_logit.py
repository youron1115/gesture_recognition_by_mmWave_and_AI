import os
import numpy as np
import tensorflow as tf

from keras import layers, models, losses
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

def student_model(num_classes):
    
    input_shapes = (32, 32, 100, 1)
    
    model = models.Sequential()  
          
    model.add(layers.TimeDistributed(layers.Conv2D(16, (3, 3), activation='relu'),input_shape=input_shapes))    
    model.add(layers.TimeDistributed(layers.MaxPooling2D((3, 3))))
    
    model.add(layers.TimeDistributed(layers.Conv2D(32, (3, 3), activation='relu'),input_shape=input_shapes))    
    model.add(layers.TimeDistributed(layers.MaxPooling2D((3, 3)))) 
    
    model.add(layers.TimeDistributed(layers.Flatten()))        

    model.add(layers.LSTM(32))   
    model.add(layers.Dense(32, activation='relu'))       
    model.add(layers.Dense(num_classes, activation='softmax'))     
    # model.fit(data, labels, epochs=20, batch_size=14, validation_split=0.3, shuffle=True)

    #print("\nStudent model summary:")
    #model.summary()
    return model
    
class Distiller(tf.keras.Model):#
    def __init__(self, student, teacher, temperature=1.0, alpha=0.5):
        super(Distiller, self).__init__()
        self.student = student
        self.teacher = teacher
        self.temperature = temperature
        """
        1.temperature越大，softmax輸出之對各類別分布越緩和(對負標籤的關注較多)
        2.student參數較少時無法學到所有teacher的知識，temperature可以較小
        """
        self.alpha = alpha
        
        self.student_loss_fn = losses.SparseCategoricalCrossentropy()#evaluating student loss by common loss function
        self.distillation_loss_fn = losses.KLDivergence()#evaluating distillation loss by KLDivergence(相對熵)
        self.metric = tf.keras.metrics.SparseCategoricalAccuracy(name="sparse categorical accuracy")

    def compile(self, optimizer):
        super(Distiller, self).compile()
        self.optimizer = optimizer

    def train_step(self, data):
        x, y_true = data

        #分類模型最後的softmax層輸出之probability作*soft-target(特徵映射)
        
        # 前向傳播
        teacher_pred = self.teacher(x, training=False)#即x輸入給teacher使之輸出(predict)
        with tf.GradientTape() as tape:#記錄梯度
            student_pred = self.student(x, training=True)#即x輸入給student使之輸出(predict)

            # student 對真實 label 的 loss
            student_loss = self.student_loss_fn(y_true, student_pred)#計算student對真實label的loss

            
            # predict->temperature scaling->softmax(make it smooth)
            teacher_soft = tf.nn.softmax(teacher_pred / self.temperature)
            student_soft = tf.nn.softmax(student_pred / self.temperature)
            distill_loss = self.distillation_loss_fn(teacher_soft, student_soft)#uses soft value of teacher and student predicting
            
            # 總損失
            total_loss = self.alpha * student_loss + (1 - self.alpha) * distill_loss
            #total loss = weighted avg of student loss and distillation loss

        #update gradient
        grads = tape.gradient(total_loss, self.student.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.student.trainable_variables))

        self.metric.update_state(y_true, student_pred)
        return {"loss": total_loss, "accuracy": self.metric.result()}

    def test_step(self, data):
        x, y_true = data
        student_pred = self.student(x, training=False)
        loss = self.student_loss_fn(y_true, student_pred)
        #similar to train_step but no gradient updating(sure because is in test step)
        self.metric.update_state(y_true, student_pred)
        return {"loss": loss, "accuracy": self.metric.result()}

def fit_model(data, labels, teacher_model_path, parameter_epoch=10):
    
    student = student_model(2)  # 使用自定義的 student 架構

    teacher_model_path=os.path.join(teacher_model_path, 'gesture_model_RDI_data.h5')
    teacher = models.load_model(teacher_model_path)  # 載入教師模型
    teacher.trainable = False  # 凍結權重
    
    distiller = Distiller(student=student, teacher=teacher, temperature=1.25, alpha=0.7)
    distiller.compile(optimizer=tf.keras.optimizers.Adam())#assign using what optimizer
    distiller.fit(x=train_data, y=train_labels, epochs=parameter_epoch, batch_size=int(data.shape[0]/5.0), validation_split=0.3)
    return distiller

def save_model(model, save_path):
    whole_path = os.path.join(save_path, 'KD_model_lowdata.h5')
    model.student.save(whole_path)
    print("\nModel saved to {} ".format(whole_path))

current_path = os.path.dirname(os.path.abspath(__file__))
train_data_path=os.path.join(current_path, 'processed_data', 'KD_train_0.4.npz')
train_data=np.load(train_data_path)['data']
train_labels=np.load(train_data_path)['labels']

epoch=50
kd_model=fit_model(train_data, train_labels, os.path.join(current_path, 'teacher_model'), epoch)
print("\n epoch = {} training complete".format(epoch))

save_path=os.path.join(os.path.dirname(__file__), 'model')
save_model(kd_model, save_path)


