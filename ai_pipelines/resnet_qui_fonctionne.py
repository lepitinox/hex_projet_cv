import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import seaborn as sn

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Flatten, Dropout,BatchNormalization ,Activation
from tensorflow.keras.models import Model, Sequential

input_shape = (256, 256, 3)

base_model = tf.keras.applications.ResNet50(input_shape=input_shape,include_top=False,weights="imagenet")

model=Sequential()
model.add(base_model)
model.add(Dropout(0.5))
model.add(Flatten())
model.add(BatchNormalization())
model.add(Dense(32,kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(32,kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(32,kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(26,activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy',metrics=['accuracy'])

from datamanagement.data_holder import DataHolder
DataHolder.target_size = (256, 256)
data = DataHolder(None,None, update=False)
data.uniform_sample(0.1)
data.test_balanced_sample(1)
data.create_data_pipline()
train_dataset = data.train_generator
valid_dataset = data.validation_generator
history=model.fit(train_dataset, validation_data=valid_dataset,epochs = 60,verbose = 1,)
print(history.history.keys())

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')

