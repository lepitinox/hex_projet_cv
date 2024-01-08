import os
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from data_holder import DataHolder
from data_loader import train_df, test_df
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


data = DataHolder(train_df, test_df, update=False)

x_train, y_train, x_test, y_test = data.give_me_my_data(data_type="RGB")

train_df = pd.DataFrame({"filepath": x_train["path"], "label": y_train})
validation_df = pd.DataFrame({"filepath": x_test["path"], "label": y_test})

train_df["label"] = train_df["label"].astype(str)
validation_df["label"] = validation_df["label"].astype(str)

train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

# This generator will read images found in subfolders of 'data/train',
# and indefinitely generate batches of augmented image data
train_generator = train_datagen.flow_from_dataframe(dataframe=train_df,
                                                    directory="data/train",
                                                    x_col="filepath",
                                                    y_col="label",
                                                    class_mode="categorical",
                                                    target_size=(64, 64),
                                                    batch_size=32)

# This is a similar generator, for validation data
validation_generator = test_datagen.flow_from_dataframe(dataframe=validation_df,
                                                        directory="data/test",
                                                        x_col="filepath",
                                                        y_col="label",
                                                        class_mode="categorical",
                                                        target_size=(64, 64),
                                                        batch_size=32)

# 26 categories to classify
num_classes = 26

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', activation='relu',
                    input_shape=(64, 64, 3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(num_classes, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy',
                metrics=['accuracy'])

hist = model.fit(train_generator,
                            steps_per_epoch=100,
                            epochs=10,
                            validation_data=validation_generator,
                            validation_steps=800,
                            verbose=1)

model.save('model.h5')

# Plot the training and validation loss + accuracy
plt.plot(hist.history["loss"], label="train_loss")
plt.plot(hist.history["val_loss"], label="val_loss")
plt.plot(hist.history["accuracy"], label="train_acc")
plt.plot(hist.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()
plt.savefig("plot.png")
