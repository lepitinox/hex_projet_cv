import matplotlib
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.models import Sequential

from datamanagement.data_holder import DataHolder
from datamanagement.data_loader import train_df, test_df

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def create_model_cnn(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu',
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))
    return model


if __name__ == '__main__':
    # Configs
    data = DataHolder(train_df, test_df, update=False)
    data.sample(0.2)
    show_graph = False
    save_graph = "cnn.png"
    num_classes = 26
    input_shape = (64, 64, 3)
    spe = data.train_df.shape[0] // data.batch_size
    vs = data.test_df.shape[0] // data.batch_size

    model = create_model_cnn(input_shape, num_classes)

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy']
                  )

    hist = model.fit(data.train_generator,
                     steps_per_epoch=spe,
                     epochs=10,
                     validation_data=data.validation_generator,
                     validation_steps=vs,
                     verbose=1)

    plt.plot(hist.history["loss"], label="train_loss")
    plt.plot(hist.history["val_loss"], label="val_loss")
    plt.plot(hist.history["accuracy"], label="train_acc")
    plt.plot(hist.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    if show_graph:
        plt.show()
    if save_graph:
        plt.savefig(save_graph)
