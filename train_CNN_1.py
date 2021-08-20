import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os.path
import matplotlib.pyplot as plt
from tensorflow.python import keras

def plot_history(history):
    """Plots accuracy/loss for training/validation set as a function of the epochs
        :param history: Training history of model
        :return:
    """

    fig, axs = plt.subplots(2)

    # create accuracy sublpot
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")

    # create error sublpot
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval")

    plt.show()

train_dir = 'C:/Users/Nitro 5/Desktop/Adil/data/train'
val_dir = 'C:/Users/Nitro 5/Desktop/Adil/data/test'
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(48,48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical')

validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(48,48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical')

#crete model
emotion_model = Sequential()

#1st conv layer
emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
#2 hidden layer
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(3, 3),strides=(2,2),padding='same'))
emotion_model.add(Dropout(0.3))
emotion_model.add(keras.layers.BatchNormalization())
#3
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(3, 3),strides=(2,2),padding='same'))
emotion_model.add(Dropout(0.3))
emotion_model.add(keras.layers.BatchNormalization())
#4
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(3, 3),strides=(2,2),padding='same'))
emotion_model.add(Dropout(0.3))
emotion_model.add(keras.layers.BatchNormalization())
#flatten the output and feed it to dense layer
emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation='relu',kernel_regularizer=keras.regularizers.l2(0.001)))
emotion_model.add(Dropout(0.3))

#output layer #dropout not used in the output
emotion_model.add(Dense(7, activation='softmax'))


# cv2.ocl.setUseOpenCL(False)

emotion_model.compile(loss='categorical_crossentropy',optimizer=Adam(learning_rate=0.0001, decay=1e-6),metrics=['accuracy'])
emotion_model_info = emotion_model.fit(
        train_generator,
        steps_per_epoch=28709 // 64,
        epochs=39,
        validation_data=validation_generator,
        validation_steps=7178 // 64)

if os.path.isfile('models/emotion_model_39_030303.h5') is False:
        emotion_model.save('models/emotion_model_539_030303.h5')

plot_history(emotion_model_info)
