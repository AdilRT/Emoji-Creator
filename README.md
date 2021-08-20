# Emoji-Creator

# Requirements            ** Note: these are the packages version used while building the project

1. python >= v3.8
2. tensorflow >= 2.4
3. opencv = v4.5.3
4. PIL: pillow = v7.2.0
5. Cuda : for the use of GPU for faster training time


# Steps to follow for your convenience

1. pip install tensorflow
2. pip install opencv-python OR pip install opencv-contrib-python [for additional opencv community files]
3. pip install Pillow

# original dataset from https://www.kaggle.com/msambare/fer2013

# structure

1. Folders: emojis, models, test, train are in root directory
2. cnn_train and gui are python codes that also need to be in the root directory
3. emojis: images for each class of emotion that a model has to predict
4. models: trained neural network model
5. test: test dataset
6. train: training dataset 
