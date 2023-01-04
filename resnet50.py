import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydot
from keras.datasets import mnist
from keras.initializers import glorot_uniform, random_uniform
from keras.models import Sequential
from keras.preprocessing import image
from keras.utils import to_categorical
from keras.utils.vis_utils import model_to_dot
from PIL import Image
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow import keras


def identity_block(X, filters, f, initializer=glorot_uniform):
    f1, f2, f3 = filters
    X_short = X
    conv1 = keras.layers.Conv2D(filters=f1, kernel_size=(1, 1), strides=(
        1, 1), padding='valid', kernel_initializer=initializer())
    batchnorm1 = keras.layers.BatchNormalization(axis=3)
    conv2 = keras.layers.Conv2D(filters=f2, kernel_size=(f, f), strides=(
        1, 1), padding='same', kernel_initializer=initializer())
    batchnorm2 = keras.layers.BatchNormalization(axis=3)
    conv3 = keras.layers.Conv2D(filters=f3, kernel_size=(1, 1), strides=(
        1, 1), padding='valid', kernel_initializer=initializer())
    batchnorm3 = keras.layers.BatchNormalization(axis=3)
    add_layer = keras.layers.Add()
    relu = keras.layers.Activation('relu')
    X = conv1(X)
    X = batchnorm1(X)
    X = relu(X)
    X = conv2(X)
    X = batchnorm2(X)
    X = relu(X)
    X = conv3(X)
    X = batchnorm3(X)
    X = add_layer([X, X_short])
    X = relu(X)
    return X


def conv_block(X, filters, f, s=2, initializer=glorot_uniform):
    f1, f2, f3 = filters
    X_short = X
    conv1 = keras.layers.Conv2D(filters=f1, kernel_size=(1, 1), strides=(
        s, s), padding='valid', kernel_initializer=initializer())
    batchnorm1 = keras.layers.BatchNormalization(axis=3)
    conv2 = keras.layers.Conv2D(filters=f2, kernel_size=(f, f), strides=(
        1, 1), padding='same', kernel_initializer=initializer())
    batchnorm2 = keras.layers.BatchNormalization(axis=3)
    conv3 = keras.layers.Conv2D(filters=f3, kernel_size=(1, 1), strides=(
        1, 1), padding='valid', kernel_initializer=initializer())
    batchnorm3 = keras.layers.BatchNormalization(axis=3)
    conv_short = keras.layers.Conv2D(filters=f3, kernel_size=(1, 1), strides=(
        s, s), padding='valid', kernel_initializer=initializer())
    batch_short = keras.layers.BatchNormalization(axis=3)
    add_layer = keras.layers.Add()
    relu = keras.layers.Activation('relu')
    X = conv1(X)
    X = batchnorm1(X)
    X = relu(X)
    X = conv2(X)
    X = batchnorm2(X)
    X = relu(X)
    X = conv3(X)
    X = batchnorm3(X)
    X_short = conv_short(X_short)
    X_short = batch_short(X_short)
    X = add_layer([X, X_short])
    X = relu(X)
    return X


def ResNet50(input_shape=(64, 64, 3), classes=6, initializer=glorot_uniform):
    input = keras.layers.Input(input_shape)
    X = keras.layers.ZeroPadding2D((3, 3))(input)
    X = keras.layers.Conv2D(filters=64, kernel_size=(
        7, 7), strides=(2, 2), kernel_initializer=initializer())(X)
    X = keras.layers.BatchNormalization(axis=3)(X)
    X = keras.layers.Activation('relu')(X)
    X = keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(X)
    X = conv_block(X, filters=[64, 64, 256], f=3, s=1)
    X = identity_block(X, filters=[64, 64, 256], f=3)
    X = identity_block(X, filters=[64, 64, 256], f=3)
    X = conv_block(X, filters=[128, 128, 512], f=3, s=2)
    X = identity_block(X, filters=[128, 128, 512], f=3)
    X = identity_block(X, filters=[128, 128, 512], f=3)
    X = identity_block(X, filters=[128, 128, 512], f=3)
    X = conv_block(X, filters=[256, 256, 1024], f=3, s=2)
    X = identity_block(X, filters=[256, 256, 1024], f=3)
    X = identity_block(X, filters=[256, 256, 1024], f=3)
    X = identity_block(X, filters=[256, 256, 1024], f=3)
    X = identity_block(X, filters=[256, 256, 1024], f=3)
    X = identity_block(X, filters=[256, 256, 1024], f=3)
    X = conv_block(X, filters=[512, 512, 2048], f=3, s=2)
    X = identity_block(X, filters=[512, 512, 2048], f=3)
    X = identity_block(X, filters=[512, 512, 2048], f=3)
    X = keras.layers.AveragePooling2D((2, 2))(X)
    X = keras.layers.Flatten()(X)
    X = keras.layers.Dense(classes, activation='softmax',
                           kernel_initializer=initializer())(X)
    model = keras.models.Model(inputs=input, outputs=X)
    return model


model = ResNet50(input_shape=(64, 64, 3), classes=6)

model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

# Load the data
train_dataset = h5py.File('train_signs.h5', "r")
x_train = np.array(train_dataset["train_set_x"][:])
y_train = np.array(train_dataset["train_set_y"][:])

test_dataset = h5py.File('test_signs.h5', "r")
x_test = np.array(test_dataset["test_set_x"][:])
y_test = np.array(test_dataset["test_set_y"][:])

# Normalize image vectors
x_train = x_train/255.0
x_test = x_test/255.0

# Reshape
y_train = y_train.reshape((y_train.shape[0], 1))
y_test = y_test.reshape((y_test.shape[0], 1))

# One-hot encoding
mlb = MultiLabelBinarizer()
y_train = mlb.fit_transform(y_train)
y_test = mlb.fit_transform(y_test)
classes = y_train.shape[1]

# Train the model
model.fit(x_train, y_train, epochs=20, batch_size=8, verbose=1)

# Save the model
model.save('ResNet50.h5')

# Load the model
model = keras.models.load_model('ResNet50.h5')

# Evaluate the model
preds = model.evaluate(x_train, y_train)
print("Loss = " + str(preds[0]))
print("Test Accuracy = " + str(preds[1]))


img_path = 'img.jpg'
img = image.load_img(img_path, target_size=(64, 64))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = x/255.0
print('Input image shape:', x.shape)
plt.imshow(img)
prediction = model.predict(x)
print("Predicted digit:", np.argmax(prediction))
