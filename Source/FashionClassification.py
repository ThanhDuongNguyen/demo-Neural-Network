from __future__ import print_function
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

# get image fashion data
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# convert image pixel from (0-255) to (0,1) 
train_images = train_images / 255.0
test_images = test_images / 255.0

#create model NN
model = keras.Sequential([
    #Flatten convert matrix (28x28) to (1x784)
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
#Add optimizer, loss function
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#Training madel
model.fit(train_images, train_labels, epochs=10)
#Save model
model.save('trained_model.h5')