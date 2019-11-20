from __future__ import print_function

import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt
import numpy as np

model = keras.models.load_model('trained_model.h5')

# get image fashion data
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# convert image pixel from (0-255) to (0,1) 
test_images = test_images / 255.0

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

i = 0
result = model.predict(test_images)
plt.figure()
plt.imshow(test_images[i])
plt.colorbar()
plt.grid(False)
plt.xlabel("Dự đoán:" + class_names[np.argmax(result[i])]+"- Kết quả:" + class_names[test_labels[i]])
plt.show()


