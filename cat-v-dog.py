# This is a basic deep learning experiment with cat dog classification following the deep learning course on kaggle
# Much of the coding is drawn from previous lessons from the deep learning course as an attempt to classify a set of images on my own

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

from os import listdir
from matplotlib import image
from matplotlib import pyplot
import os
#print(os.listdir("../input"))_array.reshape(num_images, img_rows, img_cols, 1)

catImages = listdir('../input/dogs-cats-images/dog vs cat/dataset/training_set/cats')
print ("Number of Cat images - ",str(len(catImages)))


def data_prep(raw, dog):
    out_y = keras.utils.to_categorical(raw.label, 2, dog)
    num_images = raw.shape[0]
    x_as_array = raw.values[:,1:]
    x_shaped_array = x_as_array.reshape(num_images, img_rows, img_cols, 1)
    out_x = x_shaped_array / 255
    return out_x, out_y

catImages = listdir('../input/dogs-cats-images/dog vs cat/dataset/training_set/cats')
print ("Number of Cat images - ",str(len(catImages)))

dogImages = listdir('../input/dogs-cats-images/dog vs cat/dataset/training_set/dogs')
print ("Number of Dog images - ",str(len(dogImages)))


cat_input = '../input/dogs-cats-images/dog vs cat/dataset/training_set/cats/'
dog_input = '../input/dogs-cats-images/dog vs cat/dataset/training_set/dogs/'
cat_val = '../input/dogs-cats-images/dog vs cat/dataset/test_set/cats/'
dog_val = '../input/dogs-cats-images/dog vs cat/dataset/test_set/dogs/'

train_dir = '../input/dogs-cats-images/dog vs cat/dataset/training_set/'
test_dir = '../input/dogs-cats-images/dog vs cat/dataset/test_set/'

def load_images(file_input):
    loaded_images = []
    x = 0
    for filename in listdir(file_input):
        x += 1
        # load image
        img_data = image.imread(file_input + filename)
        # store loaded image
        loaded_images.append(img_data)
        #print('> loaded %s %s' % (filename, img_data.shape))
        if x % 1000 == 0:
            print(x)
    return loaded_images
""" 
training_data = []
training_data.append(load_images(cat_input))
training_data.append(load_images(dog_input))
train_array = np.array(training_data)
validation_data = []
validation_data.append(load_images(cat_val))
validation_data.append(load_images(dog_val))
val_array = np.array(validation_data)
"""
data_generator = ImageDataGenerator(rescale = 1.0/255.0)
training_data = data_generator.flow_from_directory(directory = train_dir,
                                                   target_size = (100, 100),
                                                   batch_size = 50,
                                                   class_mode = 'binary')
testing_data = data_generator.flow_from_directory(directory = test_dir,
                                                  target_size = (100, 100),
                                                  batch_size = 50,
                                                  class_mode = 'binary')


model = Sequential()
model.add(Conv2D(50, kernel_size=3,
                 strides=2,
                 activation='relu',
                 input_shape = training_data.image_shape
                 ))
model.add(Dropout(0.5))
model.add(Conv2D(30, kernel_size=(3, 3), strides=2, activation='relu'))
model.add(Dropout(0.5))
model.add(Conv2D(30, kernel_size=(3, 3), strides=2, activation='relu'))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])


#train_generator = data_denerator.fl
fit_model = model.fit_generator(training_data,
                        steps_per_epoch = 500,
                        epochs = 5,
                        validation_data = testing_data,
                        validation_steps = 500)


accuracy = fit_model.history['val_accuracy']
pyplot.plot(range(len(accuracy)), accuracy,'bo', label = 'accuracy')