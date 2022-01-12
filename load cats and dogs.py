# -*- coding: utf-8 -*-
"""
@author: Rakshit
"""
from keras.models import Sequential 
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras import backend as K
from keras.models import model_from_json
from PIL import Image,ImageChops, ImageOps
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from scipy.misc import imresize
# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution   #fliters - dimension#3x3     colored img              rectifier
classifier.add(Convolution2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Convolution2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection        power of 2
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Part 2 - Fitting the CNN to the images

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

#dataset/train_set
training_set = train_datagen.flow_from_directory('E:/machinelearning/Data/train',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')
#dataset/test_set
test_set = test_datagen.flow_from_directory('E:/machinelearning/Data/test1',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

json_file= open('model.json', 'r')
loaded_model_json= json_file.read()
json_file.close()
loaded_model= model_from_json(loaded_model_json)
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

path = "E:/machinelearning/Data/test1/3259.jpg"

img = load_img(path)
plt.imshow(img)
img = imresize(img, (64, 64))
img = img_to_array(img)
img = img.reshape(1, 64, 64, 3)
#classes = ["dog", "cat"]

print(classifier.predict_classes(img))

