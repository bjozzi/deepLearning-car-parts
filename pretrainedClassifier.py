import os
import h5py
import numpy as np
from keras import optimizers
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Activation, Dropout, Flatten, Dense, ZeroPadding2D
from vgg16 import VGG16
from dataGenerators import getGenerators

numberOfEpochs = 15
numberOfTrainingSamples = 1204
numberOfValidationSamples = 200
numberOfTestingSamples = 428
numberOfChannels = 3
imageWidth = imageHeight = 150

dataPath = './car_data/'
dataPaths = {
    'training': dataPath + 'training',
    'validation': dataPath + 'validation',
    'testing': dataPath + 'testing'
}

# Data generators
generator = getGenerators(imageHeight, imageWidth, dataPaths)

# VGG16 model topology
model = Sequential()
model.add(ZeroPadding2D((1, 1), input_shape=(imageWidth, imageHeight, numberOfChannels)))

model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

vgg16model = VGG16(weights='imagenet', include_top=False)
model.set_weights(vgg16model.get_weights())

print('VGG16 model loaded')

# Define final fully connected layers
fcModel = Sequential()
fcModel.add(Flatten(input_shape=model.output_shape[1:]))
fcModel.add(Dense(256, activation='relu'))
fcModel.add(Dropout(0.5))
fcModel.add(Dense(1, activation='sigmoid'))

# Add on top of VGG16 model
model.add(fcModel)

# Freeze all layers up to last convolutional block
for layer in model.layers[:25]:
    layer.trainable = False

# Compile the model
model.compile(loss='binary_crossentropy', optimizer=optimizers.SGD(lr=1e-4, momentum=0.9), metrics=['accuracy'])

# Stop training when a monitored quantity has stopped improving
# early_stopping = EarlyStopping(monitor='val_loss', mode='min')

# fine-tune the model
model.fit_generator(
    generator['training'],
    samples_per_epoch=numberOfTrainingSamples,
    nb_epoch=numberOfEpochs,
    validation_data=generator['validation'],
    nb_val_samples=numberOfValidationSamples
    #       callbacks=[early_stopping]
)

model.save_weights('pretrained_model_weights.hdf5')

# Test the accuracy of the model
test = model.evaluate_generator(generator['testing'], val_samples=numberOfTestingSamples)
print("Loss: %.2f" % test[0])
print("Accuracy: %.2f%%" % (test[1] * 100))
