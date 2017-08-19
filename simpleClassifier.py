from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from dataGenerators import getGenerators

numberOfEpochs = 30
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

# Define the model
model = Sequential()
model.add(Convolution2D(32, 3, 3, input_shape=(imageWidth, imageHeight, numberOfChannels), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(32, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit_generator(
    generator['training'],
    samples_per_epoch=numberOfTrainingSamples,
    nb_epoch=numberOfEpochs,
    validation_data=generator['validation'],
    nb_val_samples=numberOfValidationSamples)

# Save the weights of a trained model
model.save_weights('simple_model_weights.hdf5')

# Test the accuracy of the model
test = model.evaluate_generator(generator['testing'], val_samples=numberOfTestingSamples)
print("Loss: %.2f" % test[0])
print("Accuracy: %.2f%%" % (test[1] * 100))
