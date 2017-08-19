from keras.preprocessing.image import ImageDataGenerator


# Define data augmentation strategies
def getGenerators(imageHeight, imageWidth, dataPaths):
    dataGeneratorTraining = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=360,
        fill_mode='nearest')
    generatorTraining = dataGeneratorTraining.flow_from_directory(
        dataPaths['training'],
        target_size=(imageWidth, imageHeight),
        batch_size=32,
        class_mode='binary')

    dataGeneratorValidation = ImageDataGenerator()
    generatorValidation = dataGeneratorValidation.flow_from_directory(
        dataPaths['validation'],
        target_size=(imageWidth, imageHeight),
        batch_size=32,
        class_mode='binary')

    dataGeneratorTesting = ImageDataGenerator()
    generatorTesting = dataGeneratorTesting.flow_from_directory(
        dataPaths['testing'],
        target_size=(imageHeight, imageWidth),
        batch_size=38,
        class_mode='binary')

    return {
        'training': generatorTraining,
        'validation': generatorValidation,
        'testing': generatorTesting
    }
