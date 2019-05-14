import matplotlib.pyplot as plt
import numpy as np

from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint



def plot_history(history, acc, val_acc):
    """ Plot training history
    Parameters
    ----------
    history: Object
        Meta model
    acc: string
        Key for accuracy field.
    val_acc: string
        Key for validation accuracy field.
    """
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    # Accuracy
    axs[0].plot(history.history[acc])
    axs[0].plot(history.history[val_acc])
    axs[0].set_title('model accuracy')
    axs[0].set_ylabel('accuracy')
    axs[0].set_xlabel('epoch')
    axs[0].legend(['train', 'test'], loc='upper left')

    # Loss
    axs[1].plot(history.history['loss'])
    axs[1].plot(history.history['val_loss'])
    axs[1].set_title('model loss')
    axs[1].set_ylabel('loss')
    axs[1].set_xlabel('epoch')
    axs[1].legend(['train', 'test'], loc='upper left')
    plt.show()

def eval_model(model, weights, test_tensors, test_targets, is_categorical = True):
    """ Eval model's accuracy
    Parameters
    ----------
    model: Sequential
        Model to evaluate.
    weights: object
        Model's weights.
    test_tensors: array
        Records for test predicions.
    test_targets: array
        Targets for test predicions.
    is_categorical: boolean
        How to treat targets. True for categorical, False for integers.

    Returns
    -------
    test_accuracy: floar
        Model's accuracy.
    """
    test_targets = np.argmax(test_targets, axis=1) if is_categorical == True else test_targets

    model.load_weights(weights)
    predictions = [np.argmax(model.predict(np.expand_dims(tensor, axis=0))) for tensor in test_tensors]
    test_accuracy = 100*np.sum(np.array(predictions)==test_targets)/len(predictions)
    return test_accuracy


def get_generators(batch_size, train_path, valid_path):
    """ Create generators for data augmentation.
    Parameters
    ----------
    batch_size: int
        Batch size.
    train_path: string.
        Path to training images.
    valid_path: string
        Path to validation images.

    Returns
    -------
    train_generator: DirectoryIterator
        DirectoryIterator for augmented training data.
    validation_generator: DirectoryIterator
        DirectoryIterator for augmented validation data.
    """
    batch_size = batch_size
    train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
            train_path,
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(
            valid_path,
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode='categorical')
    return train_generator, validation_generator

from keras.callbacks import ModelCheckpoint

def train_bottleneck(bottleneck_file, weights_file, batch_size, bottleneck_model, train_targets, valid_targets, test_targets, class_weights):
    """ Train model from bottleneck features.
    Parameters
    ----------
    bottleneck_file: file.
        File with bottleneck features.
    weights_file: string.
        Path to model's weights.
    batch_size: int
        Batch size.
    bottleneck_model: Sequential
        Model to train.
    train_targets: array
        Train targets.
    valid_targets: array
        Valid targets.
    test_targets: array
        Test targets.
    class_weights: array
        Balanced class list.

    Returns
    -------
    accuracy: float
      Model's accuracy.
    """
    bottleneck_features = np.load(bottleneck_file)

    train = bottleneck_features['train']
    valid = bottleneck_features['valid']
    test = bottleneck_features['test']

    checkpointer = ModelCheckpoint(filepath=weights_file,
                               verbose=1, save_best_only=True)

    bottleneck_history = bottleneck_model.fit(
          train,
          train_targets,
          class_weight = class_weights,
          validation_data=(valid, valid_targets),
          epochs=batch_size,
          batch_size=batch_size,
          callbacks=[checkpointer],
          verbose=1)

    plot_history(bottleneck_history)
    eval_model(bottleneck_model, weights_file, test, test_targets)


def get_class_weights():
    """ Get a Dictionary of balanced class weights.

    Returns
    -------
    dict_class_weights: Dictionary
       Dictionary of balanced class weights
    """
    from sklearn.utils import class_weight
    targets = np.unique(train_targets)

    class_weights = class_weight.compute_class_weight('balanced', targets, train_targets)

    dict_class_weights = dict(zip(targets, class_weights))

def get_bottleneck_features(bottleneck_file):
    """ Get bottleneck features from a given bottleneck file.
    Parameters
    ----------
    bottleneck_file: String
        Path to bottleneck file.

    Returns
    -------
    train: array
       Training bottleneck features
    valid: array
       Validation bottleneck features
    test: array
       Test bottleneck features
    """
    bottleneck_features = np.load(bottleneck_file)
    train = bottleneck_features['train']
    valid = bottleneck_features['valid']
    test = bottleneck_features['test']
    return train, valid, test


def get_model_Xception(train_features, train_targets, shape):
    """ Get arquitecture for Xception model.
    Parameters
    ----------
    train_features: array
        Training features.
    train_targets: array
        Target features.
    shape: tuple
        Input dimensions.

    Returns
    -------
    model: Sequential
       Model graph
    """

    from keras.callbacks import ModelCheckpoint
    model = Sequential()

    model.add(Conv2D(filters=2048, kernel_size=2, padding='same', activation='relu', kernel_initializer='random_uniform', input_shape=shape))
    model.add(Conv2D(filters=2048, kernel_size=2, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))

    model.add(Dropout(0.05))
    model.add(GlobalAveragePooling2D(input_shape=train_targets.shape[1:]))
    model.add(Dense(133, activation='softmax'))

    return model


def train_model(model, epochs, train_features, valid_features, train_targets, valid_targets, weights_file):
    """ Get arquitecture for Xception model.
    Parameters
    ----------
    train_features: array
        Training features.
    train_targets: array
        Target features.
    valid_features: array
        Validation features.
    valid_targets: array
        Validation targets.
    weights_file: string.
        Path to weights file.

    Returns
    -------
    model: Sequential
        Trained model
    history: Object
        Training history
    """
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

    checkpointer = ModelCheckpoint(filepath=weights_file, verbose=1, save_best_only=True)

    history = model.fit(
      train_features,
      train_targets,
      validation_data=(valid_features, valid_targets),
      epochs=epochs,
      batch_size=16,
      callbacks=[checkpointer],
      verbose=1)
    return model, history

def build_bottleneck_model(bottleneck, n, epochs, train_targets, valid_targets):
    """ Creates and trains a bottleneck model.
    Parameters
    ----------
    bottleneck: string
        Path to bottleneck file.
    n: int
        Number of models to create
    epochs: int
        Epochs
    train_targets: array
        Target features.
    valid_targets: array
        Validation targets.

    Returns
    -------
    model: Sequential
        Trained model
    history: Object
        Training history
    test_features: array
        bottleneck test features
    """
    bottleneck_file = bottleneck + '.npz'
    weights_file = 'weights.'+bottleneck+'_'+str(n)+'.hdf5'
    train_features, valid_features, test_features = get_bottleneck_features(bottleneck_file)

    input_shape = (train_features.shape[1], train_features.shape[2], train_features.shape[3])
    model = get_model_Xception(train_features, train_targets, input_shape)
    model, history = train_model(model, epochs, train_features, valid_features, train_targets, valid_targets, weights_file)
    return model, history, test_features
