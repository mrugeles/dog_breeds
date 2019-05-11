import matplotlib.pyplot as plt
import numpy as np

from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint


def plot_history(history, acc, val_acc):
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
  test_targets = np.argmax(test_targets, axis=1) if is_categorical == True else test_targets

  model.load_weights(weights)
  predictions = [np.argmax(model.predict(np.expand_dims(tensor, axis=0))) for tensor in test_tensors]
  test_accuracy = 100*np.sum(np.array(predictions)==test_targets)/len(predictions)
  return test_accuracy


def get_generators(batch_size, train_path, valid_path):
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
    from sklearn.utils import class_weight
    targets = np.unique(train_targets)

    class_weights = class_weight.compute_class_weight('balanced', targets, train_targets)

    dict_class_weights = dict(zip(targets, class_weights))

def get_bottleneck_features(bottleneck_file):

  bottleneck_features = np.load(bottleneck_file)
  train = bottleneck_features['train']
  valid = bottleneck_features['valid']
  test = bottleneck_features['test']
  return train, valid, test


def get_model_Xception(train_features, train_targets, shape):


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
  bottleneck_file = bottleneck + '.npz'
  weights_file = 'weights.'+bottleneck+'_'+str(n)+'.hdf5'
  train_features, valid_features, test_features = get_bottleneck_features(bottleneck_file)

  input_shape = (train_features.shape[1], train_features.shape[2], train_features.shape[3])
  model = get_model_Xception(train_features, train_targets, input_shape)
  model, history = train_model(model, epochs, train_features, valid_features, train_targets, valid_targets, weights_file)
  return model, history, test_features
