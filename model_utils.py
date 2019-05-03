import matplotlib.pyplot as plt
import numpy as np

from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img


def plot_history(history, acc, val_acc):
  # Accuracy
  plt.plot(history.history[acc])
  plt.plot(history.history[val_acc])
  plt.title('model accuracy')
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['train', 'test'], loc='upper left')
  plt.show()
  # Loss
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'test'], loc='upper left')
  plt.show()

def eval_model(model, weights, test_tensors, test_targets):
  model.load_weights(weights)
  predictions = [np.argmax(model.predict(np.expand_dims(tensor, axis=0))) for tensor in test_tensors]
  test_accuracy = 100*np.sum(np.array(predictions)==np.argmax(test_targets, axis=1))/len(predictions)
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
