import matplotlib.pyplot as plt
import numpy as np

from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img


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

###
#      This method use grid search to tune a classifier.
#
#      Args:
#       clf (classifier): Classifier to tune.
#       parameters (dict): Clasiffier parameters.
#       X_train: (dataset): Features dataset to train the model.
#       y_train: (dataset): Targe feature dataset to train the model.
#       X_test: (dataset): Features dataset to test the model.
#       y_test: (dataset): Targe feature dataset to test the model.
#      Returns:
#       best_clf (classifier): Classifier with the best score.
#       default_score (float): Classifier score before being tuned.
#       tuned_score (float): Classifier score after being tuned.
#       cnf_matrix (float): Confusion matrix.
###
def tune_classifier(clf, parameters, X_train, X_test, y_train, y_test):

  from sklearn.metrics import make_scorer
  from sklearn.model_selection import GridSearchCV
  from sklearn.ensemble import ExtraTreesClassifier
  from sklearn.metrics import accuracy_score

  scorer = make_scorer(accuracy_score)

  grid_obj = GridSearchCV(clf, cv = 4, param_grid=parameters,  scoring=scorer, iid=False)
  grid_fit = grid_obj.fit(X_train, y_train)
  best_clf = grid_fit.best_estimator_
  predictions = (clf.fit(X_train, y_train)).predict(X_test)
  best_predictions = best_clf.predict(X_test)

  default_score = 100*accuracy_score(y_test, predictions)
  tuned_score = 100*accuracy_score(y_test, best_predictions)

  return best_clf, default_score, tuned_score
