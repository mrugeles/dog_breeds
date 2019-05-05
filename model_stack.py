import model_utils

from sklearn.datasets.samples_generator import make_blobs
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from keras.models import load_model
from keras.utils import to_categorical
from numpy import dstack


from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint


def fit_model(model, train_tensors, train_targets, valid_tensors, valid_targets, model_file, class_weights):
  model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
  epochs = 100

  checkpointer = ModelCheckpoint(filepath=model_file,
                               verbose=0, save_best_only=True)

  model.fit(train_tensors,
            train_targets,
            validation_data=(valid_tensors, valid_targets),
            #class_weight = class_weights,
            epochs=epochs,
            batch_size=32,
            callbacks=[checkpointer],
            verbose=0)
  return model

def fit_model_with_generators(model, epochs, model_file, train_generator, validation_generator, class_weights):
    from keras.optimizers import SGD
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
    batch_size = 16
    checkpointer = ModelCheckpoint(filepath=model_file,
                                 verbose=0, save_best_only=True)
    history = model.fit_generator(
            train_generator,
            class_weight = class_weights,
            steps_per_epoch=2000 // batch_size,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=800 // batch_size,
          callbacks=[checkpointer], verbose=0)
    return model

# load models from file
def load_all_models(n_models, folder):
	all_models = list()
	for i in range(n_models):
		# define filename for this ensemble
		filename = folder + '/model_' + str(i + 1) + '.h5'
		# load model from file
		model = load_model(filename)
		# add to list of members
		all_models.append(model)
		print('>loaded %s' % filename)
	return all_models

# create stacked model input dataset as outputs from the ensemble
def stacked_dataset(members, inputX):
  stackX = None
  for model in members:
    # make prediction
    yhat = model.predict(inputX, verbose=0)
    # stack predictions into [rows, members, probabilities]
    if stackX is None:
      stackX = yhat
    else:
      stackX = dstack((stackX, yhat))
  # flatten predictions to [rows, members x probabilities]
  stackX = stackX.reshape((stackX.shape[0], stackX.shape[1]*stackX.shape[2]))
  return stackX

# fit a model based on the outputs from the ensemble members
def fit_stacked_model(members, inputX, inputy):
  # create dataset using ensemble
  stackedX = stacked_dataset(members, inputX)
  # fit standalone model
  model = LogisticRegression(random_state = 9034)
  model.fit(stackedX, inputy)
  return model

def tune_stacked_model(clf, parameters, members, X_train, y_train, X_test, y_test):
    # create dataset using ensemble
  X_train = stacked_dataset(members, X_train)
  X_test = stacked_dataset(members, X_test)

  best_clf, default_score, tuned_score = model_utils.tune_classifier(clf, parameters, X_train, X_test, y_train, y_test)
  return best_clf, default_score, tuned_score

# make a prediction with the stacked model
def stacked_prediction(members, model, inputX):
	# create dataset using ensemble
	stackedX = stacked_dataset(members, inputX)
	# make a prediction
	yhat = model.predict(stackedX)
	return yhat
