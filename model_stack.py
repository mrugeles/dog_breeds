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
    """ Generic function to fit a model with tensors created from images loaded from file system.

    Parameters
    ----------
    model: Sequential
        Model to fit.
    train_tensors: array
        Training features.
    train_targets: array
        Training targets.
    valid_tensors: array
        Validation features.
    valid_targets: array
        Validation targes.
    model_file: string
        Path to store the best model.
    class_weights: array
        Array of balanced class weights.

    Returns
    -------
    model: Sequential
        Trained model.
    """
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
    epochs = 100

    checkpointer = ModelCheckpoint(filepath=model_file,
                               verbose=0, save_best_only=True)

    model.fit(train_tensors,
            train_targets,
            validation_data=(valid_tensors, valid_targets),
            epochs=epochs,
            batch_size=32,
            callbacks=[checkpointer],
            verbose=0)
    return model

def fit_model_with_generators(model, epochs, model_file, train_generator, validation_generator, class_weights):
    """ Generic function to fit a model with image generatos
    Parameters
    ----------
    model: Sequential
        Model to fit.
    train_tensors: array
        Training features.
    train_targets: array
        Training targets.
    valid_tensors: array
        Validation features.
    valid_targets: array
        Validation targes.
    model_file: string
        Path to store the best model.
    class_weights: array
        Array of balanced class weights.

    Returns
    -------
    model: Sequential
        Trained model.
    """
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

def load_all_models(n_models, folder):
    """ Load models from file system
    Parameters
    ----------
    n_models: int
        Number of models to load.
    folder: string
        Path to models.

    Returns
    -------
    all_models: array
        Loaded models.
    """
    all_models = list()
    for i in range(n_models):
    	filename = folder + '/model_' + str(i + 1) + '.h5'
    	model = load_model(filename)
    	all_models.append(model)
    	print('>loaded %s' % filename)
    return all_models

def stacked_dataset(members, inputX):
    """ Stack predictions from models to create a new dataset
    Parameters
    ----------
    members: array
        Model list.
    inputX: array
        Feature list to predict.

    Returns
    -------
    stackX: array
        dataset created from predictions.
    """
    stackX = None
    for model in members:
        yhat = model.predict(inputX, verbose=0)
    if stackX is None:
        stackX = yhat
    else:
        stackX = dstack((stackX, yhat))
    stackX = stackX.reshape((stackX.shape[0], stackX.shape[1]*stackX.shape[2]))
    return stackX

def fit_stacked_model(members, inputX, inputy):
    """ Fit meta model from stacked dataset.
    Parameters
    ----------
    members: array
        Model list.
    inputX: array
        Feature list to predict.
    inputy: array
        Targets.
    Returns
    -------
    model: Model
        Trained meta model.
    """
    stackedX = stacked_dataset(members, inputX)
    model = LogisticRegression(random_state = 9034)
    model.fit(stackedX, inputy)
    return model

def tune_stacked_model(clf, parameters, members, X_train, y_train, X_test, y_test):
    """ Tune meta model.
    Parameters
    ----------
    clf: Model
        Meta model to tune
    parameters: Dictionary
        Hyperparameters for tuning the meta model
    members: array
        Model list.
    X_train: array
        Training features.
    y_train: array
        Training targets.
    X_test: array
        Test features.
    y_test: array
        Test targets.
    Returns
    -------
    best_clf: Model
        Model with best score.
    default_score: float
        Score before tuning the meta model.
    tuned_score: Model
        Score after tuning the meta model.
    """
    X_train = stacked_dataset(members, X_train)
    X_test = stacked_dataset(members, X_test)

    best_clf, default_score, tuned_score = model_utils.tune_classifier(clf, parameters, X_train, X_test, y_train, y_test)
    return best_clf, default_score, tuned_score

def stacked_prediction(members, model, inputX):
    """ Predict from meta model
    Parameters
    ----------
    model: Model
        Meta model
    members: array
        Model list.
    inputX: array
        Feature list to run predicions.
    Returns
    -------
    yhat: array
        Meta model's predicions.
    """
    stackedX = stacked_dataset(members, inputX)
    yhat = model.predict(stackedX)
    return yhat
