import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os.path
import random

from sklearn.datasets import load_files
from sklearn.utils import class_weight
from keras.utils import np_utils
from keras.preprocessing import image
from glob import glob
from time import time
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from PIL import ImageFile
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

def load_sparsed_dataset(path):
    data = load_files(path)
    return np.array(data['filenames']), data['target']

def load_categorical_dataset(path):
    data = load_files(path)
    return np.array(data['filenames']), np_utils.to_categorical(np.array(data['target']), 133)

def load_datasets(train_path, valid_path, test_path, is_categorical):
  train_files, train_targets = load_categorical_dataset(train_path) if load_categorical_dataset == True else load_sparsed_dataset(train_path)
  valid_files, valid_targets = load_categorical_dataset(valid_path) if load_categorical_dataset == True else load_sparsed_dataset(valid_path)
  test_files, test_targets = load_categorical_dataset(test_path) if load_categorical_dataset == True else load_sparsed_dataset(test_path)

  dog_names = [item[18:-1] for item in sorted(glob(train_path + "/*/"))]

  print('There are %d total dog categories.' % len(dog_names))
  print('There are %s total dog images.\n' % len(np.hstack([train_files, valid_files, test_files])))
  print('There are %d training dog images.' % len(train_files))
  print('There are %d validation dog images.' % len(valid_files))
  print('There are %d test dog images.'% len(test_files))
  return train_files, train_targets, valid_files, valid_targets, test_files, test_targets, dog_names

def load_targets_dataset(path):
    data = load_files(path)
    return np_utils.to_categorical(np.array(data['target']), 133)

def load_targets_datasets(train_path, valid_path, test_path):
    train_targets = load_targets_dataset(train_path)
    valid_targets = load_targets_dataset(valid_path)
    test_files, test_targets = load_dataset(test_path)

    return train_targets, valid_targets, test_targets

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

def paths_to_tensors(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in img_paths]
    return np.vstack(list_of_tensors)


def show_target_distribution(path):
  path_from = len(path) + 1
  path_to = path_from + 3
  name_position = path_to + 1
  folders = [(int(path[path_from:path_to]), path[name_position:-1], path, len([f for f in os.listdir(path)if os.path.isfile(os.path.join(path, f))])) for path in sorted(glob(path + '/*/'))]
  folders = pd.DataFrame(folders, columns = ['target', 'breed', 'folder', 'count'])

  folders[['breed', 'count']].plot.bar(x = 'breed', figsize = (20, 5))

def augment_images(path):
    datagen = ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')

    for path in sorted(glob(path)):
        files = os.listdir(path)
        files = list(map(lambda file: path + file, files))
        tensors = paths_to_tensors(files)
        total_files = len(files)
        if(total_files < 100):
          n_files = 100 - total_files
          i = 0
          for batch in datagen.flow(tensors, batch_size=1, save_to_dir=path, save_prefix='augmented', save_format='jpeg'):
              i += 1
              if i > n_files:
                  break

def get_class_weights(train_targets):
    targets = np.unique(train_targets)
    return class_weight.compute_class_weight('balanced', targets, train_targets)
