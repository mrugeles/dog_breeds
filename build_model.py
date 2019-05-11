import numpy as np
import pandas as pd
import data_utils
import model_utils
import time
import os
import shutil

BATCH_SIZE = 16
TRAIN_PATH = 'dogImages/train'
VALID_PATH = 'dogImages/valid'
TEST_PATH = 'dogImages/test'

APP_PATH = 'app/'

MODEL_FILE = 'model.hdf5'
bottlenecks = ['DogXceptionData']


def main():
    print('Loading dataset...')
    train_files, train_targets = data_utils.load_categorical_dataset(TRAIN_PATH)
    valid_files, valid_targets = data_utils.load_categorical_dataset(VALID_PATH)
    test_files, test_targets = data_utils.load_categorical_dataset(TEST_PATH)

    bottleneck = 'DogXceptionData'

    models = pd.DataFrame(columns = ['Model', 'Accuracy'])
    for i in range(6):
        start_time = time.time()
        n = i + 1
        print('Training Sub model %d' % (n))
        model, history, test_features =  model_utils.build_bottleneck_model(bottleneck, n ,20, train_targets, valid_targets)
        accuracy = model_utils.eval_model(model, 'weights.'+bottleneck+'_'+str(n)+'.hdf5', test_features, test_targets)
        models = models.append({'Model': 'weights.'+bottleneck+'_'+str(n)+'.hdf5', 'Accuracy':accuracy}, ignore_index = True)
        end_time = time.time() - start_time
        print('Sub model %s %d accuracy: %.4f%%' % (bottleneck, n, accuracy))
        print('Training time: %.4f%% minutes\n' % (end_time / 60))

    best_model = models.sort_values(by=['Accuracy'], ascending = False)
    best_model = best_model.head(1)[['Model']].values[0][0]
    shutil.move(best_model, APP_PATH + MODEL_FILE)

if __name__ == '__main__':
    main()
