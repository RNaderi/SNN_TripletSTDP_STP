'''
Created by A.Rezaei
'''
"''''''''''''''''''''''''''''''''''''''''''''' Imports '''''''''''''''''''''''''''''''''''''''''''''''''"
import numpy as np
import keras
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, SimpleRNN, Input
from tensorflow.keras import layers, models, losses, Model
from tensorflow.keras.utils import to_categorical
from keras import backend as k
import matplotlib.pyplot as plt
import pandas as pd
import time
import os
from tqdm import tqdm

"''''''''''''''''''''''''''''''''''''''''''''' Functions '''''''''''''''''''''''''''''''''''''''''''''''''"

def prepare_data(dataset_dir, dataset_name):
    
    x_train = np.load(r'{}\{}\training.npy'.format(dataset_dir, dataset_name))
    y_train = np.load(r'{}\{}\trainingLabels.npy'.format(dataset_dir, dataset_name))

    x_test = np.load(r'{}\{}\testing.npy'.format(dataset_dir, dataset_name))
    y_test = np.load(r'{}\{}\testingLabels.npy'.format(dataset_dir, dataset_name))

    num_category = len(np.unique(y_train))
    img_rows, img_cols = x_train.shape[1], x_train.shape[2]
    if k.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols)
        input_shape = (img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols)
        input_shape = (img_rows, img_cols)
        
    # convert to one-hot vector
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    
    #more reshaping
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    
    return x_train, x_test, y_train, y_test, num_category, input_shape

def create_model(num_category, input_shape, units=256, dropout=0.2):
    

    model = Sequential()
    
    model.add(Input(input_shape))
    
    model.add(SimpleRNN(units=units,
                        dropout=dropout))
    
    model.add(Dense(num_category))
    model.add(Activation('softmax'))
    
    return model

def train_test_model(x_train, x_test, y_train, y_test,
                     num_category, input_shape, dataset_name,
                     epochs, batch_size=128, lr = 0.001):
    
    model = create_model(num_category, input_shape)
        
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(lr),
                  metrics=['accuracy'])

    start_time_train = time.time()
    history = model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=0)
    end_time_train = time.time()
    
    start_time_test = time.time()
    score = model.evaluate(x_test, y_test, verbose=0)
    end_time_test = time.time()
    
    results = pd.DataFrame({'Dataset': dataset_name, 'Epoch':epochs,
                                  'Train ACC': history.history['accuracy'].pop(),
                                  'Train loss': history.history['loss'].pop(),
                                  'Train time (sec)': end_time_train - start_time_train,
                                  'Test ACC': score[1], 'Test loss':score[0],
                                  'Test time (sec)':end_time_test - start_time_test,
                                  'Number of parameters': model.count_params()}, index=[0])
    
    del model
    return results

"''''''''''''''''''''''''''''''''''''''''''''' Main '''''''''''''''''''''''''''''''''''''''''''''''''"

#disable warnings
tf.get_logger().setLevel('INFO')

main_dir = r'D:\Workspace\Snn'
save_dir = r'{}\Revision codes and data'.format(main_dir)
dataset_dir = r'{}\Revision codes and data\data\ann'.format(main_dir)

dataset_list = os.listdir(dataset_dir)
datasets = ['mnist-alphabet',
            'mnist-digit', 
            'noisy-mnist-alphabet', 'noisy-mnist-digit', 
            'shd'
            ]
results = pd.DataFrame()

for dataset_name in tqdm(datasets):
    
    x_train, x_test, y_train, y_test, num_category, input_shape = prepare_data(dataset_dir, dataset_name)
    
    "''''''''''''''''''''''''' 1 Epoch''''''''''''''''''''''''''''''''''"
    
    results_1 = train_test_model(x_train, x_test, y_train, y_test,
                                 num_category, input_shape, dataset_name, 1)
    
    "''''''''''''''''''''''''' 5 Epochs ''''''''''''''''''''''''''''''''''"
    results_5 = train_test_model(x_train, x_test, y_train, y_test,
                                 num_category, input_shape, dataset_name, 5)
    
    "''''''''''''''''''''''''' 15 Epochs ''''''''''''''''''''''''''''''''''"
    results_15 = train_test_model(x_train, x_test, y_train, y_test,
                                  num_category, input_shape, dataset_name, 15)
    
    results = pd.concat([results, results_1, results_5, results_15])

# results.to_csv(r'{}\RNN_results.csv'.format(save_dir), index=False)