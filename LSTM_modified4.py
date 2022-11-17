import tensorflow as tf
from tensorflow import keras

import os

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

print(f"Tensorflow version : {tf.version.VERSION}")


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, BatchNormalization, Dropout
from tensorflow.keras import optimizers


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


tf.random.set_seed(13)
tf.debugging.set_log_device_placement(False)


import tensorflow as tf
from tensorflow import keras

import os

from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, BatchNormalization, Dropout
from tensorflow.keras import optimizers

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

train_filename = './cicids2017_train.csv'
test_filename = './cicids2017_test.csv'


import pandas as pd

train_data = pd.read_csv(train_filename)

train_data

test_data = pd.read_csv(test_filename)
test_data



X_train = train_data.loc[:, train_data.columns != 'classification']
y_train = train_data['classification']

X_train_normalized = X_train

X_test = test_data.loc[:, test_data.columns != 'classification']
y_test = test_data['classification']
X_test_normalized = X_test

#function for normalization
def standarize(numeric_dataset):
    # standardized_value = (x - mean)/ standard_deviation

    # calculate mean and standard deviation per numeric columns
    mean_val = numeric_dataset.mean(axis=0)
    std_dev_val = numeric_dataset.std(axis=0)

    # standardization
    matrix_standardized = (numeric_dataset - mean_val) / std_dev_val

    return matrix_standardized

#normalize train/test data
for i in range(len(X_train.columns)):
    train_data_columns = X_train.iloc[:, i]
    normalized_train_data = standarize(train_data_columns)
    X_train_normalized.iloc[:, i] = normalized_train_data

    test_data_columns = X_test.iloc[:, i]
    normalized_test_data = standarize(test_data_columns)
    X_test_normalized.iloc[:, i] = normalized_test_data


def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
        print(result[feature_name].min())
    return result



X_train_normalized0 = normalize(X_train_normalized)
X_test_normalized0 = normalize(X_test_normalized)

# get max num
max_x_train = X_train_normalized0.max(numeric_only=True).max()  # np.max(X_train)
max_x_test = X_test_normalized0.max(numeric_only=True).max()  # np.max(X_test)

print(max_x_train)
print(max_x_test)
if max_x_train > max_x_test:
    max_num = int(max_x_train) + 1
else:
    max_num = int(max_x_test) + 1



# define and fit the model
def get_model(input_length, X_train, y_train):
    # define model
    model = Sequential()
    model.add(keras.layers.Embedding(input_dim=max_num, output_dim=50, input_length=input_length))
    model.add(keras.layers.LSTM(50, return_sequences=True, input_length=input_length))
    model.add(keras.layers.Dropout(0.7))
    model.add(keras.layers.LSTM(50, return_sequences=False))
    model.add(keras.layers.Dropout(0.7))
    model.add(keras.layers.Dense(1))

    # compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit model
    model.fit(X_train, y_train, epochs=100, verbose=0)
    return model



input_length = X_train.shape[1]
print(X_train.shape, y_train.shape)
model = get_model(input_length, X_train_normalized0, y_train)
model.summary()


from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, \
    confusion_matrix

# predict probabilities for test set
y_test_pred = model.predict(X_test, verbose=0)

Y_Testshaped = y_test.values

print('F1 : ' + str(f1_score(Y_Testshaped, y_test_pred.round(), average=None)))



from sklearn.metrics import f1_score

f1 = f1_score(y_test.values, y_test_pred.round(), average=None)
print('f1 score :', f1)

