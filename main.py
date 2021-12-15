import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow import keras
from keras.layers import Dense, Dropout
from keras.models import Sequential

# %matplotlib inline


train_data = pd.read_csv('train.csv.zip')
test_data = pd.read_csv('test.csv.zip')

train_data = pd.concat([train_data, pd.get_dummies(train_data['color'])], axis=1)
train_data.drop('color', axis=1, inplace=True)

X = train_data.drop(['id', 'type'], axis=1)
Y = pd.get_dummies(train_data['type'])

# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=42)

model = Sequential()
model.add(Dense(64, input_shape=(X.shape[1],)))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
