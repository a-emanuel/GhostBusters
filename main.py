import numpy as np
import pandas as pd
from tensorflow.keras.optimizers import SGD
from keras.layers import Dense, Dropout
from keras.models import Sequential
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# %matplotlib inline
def processData(data):
    data = pd.concat([data, pd.get_dummies(data['color'])], axis=1)
    data.drop('color', axis=1, inplace=True)
    return data


def trainModel(train_data):
    # Process training data
    train_data = processData(train_data)

    X = train_data.drop(['id', 'type'], axis=1)
    Y = pd.get_dummies(train_data['type'])

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=42)

    # Create and train the network
    model = Sequential()
    model.add(Dropout(0.01))
    model.add(Dense(10, activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(1e-6)))
    model.add(Dense(3, activation='softmax', kernel_initializer='he_normal', kernel_regularizer=l2(1e-6)))
    opt = SGD(learning_rate=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    train = model.fit(x=X_train, y=y_train, batch_size=16, epochs=15, verbose=2, validation_data=(X_test, y_test))

    return model, train


def classifyData(model, test_data):
    test_data = processData(test_data)

    pred = model.predict(test_data.drop('id', axis=1))

    pred_final = [np.argmax(i) for i in pred]
    submission = pd.DataFrame({'id': test_data['id'], 'type': pred_final})
    #submission = pd.concat([test_data, pred_final])
    submission['type'].replace(to_replace=[0, 1, 2], value=['Ghost', 'Ghoul', 'Goblin'], inplace=True)
    return submission


def ploting(train):
    plt.plot(train.history['val_accuracy'], label='Validation accuracy')
    plt.plot(train.history['accuracy'], color='red', marker='.', linestyle='--', label='Training accuracy')
    plt.legend()

    plt.figure()
    plt.plot(train.history['val_loss'], label='Validation loss')
    plt.plot(train.history['loss'], color='red', marker='.', linestyle='--', label='Training loss')
    plt.legend()

    plt.show()

def main():
    train_data = pd.read_csv('train.csv.zip')
    test_data = pd.read_csv('test.csv.zip')

    model, train = trainModel(train_data)
    submission = classifyData(model, test_data)
    print(submission)
    print(submission['type'].value_counts())

    ploting(train)




main()
