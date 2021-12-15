import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import SGD
from tensorflow import keras
from keras.layers import Dense, Dropout
from keras.models import Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
# %matplotlib inline


train_data = pd.read_csv('train.csv.zip')
test_data = pd.read_csv('test.csv.zip')

train_data = pd.concat([train_data, pd.get_dummies(train_data['color'])], axis=1)
train_data.drop('color', axis=1, inplace=True)
test_data=pd.concat([test_data,pd.get_dummies(test_data['color'])],axis=1)
test_data.drop('color',axis=1,inplace=True)

X = train_data.drop(['id', 'type'], axis=1)
Y = pd.get_dummies(train_data['type'])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=42)

model = Sequential()
model.add(Dropout(0.01))
model.add(Dense(10, activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(1e-6)))
#model.add(Dense(10, activation='softmax', kernel_initializer='he_normal', kernel_regularizer=l2(1e-6)))
model.add(Dense(3, activation='softmax', kernel_initializer='he_normal', kernel_regularizer=l2(1e-6)))
#callbacks = [TensorBoard(), ModelCheckpoint('model-{epoch}.h5')]
opt = SGD(learning_rate=0.01, momentum=0.9)
#model.add(Dense(64, input_shape=(X.shape[1],)))
#model.add(Dense(32, activation='relu'))
#model.add(Dense(32, activation='relu'))
#model.add(Dense(32, activation='relu'))
#model.add(Dense(16, activation='relu'))
#model.add(Dense(8, activation='relu'))
#model.add(Dense(3, activation='softmax'))
model.compile(optimizer=opt, loss='categorical_crossentropy',metrics=['accuracy'])
train=model.fit(x=X_train,y=y_train,batch_size=16,epochs=15,verbose=2,validation_data=(X_test,y_test))

pred=model.predict(test_data.drop('id',axis=1))

pred_final=[np.argmax(i) for i in pred]
submission = pd.DataFrame({'id':test_data['id'], 'type':pred_final})
submission['type'].replace(to_replace=[0,1,2],value=['Ghost','Ghoul','Goblin'],inplace=True)
print(submission)