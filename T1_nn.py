import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense,Activation
from keras.optimizers import SGD
from sklearn.preprocessing import MultiLabelBinarizer
import pickle
import os
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt
import tensorflow as tf


data = pd.read_csv("T1_Artificial_Data.csv")
Y = data.values[:,7:8]
X = data.values[:,1:7]

X_mlb = MultiLabelBinarizer()
X_mlb.fit(X)
X = X_mlb.transform(X)
print(X_mlb.classes_)
#print(X[:5,:])
f = open("./model/T1_X_mlb", "wb")
f.write(pickle.dumps(X_mlb))
f.close()

Y_mlb = MultiLabelBinarizer()
Y_mlb.fit(Y)
Y = Y_mlb.transform(Y)
print(Y_mlb.classes_)
#print(Y[:5,:])
f = open("./model/T1_Y_mlb", "wb")
f.write(pickle.dumps(Y_mlb))
f.close()

X_df = pd.DataFrame(X)
Y_df = pd.DataFrame(Y)

'''
#X_df = pd.get_dummies(X_df)
#Y_df = pd.get_dummies(Y_df)
'''

print(X_df.head(5))
print(Y_df.head(5))

X_train, X_test, Y_train, Y_test = train_test_split(X_df, Y_df, test_size = 0.33, random_state=42)
print("X_train_shape:", X_train.shape)
print("Y_train_shape", Y_train.shape)
print("X_test_shape", X_test.shape)
print("Y_test_shape", Y_test.shape)

model=Sequential([Dense(X_train.shape[1], input_dim=X_train.shape[1]),
                  Activation('relu'),
                  Dense(50),
                  Activation('relu'),
                  Dense(64),
                  Activation('relu'),
                  Dense(32),
                  Activation('relu'),
                  Dense(20),
                  Activation('relu'),
                  Dense(Y_train.shape[1]),
                  Activation('softmax')])

model.summary()

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss="binary_crossentropy", optimizer=sgd, metrics=["accuracy"])
model.fit(X_train,Y_train,epochs=30,batch_size=2)

loss, accuracy = model.evaluate(X_test,Y_test)
print("test loss:",loss)
print("test accuracy:", accuracy)
prob = model.predict(X_test.head(10))
#print(prob[:10,:])
idxs = np.argsort(-prob)[:,:3]

X_test = np.array(X_test)
#print(X_test.shape[1])
#print(len(X_mlb.classes_))
np.set_printoptions(suppress=True)
for i in range(idxs.shape[0]):
    print(X_mlb.inverse_transform(X_test)[i])
    for j in range(idxs.shape[1]):
        print(Y_mlb.classes_[idxs[i][j]], ":",np.around(prob[i][idxs[i][j]]*100,4),end="% ")
    print("\n")

model.save("./model/Task_1.hdf5")