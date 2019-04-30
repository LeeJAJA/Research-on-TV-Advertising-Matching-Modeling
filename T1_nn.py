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


<<<<<<< HEAD:T1_nn.py
<<<<<<< HEAD:T1_nn.py
data = pd.read_csv("T1_Artificial_Data.csv")
Y = data.values[:,7:8]
X = data.values[:,1:7]
=======
data = pd.read_csv("sample.csv")
X = data.values[:,1:3]
Y = data.values[:,4:9]
>>>>>>> parent of 19c82bd... update:nn.py
=======
data = pd.read_csv("sample.csv")
X = data.values[:,1:3]
Y = data.values[:,4:9]
>>>>>>> parent of 19c82bd... update:nn.py

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

model=Sequential([Dense(10, input_dim=X_train.shape[1]),
                  Activation('relu'),
                  Dense(24),
                  Activation('relu'),
                  Dense(50),
                  Activation('relu'),
                  Dense(50),
                  Activation('relu'),
                  Dense(30),
                  Activation('tanh'),
                  Dense(Y_train.shape[1]),
                  Activation('sigmoid')])

model.summary()

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss="binary_crossentropy", optimizer=sgd, metrics=["accuracy"])
model.fit(X_train,Y_train,epochs=10,batch_size=4)

loss, accuracy = model.evaluate(X_test,Y_test)
print("test loss:",loss)
print("test accuracy:", accuracy)
prob = model.predict(X_test.head(500))
#print(prob[:10,:])
idxs = np.argsort(-prob)[:10,:5]

<<<<<<< HEAD:T1_nn.py
<<<<<<< HEAD:T1_nn.py
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
=======
for (i, j) in enumerate(idxs):
    print(Y_mlb.classes_[j], ":",prob[i][j] * 100,"\n")
>>>>>>> parent of 19c82bd... update:nn.py
=======
for (i, j) in enumerate(idxs):
    print(Y_mlb.classes_[j], ":",prob[i][j] * 100,"\n")
>>>>>>> parent of 19c82bd... update:nn.py
