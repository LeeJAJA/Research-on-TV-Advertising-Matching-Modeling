import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense,Activation
from keras.optimizers import SGD
from sklearn.preprocessing import MultiLabelBinarizer
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt
import tensorflow as tf


data = pd.read_csv("sample.csv")
X = data.values[:,1:3]
Y = data.values[:,4:9]

X_mlb = MultiLabelBinarizer()
X_mlb.fit(X)
X = X_mlb.transform(X)
print(X_mlb.classes_)
#print(X[:5,:])

Y_mlb = MultiLabelBinarizer()
Y_mlb.fit(Y)
Y = Y_mlb.transform(Y)
print(Y_mlb.classes_)
#print(Y[:5,:])

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

for (i, j) in enumerate(idxs):
    print(Y_mlb.classes_[j], ":",prob[i][j] * 100,"\n")
