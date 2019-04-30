import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from keras.models import Sequential, load_model
from keras.layers import Dense,Activation
from keras.optimizers import SGD
from sklearn.preprocessing import MultiLabelBinarizer
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle


data = pd.read_csv("T1_Artificial_Data.csv")
Y = data.values[:,7:8]
X = data.values[:,1:7]


#load model
X_mlb = pickle.loads(open("./model/T1_X_mlb", "rb").read())
Y_mlb = pickle.loads(open("./model/T1_Y_mlb", "rb").read())
model = load_model("./model/Task_1.hdf5")
model.summary()

print(Y_mlb.classes_[np.argmax(model.predict(X_mlb.transform(X))[1])])
X[1][1]='0:00-2:00'
print(Y_mlb.classes_[np.argmax(model.predict(X_mlb.transform(X))[1])])



print("X_train_shape:", X_train.shape)
print("Y_train_shape", Y_train.shape)
print("X_test_shape", X_test.shape)
print("Y_test_shape", Y_test.shape)


