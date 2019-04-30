#import warnings
#warnings.filterwarnings("ignore")

import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

import keras
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout
from keras.layers import Add,Conv2D
from keras.optimizers import SGD
from sklearn.preprocessing import MultiLabelBinarizer
import pickle
from keras.utils import plot_model
import os
from keras.utils.vis_utils import plot_model

data = pd.read_csv("T3_Artificial_Data.csv")
X = data.values[1:,1:8]
Y_ = data.values[1:,8:25].astype(int)
Y_ = Y_.flatten()
X1_ = X[:,[0,2,3,4,5]]
X2_ = X[:,6]
X1 = [X1_[x,:] for x in range(0,len(X1_),24)]
X2 = [X2_[x:x+24] for x in range(0,len(X2_),24)]
Y =  [Y_[x:x+24*17] for x in range(0,len(Y_),24*17)]

X1_mlb = MultiLabelBinarizer()
X1_mlb.fit(X1)
X1 = X1_mlb.transform(X1)
print("length of X1_mlb.classes:",len(X1_mlb.classes_),end="\n\n")

X1_df = pd.DataFrame(X1)
X2_df = pd.DataFrame(X2).replace(['无', '个人用品', '互联网', '交通', '农业', '化妆品/浴室用品', '商业/服务性行业', '娱乐及休闲', '家居用品', '家用电器',
        '房地产', '数码/电脑/办公用品', '清洁用品', '药品', '衣着', '酒类', '金融/投资行业', '食品及饮料'],
                                 [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17])
Y_df = pd.DataFrame(Y)


print(X1_df.head(5))
print(X2_df.head(5))
print(Y_df.head(5))

X1_train, X1_test, X2_train, X2_test, Y_train, Y_test = train_test_split(X1_df, X2_df, Y_df, test_size = 0.33, random_state=42)
print("X1_train_shape:", X1_train.shape)
print("X2_train_shape:", X2_train.shape)
print("Y_train_shape:", Y_train.shape)
print("X1_test_shape:", X1_test.shape)
print("X2_test_shape:", X2_test.shape)
print("Y_test_shape:", Y_test.shape)

input1 = keras.layers.Input(shape=(X1_train.shape[1],))
x1 = keras.layers.Dense(X1_train.shape[1], activation='relu')(input1)
x1 = keras.layers.Dense(100, activation='relu')(x1)
x1 = keras.layers.Dropout(0.2)(x1)
x1 = keras.layers.Dense(Y_train.shape[1], activation='relu')(x1)

input2 = keras.layers.Input(shape=(X1_train.shape[1],1))
x2 = keras.layers.Conv1D(32, 3, padding='Same', activation='relu')(input2)
x2 = keras.layers.core.Flatten()(x2)
x2 = keras.layers.Dense(400, activation='relu')(x2)
x2 = keras.layers.Dense(Y_train.shape[1], activation='relu')(x2)

added = keras.layers.Add()([x1, x2])
out = keras.layers.Dense(Y_train.shape[1])(added)
model = keras.models.Model(inputs=[input1, input2], outputs=out)

model.summary()

model.compile(loss='mean_squared_error', optimizer='adam')
model.summary()
X2_train_r = np.zeros((10, 24, 1))
X2_train_r[:,:,0] =  X2_train.values[:,:]
X2_test_r = np.zeros((5, 24, 1))
X2_test_r[:,:,0] =  X2_test.values[:,:]

print(X2_train)
history = model.fit([X1_train, X2_train_r], Y_train, batch_size=2, nb_epoch=60, validation_data=([X1_test, X2_test_r], Y_test))

plot_model(model, to_file='./Figure/model3.png', show_shapes=True, show_layer_names=True)
model.save("./model/Task_3.hdf5")


# 绘制训练 & 验证的损失值
plt.plot(history.history['loss'],"g")
plt.plot(history.history['val_loss'],"r")
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='lower left')
plt.show()