#import warnings
#warnings.filterwarnings("ignore")

import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout
from keras.optimizers import SGD
from sklearn.preprocessing import MultiLabelBinarizer
import pickle
import itertools
from keras.utils import plot_model
import os
from keras.utils.vis_utils import plot_model
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
                  Dropout(0.2),
                  Dense(64),
                  Activation('relu'),
                  Dropout(0.2),
                  Dense(128),
                  Activation('relu'),
                  Dense(Y_train.shape[1]),
                  Activation('softmax')])

model.summary()

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss="binary_crossentropy", optimizer=sgd, metrics=["accuracy"])
history = model.fit(X_train,Y_train,epochs=30,batch_size=5,validation_split=0.25)

loss, accuracy = model.evaluate(X_test,Y_test)

pred_y = model.predict(X_train)
pred_label = np.argsort(-pred_y)[:,:1]
true_label = np.argsort(-Y_train)[:][0]
print(Y_test.head(5))
print(true_label.head(5))

print("test loss:",loss)
print("test accuracy:", accuracy)
prob = model.predict(X_test.head(10))
#print(prob[:10,:]
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

plot_model(model, to_file='./Figure/model.png')
model.save("./model/Task_1.hdf5")

# 绘制训练 & 验证的准确率值
plt.plot(history.history['acc'],"g")
plt.plot(history.history['val_acc'],"r")
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# 绘制训练 & 验证的损失值
plt.plot(history.history['loss'],"g")
plt.plot(history.history['val_loss'],"r")
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='lower left')
plt.show()


confusion_mat = confusion_matrix(true_label, pred_label)
def plot_sonfusion_matrix(cm, classes, normalize=False, title='Confusion matrix',cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]
    thresh = cm.max()/2.0
    for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j,i,cm[i,j], horizontalalignment='center',color='white' if cm[i,j] > thresh else 'black')
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predict label')
    plt.show()

plot_sonfusion_matrix(confusion_mat, classes = range(10))
