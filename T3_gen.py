import warnings
warnings.filterwarnings("ignore")

import random
import numpy as np
import pandas as pd
from keras.models import load_model
import pickle

dict = {0:'0:00-2:00',
        1:'2:00-4:00',
        2:'4:00-6:00',
        3:'6:00-8:00',
        4:'8:00-10:00',
        5:'10:00-12:00',
        6:'12:00-14:00',
        7:'14:00-16:00',
        8:'16:00-18:00',
        9:'18:00-20:00',
        10:'20:00-22:00',
        11:'22:00-0:00'}

data = pd.read_csv("T1_Artificial_Data.csv")
Y = data.values[:,7:8]
X = data.values[:,1:8]

#load model
X_mlb = pickle.loads(open("./model/T1_X_mlb", "rb").read())
Y_mlb = pickle.loads(open("./model/T1_Y_mlb", "rb").read())
model = load_model("./model/Task_1.hdf5")
model.summary()

df = pd.DataFrame()

for line in range(30):
    print("*", Y_mlb.classes_[np.argmax(model.predict(X_mlb.transform(X))[line])])
    for i in range(12):
        X[line][1]=dict[i]
        #print(X[line],end=" ")
        if random.random()>0.6:
            if random.random()>=(abs(i-5.5)/10)+0.15:
                #print(Y_mlb.classes_[np.argmax(model.predict(X_mlb.transform(X))[line])])
                X[line,6]=Y_mlb.classes_[np.argmax(model.predict(X_mlb.transform(X))[line])]
                df = df.append(pd.Series(X[line]), ignore_index=True)
                print(X[line])
            else:
                #print(Y_mlb.classes_[int(random.uniform(0,len(Y_mlb.classes_)+1))])
                X[line, 6]=Y_mlb.classes_[int(random.uniform(0,len(Y_mlb.classes_)))]
                df = df.append(pd.Series(X[line]),ignore_index=True)
                print(X[line])
        else:
            X[line, 6] ="无"
            print(X[line])
            df = df.append(pd.Series(X[line]), ignore_index=True)

df2=pd.DataFrame(columns=list(Y_mlb.classes_))
df.columns=['年龄段','投放时间','性别','教育水平','所在行业','消费水平','广告类型']
#print(df)
df.to_csv("T3_sample.csv")
df2.to_csv("T3_sample_assist.csv")

