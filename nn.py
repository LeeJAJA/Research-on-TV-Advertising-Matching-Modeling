import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf


data = pd.read_csv("sample.csv")
X = data.values[:,1:3]
Y = data.values[:,4:9]
X_df = pd.DataFrame(X)
Y_df = pd.DataFrame(Y)
X_df = pd.get_dummies(X_df)
Y_df = pd.get_dummies(Y_df)
print(X_df.head(5))
print(Y_df.head(5))