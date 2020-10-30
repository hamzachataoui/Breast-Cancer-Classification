# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler


raw_data = pd.read_csv('data.csv')
dataset = raw_data.copy()
dataset['diagnosis']=dataset['diagnosis'].map({'B':0,'M':1})
raw_x = dataset.iloc[:, 2:31].values
raw_y = dataset.iloc[:, 1].values

# +
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(learning_rate_init = 0.001, max_iter = 200, momentum = 0.0)

scaler = StandardScaler()
x = scaler.fit(raw_x).transform(raw_x)

sampling = SMOTE(random_state=0)
x_train , y_train = sampling.fit_resample(x, raw_y)

#Fitting model with trainig data
mlp.fit(x_train, y_train)
# -

# Saving model to disk
pickle.dump(mlp, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))


f = scaler.transform([[ 1.097064   ,-2.073335    ,1.269934    ,0.984375    ,1.568466    ,3.283515,
  3.283515    ,2.532475    ,2.21751501,  2.25574689,  2.48973393, -0.56526506,
  2.83303087,  2.48757756, -0.21400165, -0.21400165,  0.72402616,  0.66081994,
  1.14875667  ,0.90708308  ,1.88668963 ,-1.35929347  ,2.30360062,  2.00123749,
  1.30768627  ,2.61666502  ,2.10952635  ,2.29607613  ,2.75062224]])

f




