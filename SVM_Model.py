# -*- coding: utf-8 -*-
"""
Created on Tue May  1 01:12:29 2018

@author: HP
"""

import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

df=pd.read_csv('E:/8 semester/FYP-II/dataset/ILDP_train.csv')
df_train=df.iloc[:,:]
X_train=df_train.iloc[:,1:-1]
y_train=df_train.iloc[:,-1]

df=pd.read_csv('E:/8 semester/FYP-II/dataset/ILDP_testHealty.csv')
df_test=df.iloc[:,:]
X_test=df_test.iloc[:,1:-1]
y_test=df_test.iloc[:,-1]

'''
plt.style.use('ggplot') # Using ggplot for visualization
plt.title('Frequencies of Age')
plt.xlabel('Age')
plt.hist(df['Age'])
plt.show()

plt.title('Protiens vs Target')
plt.xlabel('Protiens')
plt.ylabel('Target')
plt.scatter(df['Total.Protiens'], df['data.Y'])
plt.show()
'''
model_pipeline=Pipeline([
                         ('SVM', SVC())])
model_pipeline.fit(X_train, y_train)
y_pred=model_pipeline.predict(X_test)
accuracy=accuracy_score(y_test, y_pred)
conf_matrix=confusion_matrix(y_test, y_pred)
print('Accuracy from SVM Model: ', accuracy)
print('Confusion Matrix from SVM Model: \n', conf_matrix)

#model_pkl = open('model_20180433.pkl', 'wb')
#pickle.dump(model_pipeline, model_pkl)
# Close the pickle instances
#model_pkl.close()
