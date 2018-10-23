#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 22:44:05 2018

@author: danish
"""

import pandas as pd
df = pd.read_csv("/home/ubuntu/miRNA_matrix.csv")
df.pop('file_id')
df=df.iloc[0:500,:]

from sklearn.preprocessing import StandardScaler
# Separating out the features
dataset = df.values
x = dataset[:,0:1881].astype(float)
y = dataset[:,1881]
# Standardizing the features
x = StandardScaler().fit_transform(x)
val=2
from sklearn.decomposition import PCA
pca = PCA(n_components=val)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])

finalDf = pd.concat([principalDf, df[['label']]], axis = 1)

import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline


dataset = finalDf.values
X = dataset[:,0:val].astype(float)
Y = dataset[:,val]

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)

# define baseline model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(8, input_dim=val, activation='relu'))
	model.add(Dense(36, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

estimator = KerasClassifier(build_fn=baseline_model, epochs=10, batch_size=5, verbose=1)
kfold = KFold(n_splits=10, shuffle=True, random_state=42)
results = cross_val_score(estimator, X, dummy_y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))



fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = list(numpy.unique(df.label))
for target in targets:
    indicesToKeep = finalDf['label'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , s = 50)
ax.legend(targets)
ax.grid()



import matplotlib.pyplot as plt
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)

targets = numpy.unique(df.label)
colors = range(36)

for target, color in zip(targets,colors):
    indicesToKeep = finalDf['label'] == target
    ax.scatter(finalDf.iloc[indicesToKeep, 0]
               , finalDf.iloc[indicesToKeep, 1]
               , c = color
               , s = 50
               )
ax.legend(targets)
ax.grid()
