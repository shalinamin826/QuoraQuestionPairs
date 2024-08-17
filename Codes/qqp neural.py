# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 22:50:35 2017

@author: Shalin
"""
import pandas as pd
from tqdm import trange
import numpy as np
from datetime import datetime as dt
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import classification_report as report
from keras import losses

def mod(x):
    if x>=0:
        return x
    else:
        return x*-1
        
def reset_cols(data):
    try:
        data.columns = list(np.arange(data.shape[1]))
    except IndexError:
        data.columns = [0]
    return data        
    
def normalize(data,mm=0):
       scaler = preprocessing.MinMaxScaler()
       if mm==1:
           return data.max(),data.min(),pd.DataFrame(scaler.fit_transform(data))
       else:
           return pd.DataFrame(scaler.fit_transform(data))

def weekdayFromList(l):
    return datetime(l).weekday()

def denorm(data,miny,maxy):
    return data*(maxy-miny)+miny

def gridSearch(datax,datay):
    param_grid = dict(epochs=[150,300], batch_size=[25])
    estimator = KerasRegressor(build_fn=layers, verbose = 1)
    grid = GridSearchCV(estimator=estimator, param_grid=param_grid, verbose=1)
    grid_result = grid.fit(datax, datay)
#    print("\nDONE!!")
    results = grid_result.cv_results_
    resFrame = pd.DataFrame.from_dict(results)
    return resFrame

def nnAnalyze(train_X, train_y, e=500, b=25, v=0):

    his = layers()
    model = his.fit(train_X, train_y, validation_split=0.33,
                    epochs=e, batch_size=b, verbose=v)
    plt.subplot(121)
    plt.title('model loss e_'+str(e)+' b_'+str(b))
    plt.plot(model.history['loss'])
    plt.plot(model.history['val_loss'])
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['trainLoss','ValLoss'], loc='upper left')
    
    plt.subplot(122)   
    plt.title('model acc e_'+str(e)+' b_'+str(b))
    plt.plot(model.history['acc'])
    plt.plot(model.history['val_acc'])
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.legend(['trainAcc', 'ValAcc'], loc='upper left')
    
    plt.show()
    
directory = "S:/Edu/Sem 3 MSCS/NLP/Project/Quora Question Pairs/"

#From csv
trainDf = pd.read_csv(directory+'features0.csv')
dataX = reset_cols(trainDf.iloc[:,:-1]).reset_index(drop=True)
dataY = reset_cols(trainDf.iloc[:,-1]).reset_index(drop=True)

#Seperating into test and train
seperator = int(dataX.shape[0]*2/3)
trainX = dataX.iloc[:seperator,:].reset_index(drop=True)
normTrainX = normalize(trainX).reset_index(drop=True)
trainY = dataY.iloc[:seperator].reset_index(drop=True)
testX = dataX.iloc[seperator:,:].reset_index(drop=True)
normTestX = normalize(testX).reset_index(drop=True)
testY = dataY.iloc[seperator:].reset_index(drop=True)

#Normalizing test and train data
normX = normalize(dataX).reset_index(drop=True)
normY = dataY.reset_index(drop=True)
np.random.seed(7)

def layers(no_layers=1, input_dim = normX.shape[1], out_dim=1, activation='relu', loss='binary_crossentropy', optimizer='rmsprop'):
        metrics = ['accuracy']
        # create model    
        model = Sequential()
        if no_layers==1:
            model.add(Dense(8, input_dim=input_dim, kernel_initializer='normal', activation=activation))
        elif no_layers==2:
            model.add(Dense(10, input_dim=input_dim, kernel_initializer='normal', activation=activation))
            model.add(Dense(5, kernel_initializer='normal', activation=activation))
    
        elif no_layers==3:
            model.add(Dense(10, input_dim=input_dim, kernel_initializer='normal', activation=activation))
            model.add(Dense(5, kernel_initializer='normal', activation=activation))
            model.add(Dense(3, kernel_initializer='normal', activation=activation))
    
        model.add(Dense(out_dim, kernel_initializer='normal', activation=activation))    
        # Compile model
        model.compile(loss=loss, optimizer=optimizer, metrics=metrics)    
        return model

#girdAnalysis = gridSearch(normX.as_matrix(),normY.as_matrix())
nnAnalyze(normX.as_matrix(),normY.as_matrix(),e=600,b=50,v=1)

e, b = 300, 50
estimator = KerasClassifier(build_fn=layers, epochs = e, batch_size=b, verbose = 1)
estimator.fit(trainX.as_matrix(),trainY.as_matrix())

score = estimator.score(trainX.as_matrix(),trainY.as_matrix())
scoreTest = estimator.score(testX.as_matrix(),testY.as_matrix())
