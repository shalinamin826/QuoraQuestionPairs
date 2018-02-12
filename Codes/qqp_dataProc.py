# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 18:48:21 2017

@author: Shalin

NLP Project: Quora question pair

"""

import pandas as pd
from tqdm import trange
import numpy as np
from datetime import datetime as dt
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import logging 
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import spacy
import subject_verb_adj_obj as svo
from joblib import Parallel, delayed
import multiprocessing
from nltk.corpus import wordnet

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
    
def getListSimilarity(l1,l2):
    simIndex = 0
    
    if len(l1)>len(l2):
       for i in range(len(l2)):
           try:
               temp0 = int(l1[i])
               temp1 = int(l2[i])
               if temp0==temp1:
                   simIndex+=1
           except:
               
               if l1[i]=="'m":
                   l1[i]='am'
               
               if l2[i]=="'m":
                   l2[i]='am'
                   
               temp = parserSim(l1[i]+' '+l2[i])
    #           print("Comparing: ",l1[i]+' '+l2[i])
               simIndex+=temp[0].similarity(temp[1])
               
       simIndex/=len(l2)
    
    else:
       for i in range(len(l1)):
           if l1[i]=="'m":
               l1[i]='am'
               
           if l2[i]=="'m":
               l2[i]='am'
               
           temp = parserSim(l1[i]+' '+l2[i])
#           print("Comparing: ",l1[i]+' '+l2[i])
           simIndex+=temp[0].similarity(temp[1])
       simIndex/=len(l1)
   
    return simIndex

def getListSimilarity2(l1,l2):
    simIndex = 0
    q1s, q1v, q1o= l1
    q2s, q2v, q2o= l2
    
    q1svo = []
    q2svo = []
    
    for i in range(len(q1s)):
        if q1o[i]=="'m":
            q1o[i]='am'
        if q1v[i]=="'m":
            q1v[i]='am'
        if q1s[i]=="'m":
            q1s[i]='am'
        q1svo+=[parserSim(q1s[i]+' '+q1v[i]+' '+q1o[i])]
        
    for i in range(len(q2s)):
        if q2o[i]=="'m":
            q2o[i]='am'
        if q2v[i]=="'m":
            q2v[i]='am'
        if q2s[i]=="'m":
            q2s[i]='am'            
        q2svo+=[parserSim(q2s[i]+' '+q2v[i]+' '+q2o[i])]
    
    if len(q1svo)<len(q2svo):
        for s1 in q1svo:
            maxSim = 0
            for s2 in q2svo:
#                print("Comparing: ",s1,' & ',s2)
                tempSim = s1.similarity(s2)
                if tempSim>maxSim:
                    maxSim=tempSim
            simIndex+=maxSim
        simIndex/=len(q1svo)
        
    else:
        for s1 in q2svo:
            maxSim = 0
            for s2 in q1svo:
#                print("Comparing: ",s1,' & ',s2)
                tempSim = s1.similarity(s2)
                if tempSim>maxSim:
                    maxSim=tempSim
            simIndex+=maxSim
        simIndex/=len(q2svo)
        
    return simIndex
             
directory = "S:/Edu/Sem 3 MSCS/NLP/Project/Quora Question Pairs/"
testData = pd.read_csv(directory+'test.csv')
trainData = pd.read_csv(directory+'train.csv')

parserEng = spacy.load('en')
parserSim = spacy.load('en_core_web_md')

#    inputIndices = np.arange(10000)
#    startTime = dt.now()        
#    Parallel(n_jobs=-1, backend="threading")(delayed(procRow)(trainDataList,i) for i in inputIndices)
#    print("Done in ", dt.now()-startTime)

trainDataList= []    
for i in trange(105778,trainData.shape[0]):    
    tempData = trainData.iloc[i,:]
    #Features: similarity, num_nouns, num_verbs, diff_nouns, diff_verbs
    try:
        q1 = parserSim(tempData[3])
        q2 = parserSim(tempData[4])
    except:
        q1 = parserSim(str(tempData[3]))
        q2 = parserSim(str(tempData[4]))
        
    similarity = q1.similarity(q2)
    
    q1_pos = [[x.text, x.pos_] for x in q1]
    q2_pos = [[x.text, x.pos_] for x in q2]
    
    
    
    q1_nouns = [x.text for x in q1 if x.pos_=='NOUN' or x.pos_=='PROPN']
    q1_num_nouns = len(q1_nouns)
    q2_nouns = [x.text for x in q2 if x.pos_=='NOUN' or x.pos_=='PROPN']
    q2_num_nouns = len(q2_nouns)
    
    q1_verbs = [x.text for x in q1 if x.pos_=='VERB']
    q1_num_verbs = len(q1_verbs)
    q2_verbs = [x.text for x in q2 if x.pos_=='VERB']
    q2_num_verbs = len(q2_verbs)
    
    q1_q2_diffNouns = len(list(set(q1_nouns)-set(q2_nouns)))
    q1_q2_diffVerbs = len(list(set(q1_verbs)-set(q2_verbs)))
    
    q2_q1_diffNouns = len(list(set(q2_nouns)-set(q1_nouns)))
    q2_q1_diffVerbs = len(list(set(q2_verbs)-set(q1_verbs)))
    
    if len(q1_nouns) != 0 and len(q2_nouns) != 0:    
        fracDiff_q1q2_nouns = q1_q2_diffNouns/len(q1_nouns)    
        fracDiff_q2q1_nouns = q2_q1_diffNouns/len(q2_nouns)
    elif len(q1_nouns) != 0 or len(q2_nouns) != 0:
        fracDiff_q1q2_nouns = 1
        fracDiff_q2q1_nouns = 1
    else:
        fracDiff_q1q2_nouns = 0
        fracDiff_q2q1_nouns = 0
        
    if len(q1_verbs) != 0 and len(q2_verbs) != 0:    
        fracDiff_q1q2_verbs = q1_q2_diffVerbs/len(q1_verbs)
        fracDiff_q2q1_verbs = q2_q1_diffVerbs/len(q2_verbs)
    elif len(q1_verbs) != 0 or len(q2_verbs) != 0:    
        fracDiff_q1q2_verbs = 1
        fracDiff_q2q1_verbs = 1
    else:    
        fracDiff_q1q2_verbs = 0
        fracDiff_q2q1_verbs = 0 
        
    q1_list = svo.findSVOs(q1)
    q2_list = svo.findSVOs(q2)
    
    q1_sub = []
    q1_verb = []
    q1_obj = []
    q2_sub = []
    q2_verb = []
    q2_obj = []
    
    token = 0
    if len(q1_list) != 0:
        for SVO in q1_list:
            q1_sub.append(SVO[0])
            q1_verb.append(SVO[1])
            q1_obj.append(SVO[2])
            
    else:
        token=1
        q1_sub.append('NA')
        q1_verb.append('NA')
        q1_obj.append('NA')
    
    if len(q2_list) != 0:
        for SVO in q2_list:
            q2_sub.append(SVO[0])
            q2_verb.append(SVO[1])
            q2_obj.append(SVO[2])
   
    else:
        token = 1
        q2_sub.append('NA')
        q2_verb.append('NA')
        q2_obj.append('NA')
    
    if token==0:
#        break
        simSub = getListSimilarity(q1_sub,q2_sub)
        simVerb = getListSimilarity(q1_verb,q2_verb)
        simObj = getListSimilarity(q1_obj,q2_obj)
        sentSim = getListSimilarity2([q1_sub,q1_verb,q1_obj],[q2_sub,q2_verb,q2_obj])
        appData = pd.Series([tempData[3], tempData[4], similarity, q1_num_nouns, q2_num_nouns,q1_num_verbs,
                             q2_num_verbs,mod(q1_num_nouns-q2_num_nouns), 
                   mod(q1_num_verbs-q2_num_verbs), fracDiff_q1q2_nouns, fracDiff_q2q1_nouns, 
                   fracDiff_q1q2_verbs, fracDiff_q2q1_verbs,q1_sub,q1_verb,q1_obj,
                   q2_sub,q2_verb,q2_obj, simSub, simVerb, simObj, (simSub+simVerb+simObj)/3,sentSim
                   ,tempData[-1]])
        
        trainDataList.append(appData)
           
trainDf = pd.DataFrame(trainDataList)
trainDf.columns = ['Question 1','Question 2','Similarity','NounCountQ1','NounCountQ2','VerbCountQ1','VerbCountQ2','Noun Difference','Verb Difference',
                   'Noun_Frac_q1q2','Noun_Frac_q2q1','Verb_Frac_q1q2','Verb_Frac_q2q1',
                   'q1_subject','q1_verb','q1_object','q2_subject','q2_verb','q2_object'
                   ,'SubjectSimilarity','VerbSimilarity','ObjectSimilarity','AvgSimilarity',
                   'svoSentSimilarity','is_duplicate']
trainDf.to_csv(directory+'features.csv',index=False, encoding='utf-8')

#Filtering unwanted columns from dataframe
xCols = [2]+list(np.arange(7,13))+list(np.arange(19,24))
yCols = 24
dataX = reset_cols(trainDf.iloc[:,xCols]).reset_index(drop=True)
dataY = reset_cols(trainDf.iloc[:,yCols]).reset_index(drop=True)

#From csv
trainDf = pd.read_csv(directory+'features0.csv')
dataX = reset_cols(trainDf.iloc[:,:-1]).reset_index(drop=True)
dataY = reset_cols(trainDf.iloc[:,-1]).reset_index(drop=True)

#Seperating into test and train
seperator = int(dataX.shape[0]*2/3)
trainX = dataX.iloc[:seperator,:].reset_index(drop=True)
trainY = dataY.iloc[:seperator].reset_index(drop=True)
testX = dataX.iloc[seperator:,:].reset_index(drop=True)
testY = dataY.iloc[seperator:].reset_index(drop=True)

#Training models
#SVM
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import inspect

def retrieve_name(var):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var]

svmLinear = SVC(kernel='linear',random_state=6)     #.fit(trainX,trainY)
svmRbf = SVC(kernel='rbf',random_state=6)
svmPoly2 = SVC(kernel='poly',degree=2,random_state=6)
svmPoly3 = SVC(kernel='poly',degree=3,random_state=6)

scoreLinear = cross_val_score(svmLinear,dataX.as_matrix(),dataY.as_matrix(),cv=4,n_jobs=-1)
scoreRbf = cross_val_score(svmRbf,dataX.as_matrix(),dataY.as_matrix(),cv=4,n_jobs=-1)
scorePoly2 = cross_val_score(svmPoly2,dataX.as_matrix(),dataY.as_matrix(),cv=4,n_jobs=-1)
scorePoly3 = cross_val_score(svmPoly3,dataX.as_matrix(),dataY.as_matrix(),cv=4,n_jobs=-1)
allScores = [scoreLinear,scoreRbf,scorePoly2,scorePoly3]
for score in allScores:
    print(retrieve_name(score)," : ",score)

#Neural Net
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from datetime import datetime

def normalize(data,mm=0):
        scaler = preprocessing.MinMaxScaler()
        
#        X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
#        X_scaled = X_std * (max - min) + min
        
        if mm==1:
            return data.max(),data.min(),pd.DataFrame(scaler.fit_transform(data))
        else:
            return pd.DataFrame(scaler.fit_transform(data))
        
def nnAnalyze(train_X, train_y, e=500, b=25, v=0):

    his = layers()
    model = his.fit(train_X, train_y, validation_split=0.33,
                    epochs=e, batch_size=b, verbose=v)
    
    plt.plot(model.history['loss'])
    plt.plot(model.history['acc'])
    plt.plot(model.history['val_loss'])
    plt.plot(model.history['val_acc'])
    
    plt.title('model loss e_'+str(e)+' b_'+str(b))
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['trainLoss', 'trainAcc', 'ValLoss', 'ValAcc'], loc='upper left')
    plt.show()

def reset_cols(data):
    data.columns = list(np.arange(data.shape[1]))
    return data

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

#Normalizing test and train data
normX = normalize(dataX).reset_index(drop=True)
normY = dataY.reset_index(drop=True)

def layers(no_layers=1, input_dim = normX.shape[1], out_dim=1, activation='relu', loss='categorical_hinge', optimizer='adam'):
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

girdAnalysis = gridSearch(normX.as_matrix(),normY.as_matrix())
nnAnalyze(normX.as_matrix(),normY.as_matrix(),e=1000,b=25,v=1)
