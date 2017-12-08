# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 19:06:46 2017

@author: shalin
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def RFC(no_tree,train_data,train_labels):
    
    classifier = RandomForestClassifier(n_estimators=no_tree,criterion='entropy',
                                        random_state=6)
    scores = cross_val_score(classifier,train_data,train_labels,cv=4)
    
    return scores 


def run():
    
    directory = "S:/Edu/Sem 3 MSCS/NLP/Project/Quora Question Pairs/"

    #From csv
    dataset = pd.read_csv(directory+'features0.csv')
    train = dataset.drop(['is_duplicate'],1)
    labels = dataset.iloc[:,12]
    
    l = [200] #No of trees to use in RFC
    scoreList = []
    for no_tree in tqdm(l):
        scores = RFC(no_tree,train,labels)
        print("Accuracy using %0.1f RFC is %0.2f (+/- %0.6f)." % (no_tree,
          scores.mean()*100,scores.std()*2))
        scoreList.append([no_tree,scores.mean()*100,scores.std()*2])
    
    for s in scoreList:
        nt,sc,scst = s
        print("Accuracy using %0.1f RFC is %0.2f (+/- %0.6f)." % (nt,sc,scst))
        
if __name__=="__main__":
    run()