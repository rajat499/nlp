#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import sys
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from util import preprocess, convert_class
import pickle

if __name__ == "__main__":
    
    filename = sys.argv[1]
    model_path = sys.argv[2]
    
    train = pd.read_csv(filename)
    train = preprocess(train, 'Subject')
    train = preprocess(train, "Content")
    df = convert_class(train)
    
    vectorizer = TfidfVectorizer().fit(df["Subject"] + " " + df["Content"])
    X_train = vectorizer.transform(df["Subject"]+" "+df["Content"])
    
    vect_path = "vectorizer.pkl"  
    with open(vect_path, 'wb') as file:  
        pickle.dump(vectorizer, file)
        
    clf = BaggingClassifier(base_estimator=SGDClassifier(), random_state=3, n_estimators=12, n_jobs=-3)
    clf = clf.fit(X_train, df.Class)
    
    pred = clf.predict(X_train)
    mat = confusion_matrix(pred, df.Class)
    total = 0
    for i in range(mat.shape[0]):
        total += mat[i][i]/sum(mat[i])

    print("Micro Accuraccy: ", total/mat.shape[0]) 
    print("Macro Accuracy: ", np.mean(pred == df.Class))
    
    with open(model_path, 'wb') as file:  
        pickle.dump(clf, file)
