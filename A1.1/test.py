#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


# In[ ]:


if __name__ == "__main__":
    
    model_path = sys.argv[1]
    filename = sys.argv[2]
    out_path = sys.argv[3]
    
    test = pd.read_csv(filename)
    test = preprocess(test, 'Subject')
    df = preprocess(test, "Content")
    
    vect_path = "vectorizer.pkl"  
    with open(vect_path, 'rb') as file:  
        vectorizer = pickle.load(file)
    
    X_val = vectorizer.transform(df["Subject"]+" "+df["Content"])
    
    with open(model_path, 'rb') as file:  
        clf = pickle.load(file)
    
    pred = clf.predict(X_val)
    np.savetxt(out_path, pred, fmt='%d', newline='\n')

