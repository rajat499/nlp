#!/usr/bin/env python
# coding: utf-8

import sklearn_crfsuite
from sklearn_crfsuite import metrics

import pickle
import sys

import nltk
from nltk.tag import pos_tag

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

import re
from itertools import groupby

from util import process_token

def process(filename):
    f = open(filename, "r")
    l = f.read()
    l = l.strip()
    l = re.sub(r'\n\s*\n', '\n\n', l).split("\n")
    l = [list(s) for e, s in groupby(l, key=bool) if e]
    word = []
    label = []
    for i, sentence in enumerate(l):
        x = []
        y = []
        for w in sentence:
            temp = w.strip().split(" ")
            x.append(temp[0])
            y.append(temp[1])
        word.append(pos_tag(x))
        label.append(y)
    
    return word, label

train_file = sys.argv[1]
val_file = sys.argv[2]
model_path = sys.argv[3]

train_token, train_label = process(train_file)
val_token, val_label = process(val_file)

train_token = process_token(train_token)
val_token = process_token(val_token)

crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1 = 0.24,
    c2 = 0.17,
    max_iterations = 50,
    all_possible_transitions=True
)

crf.fit(train_token, train_label)

with open(model_path, 'wb') as file:  
    pickle.dump(crf, file)

