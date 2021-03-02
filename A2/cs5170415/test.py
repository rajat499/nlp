import pickle
import sys

import sklearn_crfsuite
from sklearn_crfsuite import metrics

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
        for w in sentence:
            temp = w.strip().split(" ")
            x.append(temp[0])
        word.append(pos_tag(x))
    
    return word

test_file = sys.argv[1]
model_path = sys.argv[3]
out_path = sys.argv[2]

test_token = process(test_file)
test_token = process_token(test_token)

with open(model_path, 'rb') as file:  
    crf = pickle.load(file)
    
y_pred = crf.predict(test_token)
y_pred = ["\n".join(y) for y in y_pred]
y_pred = "\n".join(y_pred)

with open(out_path, 'w') as file:  
    file.write(y_pred+"\n")
