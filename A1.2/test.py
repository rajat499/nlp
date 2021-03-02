#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import dill

from util import preprocess, convert_class
import torch
torch.manual_seed(0)
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.optim as optim
from torch.optim import lr_scheduler

from torchtext.data import Field, TabularDataset, BucketIterator, Iterator
from torchtext.vocab import GloVe

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

import warnings
import sys
warnings.filterwarnings('ignore')


# In[ ]:


if __name__ == "__main__":
    
    val_file = sys.argv[1]
    outfile = sys.argv[2]
    model_name = sys.argv[3]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    val = pd.read_csv(val_file)
    val = preprocess(val, 'Subject')
    val = preprocess(val, "Content")
    val["Text"] = val.Subject+" "+val.Content
    val = val[["Text"]]
    
    val.to_csv('processed_test.csv', index=False)
    
    text_field = None
    with open("text_field", "rb") as f:
        text_field = dill.load(f)

    fields = [('Text', text_field)]
    
    valid = TabularDataset.splits(path="./", train='processed_test.csv', format='CSV', fields=fields, skip_header=True)
    
    batch_size = 48
    valid_iter = Iterator(valid[0], sort=False, shuffle=False, batch_size=batch_size, device=device)
    
    state = torch.load(model_name, map_location=device)
    
    class Model(nn.Module):
    
        def __init__(self):
            super(Model, self).__init__()

            self.embedding = nn.Embedding(len(text_field.vocab), 300)
            self.dimension = 72
            self.lstm = nn.LSTM(input_size=300, hidden_size=self.dimension, num_layers=1, 
                            batch_first=True, bidirectional=True)
            self.drop = nn.Dropout(p=0.5)
            self.fc = nn.Linear(2*self.dimension, 7)

        def forward(self, text, text_len):

            embedded_text= self.embedding(text)

            inp = pack_padded_sequence(embedded_text, text_len.cpu(), enforce_sorted=False, batch_first=True)
            out, _ = self.lstm(inp)
            out, _ = pad_packed_sequence(out, batch_first=True)

            fwd = out[range(len(out)), text_len-1, :self.dimension]
            rev = out[:, 0, self.dimension:]
            out = torch.cat((fwd, rev), 1)
            inp_fc = self.drop(out)

            out_fc = self.fc(inp_fc)
            out_fc = torch.squeeze(out_fc, 1)
            out_fc = (1+torch.tanh(out_fc))/2

            return out_fc
    
    def evaluate(model, test_iter):
        y_pred = []
        y_true = []

        model.eval()
        with torch.no_grad():
            for ((text, text_len)), _ in test_iter: 
                text = text.to(device)
                text_len = text_len.to(device)
                output = model(text, text_len)

                output = torch.argmax(output, dim=1) + 1
                y_pred.extend(output.tolist())

        np.savetxt(outfile, np.asarray(y_pred), fmt='%d', newline='\n')
        

    best_model = Model().to(device)
    best_model.load_state_dict(state["model"])
    evaluate(best_model, valid_iter)

