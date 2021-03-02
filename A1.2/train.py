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

def save_model(path, model, n):

    if path == None:
        return
    
    state_dict = {'model': model.state_dict(),
                  'vocab_len': n}
    
    torch.save(state_dict, path)
    print(f'Model saved at ==> {path}')


def load_model(path, model):

    if path==None:
        return
    
    state = torch.load(path, map_location=device)
    
    model.load_state_dict(state["model"])
    print(state["vocab_len"])
    print(f'Model loaded from <== {path}')

if __name__ == "__main__":
    train_file = sys.argv[1]
    val_file = sys.argv[2]


    # In[3]:


    device = "cuda" if torch.cuda.is_available() else "cpu"


    # In[4]:


    train = pd.read_csv(train_file)
    train = preprocess(train, 'Subject')
    train = preprocess(train, "Content")
    train = convert_class(train)
    train["Text"] = train.Subject+" "+train.Content
    train = train[["Class", "Text"]]

    val = pd.read_csv(val_file)
    val = preprocess(val, 'Subject')
    val = preprocess(val, "Content")
    val = convert_class(val)
    val["Text"] = val.Subject+" "+val.Content
    val = val[["Class", "Text"]]

    train.to_csv('processed_train.csv', index=False)
    val.to_csv('processed_val.csv', index=False)

    label_field = Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.float)
    text_field = Field(tokenize='spacy', batch_first=True, include_lengths=True, lower=True)
    fields = [('Class', label_field), ('Text', text_field)]

    train, valid = TabularDataset.splits(path="./", train='processed_train.csv', 
                                               validation='processed_val.csv', format='CSV', 
                                               fields=fields, skip_header=True)

    text_field.build_vocab(train, min_freq=2, vectors='glove.840B.300d')


    # In[10]:
    with open("text_field","wb") as f:
        dill.dump(text_field, f)


    batch_size = 48
    train_iter = BucketIterator(train, batch_size=batch_size, sort_key=lambda x: len(x.Text),
                                device=device, sort=True, sort_within_batch=True, shuffle=True)
    valid_iter = Iterator(valid, sort=False, shuffle=False, batch_size=batch_size, device=device)
    
    class Model(nn.Module):

        def __init__(self):
            super(Model, self).__init__()

            self.embedding = nn.Embedding(len(text_field.vocab), 300)
            self.embedding.weight.data = text_field.vocab.vectors.cuda()
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


    weights = [272, 163, 92, 58, 385, 41, 611]
    class_weights = (1622/torch.FloatTensor(weights)).cuda()
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    model = Model().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30], gamma=0.1)

    def train(num_epochs, model_name):

        running_loss = 0.0
        valid_running_loss = 0.0
        step = 0
        train_loss_list = []
        valid_loss_list = []
        steps_list = []
        best_valid_loss = float("Inf")

        # training loop
        model.train()
        for epoch in range(num_epochs):
            scheduler.step()
            for (labels, (text, text_len)), _ in train_iter:   

                labels = (labels.long()-1).to(device)
                text = text.to(device)
                text_len = text_len.to(device)
                output = model(text, text_len)

                loss = criterion(output, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                step += 1

                if step % len(train_iter) == 0:
                    model.eval()
                    with torch.no_grad():                    
                      # validation loop
                      for (labels, (text, text_len)), _ in valid_iter: 
                        labels = (labels.long()-1).to(device)
                        text = text.to(device)
                        text_len = text_len.to(device)
                        output = model(text, text_len)

                        loss = criterion(output, labels)
                        valid_running_loss += loss.item()

                    average_train_loss = running_loss / len(train_iter)
                    average_valid_loss = valid_running_loss / len(valid_iter)
                    train_loss_list.append(average_train_loss)
                    valid_loss_list.append(average_valid_loss)
                    steps_list.append(step)

                    running_loss = 0.0                
                    valid_running_loss = 0.0
                    model.train()

                    print('Epoch [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}'
                          .format(epoch+1, num_epochs, average_train_loss, average_valid_loss))

                    if best_valid_loss > average_valid_loss:
                        best_valid_loss = average_valid_loss
                        save_model(model_name, model, len(text_field.vocab))

        plt.plot(steps_list, train_loss_list, label='Train')
        plt.plot(steps_list, valid_loss_list, label='Valid')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.legend()
        plt.show() 
        print('Finished Training!')

    model_name = sys.argv[3]
    train(40, model_name)


    # In[13]:


    # Evaluation Function
    def evaluate(model, test_iter):
        y_pred = []
        y_true = []

        model.eval()
        with torch.no_grad():
            for (labels, (text, text_len)), _ in test_iter: 
                labels = (labels.long()).to(device)
                text = text.to(device)
                text_len = text_len.to(device)
                output = model(text, text_len)

                output = torch.argmax(output, dim=1) + 1
                y_pred.extend(output.tolist())
                y_true.extend(labels.tolist())

        cm = confusion_matrix(y_true, y_pred)
        total = 0
        for i in range(cm.shape[0]):
            total += cm[i][i]/sum(cm[i])

        print("Micro Accuraccy: ", total/cm.shape[0]) 
        print("Macro Accuracy: ", np.mean(np.asarray(y_true) == np.asarray(y_pred)))
    #     np.savetxt("test.txt", np.asarray(y_pred), fmt='%d', newline='\n')

    best_model = Model().to(device)
    load_model(model_name, best_model)
    evaluate(best_model, valid_iter)





