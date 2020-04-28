import numpy as np
from typing import List, Tuple
from collections import Counter

from lib import *
from lib import NERDataset
from lib import NERModel
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
import torch
from tqdm import tqdm

import json


class StudentModel():
    
    def __init__(self, device):
        self.device = device
        self.params = Params()
        with open('model/vocabulary.json') as json_file:
            data = json.load(json_file)
            self.vocabulary = Vocabulary(unknown=data["unknown"], padding=data["padding"], loaded_vocabulary=data["dict"])
        with open('model/label_vocabulary.json') as json_file:
            data = json.load(json_file)
            self.label_vocabulary = Vocabulary(unknown=data["unknown"], padding=data["padding"], loaded_vocabulary=data["dict"])

        self.nermodel = NERModel(len(self.vocabulary), len(self.label_vocabulary) , self.params).to(torch.device(self.device))
        self.nermodel.load_state_dict(torch.load("model/weights.pt", map_location=torch.device(self.device)))
        self.nermodel.eval()
        
    def windows_generator(self, sentence):
        windows_list = []
        
        for i in range(0, len(sentence), self.params.windows_size):
            window = sentence[i:i+self.params.windows_size]
            if(len(window) < self.params.windows_size):
                window += [None] * (self.params.windows_size - len(window))            
            windows_list.append(window)
        return windows_list

    

    def predict(self, tokens: List[List[str]]) -> List[List[str]]:
        ret = []
        for sentence in tokens:
            windows_list = self.windows_generator(sentence)
            sentence_pred = []
            
            for window in windows_list:
                n_none = window.count(None)
                encoded_window = []
                for w in window:
                    if(w is None):
                        encoded_window.append(self.vocabulary["<pad>"])
                    else:
                        encoded_window.append(self.vocabulary[w])            

               

                encoded_window = torch.LongTensor(encoded_window).to(torch.device(self.device))

                pred = self.nermodel(encoded_window.unsqueeze(0))
                pred = pred.squeeze(0)
                if(n_none > 0):
                    pred = pred[:-n_none]
                
                pred = torch.argmax(pred, dim=1)
                decoded_pred = NERDataset.decode_sentence(pred.tolist(), self.label_vocabulary)
                sentence_pred.extend(decoded_pred)
                
            
            ret.append(sentence_pred)
        return ret

def flat_list(l):
    return [_e for e in l for _e in e]


a = StudentModel("cuda")



f = open("data/dev.tsv")
pad_idx = a.label_vocabulary["<pad>"]

batch=[]
pred = []

tokens_s = []
labels_s = []

tokens = []
labels = []




for line in f.readlines():
    line = line.strip()
    
    if line.startswith('# '):
        tokens = []
        labels = []
    elif line == '':
        tokens_s.append(tokens)
        labels_s.append(labels)
    else:
        _, token, label = line.split('\t')
        tokens.append(token)
        labels.append(label)



pred_counter = Counter()
truth_counter = Counter() 
batch_labels = []
total_pred = []
progress_bar = tqdm(total=len(labels_s), desc='Evaluating')
for sentence, truth in zip(tokens_s, labels_s):
    batch.append(sentence)
    batch_labels.append(truth)
    if(len(batch) == 256):
        pred = a.predict(batch)
        total_pred.append(pred)
        
        for t, l, p in zip(batch, batch_labels , pred):
            for tt, lt, pt in zip(t,l,p):
                pred_counter[pt]+=1
                truth_counter[lt]+=1
        batch = []
        batch_labels = []
        progress_bar.update(256)
progress_bar.close()

pred = a.predict(batch)
total_pred.append(pred)
for t, l, p in zip(batch, batch_labels , pred):
    for tt, lt, pt in zip(t,l,p):
        pred_counter[pt]+=1
        truth_counter[lt]+=1




print("PRED: " + str(pred_counter))
print("GOLD: " + str(truth_counter))

flat_predictions_s = flat_list(flat_list(total_pred))
flat_labels_s = flat_list(labels_s)

p = precision_score(flat_labels_s, flat_predictions_s, average='macro')
r = recall_score(flat_labels_s, flat_predictions_s, average='macro')
f = f1_score(flat_labels_s, flat_predictions_s, average='macro')
conf = confusion_matrix(flat_labels_s, flat_predictions_s)

print(f'# precision: {p:.4f}')
print(f'# recall: {r:.4f}')
print(f'# f1: {f:.4f}')

print(conf)