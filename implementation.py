import numpy as np
from typing import List, Tuple
import torch

from model import Model
from stud.lib import Vocabulary
from stud.lib import Params
from stud.lib import NERModel
from stud.lib import NERDataset
import json


def build_model(device: str) -> Model:
    # STUDENT: return StudentModel()
    # STUDENT: your model MUST be loaded on the device "device" indicates
    return StudentModel(device)


class RandomBaseline(Model):

    options = [
        ('LOC', 98412),
        ('O', 2512990),
        ('ORG', 71633),
        ('PER', 115758)
    ]

    def __init__(self):

        self._options = [option[0] for option in self.options]
        self._weights = np.array([option[1] for option in self.options])
        self._weights = self._weights / self._weights.sum()

    def predict(self, tokens: List[List[str]]) -> List[List[str]]:
        return [[str(np.random.choice(self._options, 1, p=self._weights)[0]) for _x in x] for x in tokens]


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

        self.nermodel = NERModel(len(self.vocabulary), self.params.alphabet_size ,len(self.label_vocabulary), self.params, device = self.device).to(torch.device(self.device))
        self.nermodel.load_state_dict(torch.load("model/weights.pt", map_location=torch.device(self.device)))
        self.nermodel.eval()
        
    def windows_generator(self, sentence):
        windows_list = []
        
        for i in range(0, len(sentence), self.params.windows_shift):
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
                encoded_window_words = []
                encoded_window_chars = []
                
                for w in window:
                    if(w is None):
                        encoded_window_words.append(self.vocabulary["<pad>"])
                    else:
                        encoded_window_words.append(self.vocabulary[w])   
                
                encoded_window_chars = NERDataset.encode_chars(window, NERDataset.alphabet , self.params.max_word_length)
                
                for x in zip(encoded_window_chars, encoded_window_words):
                    x[0][-1] = x[1]
                
                encoded_window = torch.LongTensor(encoded_window_chars).to(torch.device(self.device))

                pred = self.nermodel(encoded_window.unsqueeze(0))
                pred = pred.squeeze(0)
                if(n_none > 0):
                    
                    pred = pred[:-n_none]
                
                pred = torch.argmax(pred, dim=1)
                decoded_pred = NERDataset.decode_sentence(pred.tolist(), self.label_vocabulary)
                sentence_pred.extend(decoded_pred)
            ret.append(sentence_pred)
        return ret

