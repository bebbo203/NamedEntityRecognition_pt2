import torch
from conllu import parse as conllu_parse
from collections import Counter
from tqdm import tqdm

from .params import Params
from .vocabulary import Vocabulary

class NERDataset(torch.utils.data.Dataset):
    
    def __init__(self, file_path, vocabulary = None, label_vocabulary = None):
        
        self.params = Params()
        
        with open(file_path) as reader:
            sentences = conllu_parse(reader.read())
        
        self.windows_list = self.windows_generator(sentences)
        self.words_counter, self.labels_counter = self.init_counter(sentences)
        if(vocabulary is None):
            self.vocabulary = Vocabulary(self.words_counter, "<unk>", "<pad>")
            self.label_vocabulary = Vocabulary(self.labels_counter, padding="<pad>")
        else:
            self.vocabulary = vocabulary
            self.label_vocabulary = label_vocabulary
        self.encoded_data = self.generate_dataset()
    

    def __len__(self):
        return len(self.encoded_data)
    
    def __getitem__(self, idx):
        if self.encoded_data is None:
            raise RuntimeError("""Trying to retrieve elements but index_dataset
            has not been invoked yet! Be sure to invoce index_dataset on this object
            before trying to retrieve elements. In case you want to retrieve raw
            elements, use the method get_raw_element(idx)""")
        return self.encoded_data[idx]


    def windows_generator(self, sentences):
        windows_list = []
        for sentence in sentences:
            for i in range(0, len(sentence), self.params.windows_shift):
                window = sentence[i:i+self.params.windows_size]
                if(len(window) < self.params.windows_size):
                    window += [None] * (self.params.windows_size - len(window))            
                windows_list.append(window)
        return windows_list

    def init_counter(self, sentences):
        words_counter = Counter()
        labels_counter = Counter()
        for sentence in sentences:
            for w in sentence:
                word = w["form"]
                label = w["lemma"]
                words_counter[word] += 1
                labels_counter[label] += 1
        return words_counter, labels_counter

    def generate_dataset(self):
        ret = list()
        progress_bar = tqdm(total=len(self.windows_list), desc='generating_dataset')
        for window in self.windows_list:
            
            encoded_words = self.encode_sentence(window, self.vocabulary)
            encoded_labels = [self.label_vocabulary[w["lemma"]] if w is not None else self.label_vocabulary["<pad>"] for w in window]
        
            ret.append({
                        "inputs": torch.LongTensor(encoded_words).to(torch.device(self.params.device)),
                        "outputs": torch.LongTensor(encoded_labels).to(torch.device(self.params.device))
                      })
            
            progress_bar.update(1)
        progress_bar.close()

        return ret
    
    
    @staticmethod
    def encode_sentence(sentence, vocabulary):
        ret = []
        for w in sentence:
            if(w is not None):
                word = w["form"]
                ret.append(vocabulary[word])
            else:
                ret.append(vocabulary["<pad>"])
        return ret

    @staticmethod
    def decode_sentence(sentence, vocabulary):
        ret = []
        
        for idx in sentence:
            ret.append(vocabulary.word_from_index(idx))
        return ret
    
    



