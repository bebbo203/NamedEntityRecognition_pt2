import torch
from conllu import parse as conllu_parse
from collections import Counter
from tqdm import tqdm

from .params import Params
from .vocabulary import Vocabulary

class NERDataset(torch.utils.data.Dataset):

    '''
    Dataset class: every sentence is subdivided in windows of window_size size. 
                   As input for the network a vector of dimensions: (window_size, max_word_lenght+1) is returned.
                   Everyone of these vector is a concatenation of a predetermined number (max_word_lenght) of encoded chars
                   and the index of the encoded word. 
    '''

    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'/\|_@#$%ˆ&*˜‘+-=()[]{}"

    def __init__(self, file_path, vocabulary = None, label_vocabulary = None):
        
        self.params = Params()
        
        with open(file_path) as reader:
            sentences = conllu_parse(reader.read())
        
        self.windows_list = self.windows_generator(sentences)
        self.words_counter, self.labels_counter = self.init_counter(sentences)
        if(vocabulary is None):
            self.vocabulary = Vocabulary(self.words_counter, "<unk>", "<pad>", min_freq=self.params.min_freq)
            self.label_vocabulary = Vocabulary(self.labels_counter, padding="<pad>")
        else:
            self.vocabulary = vocabulary
            self.label_vocabulary = label_vocabulary
        self.alphabet = NERDataset.alphabet
        self.encoded_data = self.generate_dataset()
    

    def __len__(self):
        return len(self.encoded_data)
    
    def __getitem__(self, idx):
        return self.encoded_data[idx]

    '''
    Returns a list of windows by cutting and padding a single sentence.
    input: list of words
    output: list of list of words 
    '''
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
            
            encoded_words = self.encode_sentence_conllu(window, self.vocabulary)
            encoded_labels = [self.label_vocabulary[w["lemma"]] if w is not None else self.label_vocabulary["<pad>"] for w in window]
            encoded_chars = self.encode_chars_conllu(window, self.alphabet, self.params.max_word_length)

            for x in zip(encoded_chars, encoded_words):
                x[0][-1] = x[1]

        
            ret.append({
                        "inputs": torch.LongTensor(encoded_chars).to(torch.device(self.params.device)),
                        "outputs": torch.LongTensor(encoded_labels).to(torch.device(self.params.device))
                      })
            
            progress_bar.update(1)
        progress_bar.close()

        return ret
    
    '''
    Given a sentence and a vocabulary, return the given sentence encoded according to the vocabulary.
    input:          a sentence composed by ConLLu parsed words
                    a vocabulary of the class Vocabulary
    output:         a list of index
    '''

    @staticmethod
    def encode_sentence_conllu(sentence, vocabulary):
        ret = []
        for w in sentence:
            if(w is not None):
                word = w["form"]
                ret.append(vocabulary[word])
            else:
                ret.append(vocabulary["<pad>"])
        return ret
    
    '''
    Same method as the one above but with words that are not in conllu format
    '''
    @staticmethod
    def encode_sentence(sentence, vocabulary):
        ret = []
        for w in sentence:
            if(w is not None):
                word = w
                ret.append(vocabulary[word])
            else:
                ret.append(vocabulary["<pad>"])
        return ret
    
    '''
    Given a sentence and an alphabet, return the given sentence encoded according to the alphabet.
    input:          a sentence composed by ConLLu parsed words
                    a string representing an alphabet
    output:         a list of lists of index
    '''

    @staticmethod
    def encode_chars(sentence, alphabet, word_length):
        window_idx = []
        for w in sentence:
            word_idx = []
            if(w is not None):
                word = w
                for c in word:
                    #0 is Padding or not found
                    if(len(word_idx) < word_length):
                        word_idx.append(alphabet.find(c.lower())+1)
                    else:
                        break
            else:
                word_idx.append(0)

            while(len(word_idx) < word_length+1):
                word_idx.append(0)
            window_idx.append(torch.LongTensor(word_idx))
        window_idx = torch.stack(window_idx)
        
        return window_idx
    
    
    @staticmethod
    def encode_chars_conllu(sentence, alphabet, word_length):
        window_idx = []
        for w in sentence:
            word_idx = []
            if(w is not None):
                word = w["form"]
                for c in word:
                    #0 is Padding or not found
                    if(len(word_idx) < word_length):
                        word_idx.append(alphabet.find(c.lower())+1)
                    else:
                        break
            else:
                word_idx.append(0)
            
            while(len(word_idx) < word_length+1):
                word_idx.append(0)

            window_idx.append(torch.LongTensor(word_idx))
        
        window_idx = torch.stack(window_idx)
        
        #Is a Tensor that contains a list of lists of words padded
        return window_idx

    '''
    Given a sentence encoded by one of the methods above, return the original sentence.
    '''

    @staticmethod
    def decode_sentence(sentence, vocabulary):
        ret = []
        
        for idx in sentence:
            ret.append(vocabulary.word_from_index(idx))
        return ret
    
    



