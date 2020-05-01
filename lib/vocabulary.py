class Vocabulary():

    '''
    A simple utility class than wraps a dictionary and return "unknow" index if the given key does'n exist.
    If "loaded_vocabulary" is a dictionary, the "init_dict" is not called and so the passed dictionary will be used.

    A counter is used to cut the words that are appears in the dataset a number of time less than the min_freq 
    '''


    def __init__(self, counter=None, unknown = None, padding = None, loaded_vocabulary = None, min_freq = 0):
        self.counter = counter
        self.unknown = unknown
        self.padding = padding
        self.min_freq = min_freq
        if(loaded_vocabulary is None):
            self.dict = self.init_dict()
        else:
            self.dict = loaded_vocabulary
        

    def __len__(self):
        return len(self.dict)

    def __getitem__(self, idx):
        if(idx in self.dict):
            return self.dict[idx]
        else:
            return self.dict[self.unknown]

    def word_from_index(self, idx): 
        for k, v in self.dict.items(): 
            if idx == v: 
                return k 

    def init_dict(self):
        d = {}
       
        if(self.padding is not None):
            d.update({self.padding: len(d)})
        if(self.unknown is not None):
            d.update({self.unknown: len(d)})
    
        for elem in self.counter:
            if(elem not in d and self.counter[elem] >= self.min_freq):
                d.update({elem: len(d)})
        
        

        return d        
