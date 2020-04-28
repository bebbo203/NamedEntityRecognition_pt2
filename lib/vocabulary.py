class Vocabulary():
    def __init__(self, counter=None, unknown = None, padding = None, loaded_vocabulary = None):
        self.counter = counter
        self.unknown = unknown
        self.padding = padding
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
            if(elem not in d):
                d.update({elem: len(d)})
        
        

        return d        
