import torch
from torch import nn

class NERModel(nn.Module):
    def __init__(self, vocab_size, alphabet_size, num_classes, params, device = None):
        super(NERModel, self).__init__()
        
        if(device is None):
            self.device = params.device
        else:
            self.device = device

        self.params = params
        self.word_embedder = nn.Embedding(vocab_size, params.word_embedding_size)
        self.char_embedder = nn.Embedding(alphabet_size, params.char_embedding_size)



        '''
        One thing to note here is that every LSTM layer has the batch_first flag at True. Since the data are passedd
        in a batch-major format this will remove the need of the reshapes of the inputs
        '''


        self.char_lstm = nn.LSTM(params.char_embedding_size  * params.max_word_length, params.char_word_embedding_size,
                                bidirectional=params.bidirectional,
                                num_layers=params.num_layers, 
                                dropout = params.dropout if params.num_layers > 1 else 0,
                                batch_first=True)


        self.lstm = nn.LSTM(params.word_embedding_size + params.char_word_embedding_size * 2, params.hidden_dim, 
                            bidirectional=params.bidirectional,
                            num_layers=params.num_layers, 
                            dropout = params.dropout if params.num_layers > 1 else 0,
                            batch_first=True)
       
        lstm_output_dim = params.hidden_dim if params.bidirectional is False else params.hidden_dim * 2

        self.dropout = nn.Dropout(params.dropout)
        self.classifier = nn.Linear(lstm_output_dim, num_classes)

    '''
    In the forward method, the model need to dismount the word index attached to the end of the repectivley encoded char vector. 
    After that operation the word index goes into the Embedding, while chars indexes go in an LSTM after a resize
    (since it's only encoding words, every word is indipendent from the others and so it can be seen as a big batch)
    '''

    def forward(self, x):
        #x = (batch_size, window_size, word_size + 1)
        word = x[:, :, -1].type(torch.LongTensor).to(self.device)
        chars = x[:, :, :-1].type(torch.LongTensor).to(self.device)
        
        #u = (batch_size, window_size, word_size, single_char_embedding_dim)
        u = self.char_embedder(chars)
        u = u.reshape(u.size()[0], u.size()[1], u.size()[2]*u.size()[3])
        
        o, (h, c) = self.char_lstm(u)
        o = self.dropout(o)

        #embeddings = (batch_size, windows_size, word_embedding_size)
        embeddings = self.word_embedder(word)
        embeddings = self.dropout(embeddings)

        
        final_emb = torch.cat((embeddings, o), dim=2)

        
        o, (h, c) = self.lstm(final_emb)
        
        o = self.dropout(o)
        output = self.classifier(o)
        
        return output