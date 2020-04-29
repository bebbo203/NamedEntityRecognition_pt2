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

        self.conv1 = nn.Conv1d(in_channels=params.max_word_length, out_channels=1, kernel_size=5)
        self.max_pool = nn.MaxPool1d(kernel_size = 2)
        
        self.char_lstm = nn.LSTM(params.char_embedding_size, params.char_word_embedding_size,
                                bidirectional=params.bidirectional,
                                num_layers=params.num_layers, 
                                dropout = params.dropout if params.num_layers > 1 else 0,
                                batch_first=True)


        self.lstm = nn.LSTM(params.word_embedding_size + params.char_word_embedding_size, params.hidden_dim, 
                            bidirectional=params.bidirectional,
                            num_layers=params.num_layers, 
                            dropout = params.dropout if params.num_layers > 1 else 0,
                            batch_first=True)
       
        lstm_output_dim = params.hidden_dim if params.bidirectional is False else params.hidden_dim * 2

        self.dropout = nn.Dropout(params.dropout)
        self.classifier = nn.Linear(lstm_output_dim, num_classes)

    def forward(self, x):
        word = x[:, :, -1].type(torch.LongTensor).to(self.device)
        chars = x[:, :, :-1].type(torch.LongTensor).to(self.device)
        
        

        #u = (batch_size, window_size, word_size - 1, single_char_embedding_dim)
        u = self.char_embedder(chars)
        

        o, (h, c) = self.char_lstm(u.view(u.size()[0]*u.size()[1], u.size()[2], u.size()[3]))

        char_embedding = h[-1].view(u.size()[0], u.size()[1], self.params.char_word_embedding_size)
          
       
        embeddings = self.word_embedder(word)
        embeddings = self.dropout(embeddings)

        final_emb = torch.cat((embeddings, char_embedding), dim=2)

       

        o, (h, c) = self.lstm(final_emb)
        
        o = self.dropout(o)
        output = self.classifier(o)
        
        return output