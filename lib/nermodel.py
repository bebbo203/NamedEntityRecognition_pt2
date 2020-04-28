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
        


        char_embedding = torch.Tensor().to(self.device)
        for i in range(self.params.windows_size):
            # Need to change dimensions since the lstm level needs the input as (n_timesteps, batch, n_features)
            #w = (batch_size, max_word_length, single_char_embedding_dim)
            w = u[:, i, : , :]

           
            #out = (batch_size, out_channels, 3?)
            out = self.max_pool(self.conv1(w))    

            #out  = (batch_size, char_embedding_dim, 23/pool_kernel)
            
           
            char_embedding = torch.cat((char_embedding, out), dim=1)
      
       
        embeddings = self.word_embedder(word)
        embeddings = self.dropout(embeddings)

        final_emb = torch.cat((embeddings, char_embedding), dim=2)

       

        o, (h, c) = self.lstm(final_emb)
        
        o = self.dropout(o)
        output = self.classifier(o)
        
        return output