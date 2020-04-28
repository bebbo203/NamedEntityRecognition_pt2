import torch
from torch import nn

class NERModel(nn.Module):
    def __init__(self, vocab_size, num_classes, params):
        super(NERModel, self).__init__()
        
        self.params = params
        self.word_embedder = nn.Embedding(vocab_size, params.word_embedding_size)



        self.lstm = nn.LSTM(params.word_embedding_size, params.hidden_dim, 
                            bidirectional=params.bidirectional,
                            num_layers=params.num_layers, 
                            dropout = params.dropout if params.num_layers > 1 else 0,
                            batch_first=True)
       
        lstm_output_dim = params.hidden_dim if params.bidirectional is False else params.hidden_dim * 2

        self.dropout = nn.Dropout(params.dropout)
        self.classifier = nn.Linear(lstm_output_dim, num_classes)

    def forward(self, x):
        word_embeddings = self.word_embedder(x)
        word_embeddings = self.dropout(word_embeddings)
        o, (h, c) = self.lstm(word_embeddings)
        o = self.dropout(o)
        output = self.classifier(o)
        
        return output