class Params():
    device = "cuda"
    test = False
    min_freq = 5
    windows_size = 100
    windows_shift = 100
    word_embedding_size = 200
    char_word_embedding_size = 25
    hidden_dim = 256
    char_hidden_dim = 128
    bidirectional = True
    num_layers = 2
    dropout = 0.3
    max_word_length= 10
    embeddings_path = None
    embeddings_path = "model/glove.6B.200d.txt"
    processed_embeddings_path = "model/processed_embeddings_path.json"
    alphabet_size = 67
    char_embedding_size = 5
    #