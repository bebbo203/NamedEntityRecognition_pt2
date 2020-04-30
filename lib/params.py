class Params():
    device = "cuda"
    test = False
    min_freq = 3
    windows_size = 175
    windows_shift = 175
    word_embedding_size = 200
    char_word_embedding_size = 75
    hidden_dim = 256
    char_hidden_dim = 512
    bidirectional = True
    num_layers = 2
    dropout = 0.5
    max_word_length= 10
    embeddings_path = None
    embeddings_path = "model/glove.6B.200d.txt"
    processed_embeddings_path = "model/processed_embeddings_path.json"
    alphabet_size = 67
    char_embedding_size = 15
    #