class Params():
    device = "cuda"
    test = False
    windows_size = 50
    windows_shift = 50
    word_embedding_size = 50
    hidden_dim = 256
    bidirectional = True
    num_layers = 2
    dropout = 0.5
    embeddings_path = "model/glove.6B.50d.txt"
    processed_embeddings_path = "model/processed_embeddings_path.json"