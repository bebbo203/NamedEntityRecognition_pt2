import torch
import json
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
import numpy as np
import os

from lib.nerdataset import NERDataset
from lib.nermodel import NERModel
from lib.params import Params
from lib.trainer import Trainer
from lib.gloveparser import GloveParser




params = Params()





#Vocabularies and dataset creation and saving

if(not params.test):
    train_dataset = NERDataset("data/train.tsv")
    valid_dataset = NERDataset("data/dev.tsv", train_dataset.vocabulary, train_dataset.label_vocabulary)
    test_dataset = NERDataset("data/test.tsv", train_dataset.vocabulary, train_dataset.label_vocabulary)
else:
    train_dataset = NERDataset("data/little_train.tsv")
    valid_dataset = NERDataset("data/little_dev.tsv", train_dataset.vocabulary, train_dataset.label_vocabulary)
    test_dataset = NERDataset("data/little_test.tsv", train_dataset.vocabulary, train_dataset.label_vocabulary)


with open("model/vocabulary.json", 'w+') as outfile:
        json.dump(train_dataset.vocabulary.__dict__, outfile)
with open("model/label_vocabulary.json", 'w+') as outfile:
        json.dump(train_dataset.label_vocabulary.__dict__, outfile)



#Pretrained embeddings weights loading

if(params.embeddings_path != None):
    embeddings_weights = np.zeros([len(train_dataset.vocabulary), params.word_embedding_size])
    if(not (os.path.exists(params.processed_embeddings_path))):
        print("Generating embeddings weights...")
        words_missing = 0
        gp = GloveParser(params.embeddings_path, params.word_embedding_size) 
        for word in train_dataset.vocabulary.dict:
            try:
                embeddings_weights[train_dataset.vocabulary[word]] = gp[word]
            except KeyError:
                words_missing += 1
                embeddings_weights[train_dataset.vocabulary[word]] = np.random.uniform(-0.5, 0.5)
        
        with open(params.processed_embeddings_path, 'w') as outfile:
            json.dump(embeddings_weights.tolist(), outfile)
        print("Embedding weights saved!")
        print("Out of %d total words, %d were found in the embedding" % (len(train_dataset.vocabulary), words_missing))
    else:
        with open(params.processed_embeddings_path) as json_file:
                data = json.load(json_file)
                embeddings_weights = torch.Tensor(data)
        print("Embedding weights loaded!")


#Dataloaders preparation, the batch here is a fixed size


train_loader = DataLoader(train_dataset, batch_size=256)
valid_loader = DataLoader(valid_dataset, batch_size=256)
test_loader = DataLoader(test_dataset, batch_size=256)

#Finally load the model and the embedding weights

nermodel = NERModel(len(train_dataset.vocabulary), params.alphabet_size, len(train_dataset.label_vocabulary) ,  params).to(torch.device(params.device))
if(params.embeddings_path != None):
    nermodel.word_embedder.weight.data.copy_(torch.Tensor(embeddings_weights))

trainer = Trainer(
    model = nermodel,
    loss_function = nn.CrossEntropyLoss(ignore_index=train_dataset.label_vocabulary["<pad>"]),
    optimizer = optim.Adam(params=nermodel.parameters())
)

trainer.train(train_loader, valid_loader, epochs=20)