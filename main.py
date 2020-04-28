import torch
import json
from torch.utils.data import DataLoader
from torch import nn
from torch import optim

from lib.nerdataset import NERDataset
from lib.nermodel import NERModel
from lib.params import Params
from lib.trainer import Trainer




params = Params()





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



train_loader = DataLoader(train_dataset, batch_size=256)
valid_loader = DataLoader(valid_dataset, batch_size=256)
test_loader = DataLoader(test_dataset, batch_size=256)

nermodel = NERModel(len(train_dataset.vocabulary), len(train_dataset.label_vocabulary) , params).to(torch.device(params.device))




trainer = Trainer(
    model = nermodel,
    loss_function = nn.CrossEntropyLoss(ignore_index=train_dataset.label_vocabulary["<pad>"]),
    optimizer = optim.Adam(params=nermodel.parameters())
)

trainer.train(train_loader, valid_loader, epochs=1000)