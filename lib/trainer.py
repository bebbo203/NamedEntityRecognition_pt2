import time
from tqdm import tqdm
import torch
from sklearn.metrics import precision_score as sk_precision
from sklearn.metrics import f1_score, recall_score

from .nermodel import NERModel

class Trainer():

    def __init__(self, model, loss_function, optimizer):
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer
    
    def train(self, train_dataset, valid_dataset, epochs = 1):
        
           
        
        start_time = time.time()

        for epoch in range(epochs):
            progress_bar = tqdm(total=len(train_dataset), desc="Epoch: "+str(epoch))
            epoch_loss = 0.0
            self.model.train()
            for step, sample in enumerate(train_dataset):
                inputs = sample["inputs"]
                labels = sample["outputs"]

                self.optimizer.zero_grad()

                predictions = self.model(inputs)
                predictions = predictions.view(-1, predictions.shape[-1])
                labels = labels.view(-1)
                
                sample_loss = self.loss_function(predictions, labels)
                sample_loss.backward()
                self.optimizer.step()
                
                epoch_loss += sample_loss.tolist()

                
                progress_bar.update(1)
            
            progress_bar.close()
            
            torch.save(self.model.state_dict(), "model/weights.pt")
            epoch_loss /= len(train_dataset)
            valid_loss, micro, macro, recall, f1 = self.evaluate(valid_dataset)

            print("\t\t total time: {:0.2f}s".format(time.time()-start_time ))
            print('\t\t train loss = {:0.4f}'.format(epoch_loss))
            print('\t\t valid loss = {:0.4f}'.format(valid_loss))
            print('\t\t micro_precision = {:0.4f}'.format(micro))
            print('\t\t macro_precision = {:0.4f}'.format(macro))
            print('\t\t recall = {:0.4f}'.format(recall))
            print('\t\t f1 = {:0.4f}'.format(f1))

            fx = open("model/plot.csv", "a+")
            fx.write("%f, %f, %f, %f, %f, %f\n" % (epoch_loss, valid_loss, micro, macro, recall, f1))
            fx.close()
    
    def evaluate(self, valid_dataset):
        valid_loss = 0.0
        pad_idx = valid_dataset.dataset.label_vocabulary["<pad>"]

        self.model.eval()
        with torch.no_grad():
            all_predictions = []
            all_labels = []
            valid_loss = 0
            for sample in valid_dataset:
                indexed_in = sample["inputs"]
                indexed_labels = sample["outputs"]
                predictions = self.model(indexed_in)

                sample_loss = self.loss_function(predictions.view(-1, predictions.shape[-1]), indexed_labels.view(-1))
                valid_loss += sample_loss
                predictions = torch.argmax(predictions, -1).view(-1)
                labels = indexed_labels.view(-1)
                
                
                valid_indices = labels != valid_dataset.dataset.label_vocabulary["<pad>"]


                valid_predictions = predictions[valid_indices]
                
                valid_labels = labels[valid_indices]
                
                all_predictions.extend(valid_predictions.tolist())
                all_labels.extend(valid_labels.tolist())
                

            
            
            micro_precision = sk_precision(all_labels, all_predictions, average="micro", zero_division=0)
            macro_precision = sk_precision(all_labels, all_predictions, average="macro", zero_division=0)
            recall = recall_score(all_labels, all_predictions, average='macro')
            f1 = f1_score(all_labels, all_predictions, average='macro')

        return valid_loss / len(valid_dataset), micro_precision, macro_precision, recall, f1