"""
Utiltiy class to train AE model for anomaly detection
given some training, validation and test data
"""

import random

import torch
import pandas as pd 
import numpy as np

from ECGDataset import ECGDataset
from lstmae import RecurrentAutoencoder

class AutoEncTrainRoutine:
    def __init__(self, seq_len=180, n_feat=1, emb_dim=32, training_data_path="data/normal_train_180.csv", window_size=180):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_ds= ECGDataset(training_data_path, 
                                  f"data/train_labels_{window_size}.csv")
        self.test_normal_ds = ECGDataset(f"data/normal_test{window_size}.csv", 
                                         f"data/normal_labels_{window_size}.csv")
        self.test_anomalie_ds = ECGDataset(f"data/anomalie_test{window_size}.csv", 
                                           f"data/anomalie_labels_{window_size}.csv")
        self.val_ds = ECGDataset(f"data/normal_val_{window_size}.csv", 
                                 f"data/val_labels_{window_size}.csv")
        self.model = RecurrentAutoencoder(seq_len, n_feat, self.device, emb_dim)
        
    def train_model(self, n_epochs=30, lr=5e-4, batch_size=1):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = torch.nn.L1Loss(reduction='sum').to(self.device)
        history = dict(train=[], val=[])

        for epoch in range(1, n_epochs + 1):
            model = self.model.train()

            train_losses = []
            val_losses = []

            train_dl = torch.utils.data.DataLoader(self.train_ds, batch_size=batch_size, shuffle=True)
            val_dl = torch.utils.data.DataLoader(self.val_ds)

            size = len(train_dl.dataset)
            for batch, (X,y) in enumerate(train_dl):
                # Compute prediction and loss
                X = X.to(self.device)
                pred = model(X)
                loss = criterion(pred, X)
                train_losses.append(loss.item())

                # Backpropagation
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                if batch % 10000 == 0:
                    loss, current = loss.item(), (batch + 1) * len(X)
                    print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            
            with torch.no_grad():  # requesting pytorch to record any gradient for this block of code
                for (seq_true, y) in val_dl:
                    seq_true = seq_true.to(self.device)   # putting sequence to gpu
                    seq_pred = model(seq_true)    # prediction

                    loss = criterion(seq_pred, seq_true)  # recording loss

                    val_losses.append(loss.item())    # storing loss into the validation losses

            train_loss = np.mean(train_losses)
            val_loss = np.mean(val_losses)

            history['train'].append(train_loss)
            history['val'].append(val_loss)

            print(f'Epoch {epoch}: train loss = {train_loss}, val loss = {val_loss}')

        return model.eval(), history
    
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
    
    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, 
                                              map_location=torch.device('cpu')))
    
if __name__=="__main__":
    vanilla_ae = AutoEncTrainRoutine()
    vanilla_ae.load_model("models/lstmae_180_embed32.pth")
    vanilla_ae.train_model()