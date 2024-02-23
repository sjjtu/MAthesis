"""
Utility class to evaluate an Autoencoder model for MITBIH data given some test data.

set threshold
compute metrics (accuracy...)
"""

import torch
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

from util.ECGDataset import ECGDataset


class Evaluator:
    def __init__(self,val_normal_path, val_anomaly_path, test_normal_path, test_anomaly_path, model) -> None:
        self.model = model
        self.val_normal_ds = ECGDataset(val_normal_path, hb_type="N")
        self.val_anomaly_ds = ECGDataset(val_anomaly_path, hb_type="A")
        self.test_normal_ds = ECGDataset(test_normal_path, hb_type="N")
        self.test_anomaly_ds = ECGDataset(test_anomaly_path, hb_type="A")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        self.pred_val_normal, self.loss_val_normal = self._reconstruct(self.val_normal_ds)
        self.pred_val_anomaly, self.loss_val_anomaly = self._reconstruct(self.val_anomaly_ds)

        self.pred_normal, self.loss_normal = self._reconstruct(self.test_normal_ds)
        self.pred_anomaly, self.loss_anomaly = self._reconstruct(self.test_anomaly_ds)

    def _reconstruct(self, test_ds):
        predictions, losses = [], []
        criterion = torch.nn.L1Loss(reduction='sum').to(self.device)
        with torch.no_grad():
            self.model = self.model.eval()
            for seq_true, y in test_ds:
                seq_true = seq_true.to(self.device)
                seq_pred = self.model(seq_true)

                loss = criterion(seq_pred, seq_true)

                predictions.append(seq_pred.cpu().numpy().flatten())
                losses.append(loss.item())
        return predictions, losses
    
    def plot_samples(self, model_name, n=3):
        plt.figure()
        plt.suptitle(f"Reconstructed regular heartbeats ({model_name})")
        for i in range(n):
            plt.subplot(1,n,i+1)
            plt.plot(self.pred_normal[i], linestyle="--")
            plt.plot(self.test_normal_ds.__getitem__(i)[0])
            plt.xticks([])
            plt.yticks([])
            
        plt.tight_layout()
        plt.legend(["Reconstruction", "Real"], loc="upper right")
        plt.show()

        plt.figure()
        plt.suptitle(f"Reconstructed anomalous heartbeats ({model_name})")
        for i in range(n):
            plt.subplot(1,n,i+1)
            plt.plot(self.pred_normal[i], linestyle="--")
            plt.plot(self.test_anomaly_ds.__getitem__(i)[0])
            plt.xticks([])
            plt.yticks([])
            
        plt.tight_layout()
        plt.legend(["Reconstruction", "Real"])
        plt.show()

    def find_threshold(self, threshold_list = np.linspace(0,5,21)):
        corr_normal = []
        corr_anomaly = []
        for th in threshold_list:
            corr_normal.append(sum(l <= th for l in self.loss_val_normal)/len(self.loss_val_normal))
            corr_anomaly.append(sum(l > th for l in self.loss_val_anomaly)/len(self.loss_val_anomaly))
        return corr_normal, corr_anomaly

    def predict_class(self, threshold):
        predictions_normal, losses_normal = [], []
        predictions_anomaly, losses_anomaly = [], []
        criterion = torch.nn.L1Loss(reduction='sum').to(self.device)
        with torch.no_grad():
            self.model = self.model.eval()
            for seq_true, y in self.test_normal_ds:
                seq_true = seq_true.to(self.device)
                seq_pred = self.model(seq_true)

                loss = criterion(seq_pred, seq_true)
                pred = "N" if loss <= threshold else "A"
                predictions_normal.append(pred)
                losses_normal.append(loss.item())

            for seq_true, y in self.test_anomaly_ds:
                seq_true = seq_true.to(self.device)
                seq_pred = self.model(seq_true)

                loss = criterion(seq_pred, seq_true)
                pred = "N" if loss <= threshold else "A"
                predictions_anomaly.append(pred)
                losses_anomaly.append(loss.item())
        return predictions_normal, predictions_anomaly

    def evaluate(self, pred_normal, pred_anomaly):
        print(f"TP: {pred_normal.count('N')}")
        print(f"FN: {pred_normal.count('A')}")
        print(f"FP: {pred_anomaly.count('N')}")
        print(f"TN: {pred_anomaly.count('A')}")
        
        print(f"Acc: {metrics.accuracy_score(['N']*len(pred_normal)+['A']*len(pred_anomaly), pred_normal+pred_anomaly)}")
        print(f"Precision: {metrics.precision_score(['N']*len(pred_normal)+['A']*len(pred_anomaly), pred_normal+pred_anomaly, pos_label='A')}")
        print(f"Recall: {metrics.recall_score(['N']*len(pred_normal)+['A']*len(pred_anomaly), pred_normal+pred_anomaly, pos_label='A')}")
        print(f"F1: {metrics.f1_score(['N']*len(pred_normal)+['A']*len(pred_anomaly), pred_normal+pred_anomaly, pos_label='A')}")

        print("Formatted (acc, prec, rec, f1):")
        print(f"{metrics.accuracy_score(['N']*len(pred_normal)+['A']*len(pred_anomaly), pred_normal+pred_anomaly)},{metrics.precision_score(['N']*len(pred_normal)+['A']*len(pred_anomaly), pred_normal+pred_anomaly, pos_label='A')}, {metrics.recall_score(['N']*len(pred_normal)+['A']*len(pred_anomaly), pred_normal+pred_anomaly, pos_label='A')}, {metrics.f1_score(['N']*len(pred_normal)+['A']*len(pred_anomaly), pred_normal+pred_anomaly, pos_label='A')}")