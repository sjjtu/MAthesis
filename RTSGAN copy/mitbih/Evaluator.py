"""
Utility class to evaluate an Autoencoder model for MITBIH data given some test data.

set threshold
compute metrics (accuracy...)
"""

import torch
import numpy as np
from sklearn import metrics

from mitbih.ECGDataset import ECGDataset


class Evaluator:
    def __init__(self, test_normal_path, test_anomaly_path, test_anomaly_label_path, model) -> None:
        self.model = model
        self.test_normal_ds = ECGDataset(test_normal_path)
        self.test_anomaly_ds = ECGDataset(test_anomaly_path, test_anomaly_label_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    def find_threshold(self):
        threshold_list = np.linspace(0,5,21)
        corr_normal = []
        corr_anomaly = []
        for th in threshold_list:
            corr_normal.append(sum(l <= th for l in self.loss_normal)/len(self.loss_normal))
            corr_anomaly.append(sum(l > th for l in self.loss_anomaly)/len(self.loss_anomaly))
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
        print(f"Precision: {metrics.precision_score(['N']*len(pred_normal)+['A']*len(pred_anomaly), pred_normal+pred_anomaly, pos_label='N')}")
        print(f"Recall: {metrics.recall_score(['N']*len(pred_normal)+['A']*len(pred_anomaly), pred_normal+pred_anomaly, pos_label='N')}")
        print(f"F1: {metrics.f1_score(['N']*len(pred_normal)+['A']*len(pred_anomaly), pred_normal+pred_anomaly, pos_label='N')}")