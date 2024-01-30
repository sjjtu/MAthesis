"""
Utility class to handle all preprocessing steps:
- load data
- standardise
- beat segmentation
- beat annotation
- split into train, validation and test data set
- save as csv
"""

import os
from pathlib import Path

import pandas as pd 
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

import wfdb
import wfdb.processing

### CONSTANT VARIABLES

# List of Patients
patients = ['100','101','102','103','104','105','106','107',
           '108','109','111','112','113','114','115','116',
           '117','118','119','121','122','123','124','200',
           '201','202','203','205','207','208','209','210',
           '212','213','214','215','217','219','220','221',
           '222','223','228','230','231','232','233','234']

# Normal Beat Symbols
normal = ['N', 'L', 'R']

# Data path
data = "data/mit-bih/"

# Window size = length of heartbeat sequence
window_size = 180

class Preprocessor:
    def __init__(self):
        self.seq_df = None
        self.label_df = None
    
    def _load_data(self):

        Path(data).mkdir(parents=True, exist_ok=True) # create dir if necessary

        if len(os.listdir(data))==0: # download data if not in folder
            wfdb.dl_database("mitdb", "data/mit-bih", annotators='all')
        
        def find_ann(peak, d1, d2, ann_list, ann_sym):
            if ((ann_list>int(peak - d1/2)) &  (ann_list<int(peak + d2/2))).sum() != 1:
                return "0 or more than 1 annotation"
            ind, = np.where((ann_list>int(peak - d1/2)) &  (ann_list<int(peak + d2/2)))[0]
            return ann_sym[ind]
        
        df = []
        ann_list = []
        skipping_count = 0
        for pat_id in patients:
            print('record number', pat_id)
            record = wfdb.rdrecord(data+pat_id,smooth_frames=True)
            ann = wfdb.rdann(data+pat_id, extension="atr")
            min_max_scaler = preprocessing.MinMaxScaler()
            lm = min_max_scaler.fit_transform(record.p_signal[:,0].reshape(-1,1)).squeeze()
            qrs = wfdb.processing.XQRS(lm, fs=360)
            qrs.detect()
            peaks = qrs.qrs_inds

            for i, peak in enumerate(peaks[1:-1]):
                d2 = qrs.qrs_inds[i+2] - peak  # we start at peaks[1] but i starts at 0
                d1 = peak - qrs.qrs_inds[i]
                start,end = peak-window_size//2, peak+window_size//2
                beat_type = find_ann(peak, d1, d2, ann.sample, ann.symbol)
                if beat_type=="0 or more than 1 annotation":
                        #print("skipping")
                        #print(i)
                        skipping_count += 1
                        continue
                        plt.figure()
                        plt.plot(lm[peak-d1:peak+d2])
                        plt.show()
                        break
                beat_type = "N" if beat_type in normal else "A"
                ann_list.append(beat_type)
                df.append(list(lm[start:end]))
                
        self.label_df = pd.DataFrame(ann_list, columns=["label"])
        self.seq_df =pd.DataFrame(df, columns=[f"t_{i}" for i in range(window_size)])
        
        print(f"skipping {skipping_count} beats")
        print(f"Total beats: {len(ann_list)} -- normal: {ann_list.count('N')}")
            
    def _train_test_val_split(self, val_ratio=.1, test_ratio=.1):
        normal_train, normal_test, y_train, y_test = train_test_split(self.seq_df[self.label_df.label=="N"], 
                                                                    self.label_df[self.label_df.label=="N"], 
                                                                    test_size=test_ratio, 
                                                                    random_state=1)
        normal_train, normal_val, y_train, y_val = train_test_split(normal_train, 
                                                                    y_train, 
                                                                    test_size=val_ratio, 
                                                                    random_state=1)
        


        anomalie_test, anomalie_val, anomalie_test_lables, anomalie_val_labels = train_test_split(self.seq_df[self.label_df.label!='N'],
                                                                                                  self.label_df[self.label_df.label!='N'],
                                                                                                  test_size=0.05,
                                                                                                  random_state=1)


        
        print(f"train set (only normal): {len(normal_train)}")
        print(f"val set (only normal): {len(normal_val)}")
        print(f"val set (only anomaly): {len(anomalie_val)}")
        print(f"test set (normal): {len(normal_test)}")
        print(f"test set (anomaly): {len(anomalie_test)}")
    
        pd.DataFrame(normal_train).to_csv(f"data/normal_train_{window_size}.csv")
        pd.DataFrame(y_train).to_csv(f"data/train_labels_{window_size}.csv")

        pd.DataFrame(normal_val).to_csv(f"data/normal_val_{window_size}.csv")
        pd.DataFrame(y_val).to_csv(f"data/val_labels_{window_size}.csv")

        pd.DataFrame(anomalie_val).to_csv(f"data/anomaly_val_{window_size}.csv")
        pd.DataFrame(anomalie_test_lables).to_csv(f"data/anomaly_val_labels_{window_size}.csv")

        pd.DataFrame(normal_test).to_csv(f"data/normal_test{window_size}.csv")
        pd.DataFrame(y_test).to_csv(f"data/normal_labels_{window_size}.csv")

        pd.DataFrame(anomalie_test).to_csv(f"data/anomalie_test{window_size}.csv")
        pd.DataFrame(anomalie_test_lables).to_csv(f"data/anomalie_labels_{window_size}.csv")
            
    def preprocess(self, val_ratio=.1, test_ratio=.1):
        self._load_data()
        self._train_test_val_split(val_ratio, test_ratio)
        return self.seq_df
        
        
if __name__ == "__main__":
    pp = Preprocessor()
    pp.preprocess()