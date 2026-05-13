import pandas as pd
import numpy as np
import cv2
import os
from sklearn.metrics import f1_score, roc_auc_score

DATA_PATH = './data/'   # Defining path - Data should be downloaded from Google Drive link in README -> Extract as is so 2 CSVs (Train & Valid) in /Data/ along with the CheXpert-v1.0-small folder containing train & valid folders

def load_chexpert_labels(csv_name='train.csv'):   # Loads CheXpert CSV and formats labels
    print("Loading CheXpert metadata...")
    df = pd.read_csv(os.path.join(DATA_PATH, csv_name))
    
    # Fill NAs and Uncertain (-1) labels with 0 for simplicity
    df = df.fillna(0)
    df = df.replace(-1, 0) 
    
    # Extract Paths and the target pathology
    target_pathology = 'Pleural Effusion'
    
    data_list = []
    for index, row in df.iterrows():
        img_path = os.path.join(DATA_PATH, row['Path'])
        label = int(row[target_pathology])
        data_list.append((img_path, label))
        
    return data_list

def calculate_metrics(y_true, y_pred, y_prob):  # Calc F1 and AUC-ROC for evaluation
    if len(np.unique(y_true)) > 1: # AUC needs at least one positive and one negative sample
        auc = roc_auc_score(y_true, y_prob)
    else:
        auc = 0.0
    f1 = f1_score(y_true, y_pred)
    return f1, auc
