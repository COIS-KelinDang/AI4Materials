import os
import numpy as np
import torch
import pandas as pd
import csv
import xml.etree.ElementTree as ET
import shutil

# -----数据集检查-----
dataset_path = r'E:\PythonProject\AI4Materials\dataset'
mor_csv_path = os.path.join(dataset_path, 'Universal_Quantification_Results.csv')
main_perf_csv_path = os.path.join(dataset_path, '电导率汇总（主任务）.csv')
aux_perf_csv_path = os.path.join(dataset_path, '特征提取（辅助任务）.csv')
new_dataset_path = r'E:\PythonProject\AI4Materials\new_dataset'

aux_perf_csv = pd.read_csv(aux_perf_csv_path)
for i in range(aux_perf_csv.shape[0]):
    if aux_perf_csv.iloc[i, 2] == 'Surface':
        sample_id = aux_perf_csv.iloc[i, 0][2:]
        SE_file = aux_perf_csv.iloc[i, 3]
        BSE_file = aux_perf_csv.iloc[i, 4]
    else:
        pass