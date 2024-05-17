# -*- coding: utf-8 -*-
#
# @Author: Yang Liu <yangliu.real@gmail.com>
# @Date: 2024/01/07

import statistics

# Group 1, seed: 21
#
# Model      IAV                LVC.cause          LVC.full           MVC                VID                VPC.full           VPC.semi
# -----      ---                ---------          --------           ---                ---                --------           --------
# BERT-base  55.56,41.67,47.62  42.86,20.69,27.91  47.14,62.21,53.63  95.83,79.31,86.79  56.72,35.19,43.43  62.04,78.35,69.25  40.00,6.670,11.43
# BERT-large 46.67,38.89,42.42  36.36,27.59,31.37  57.53,62.21,59.78  100.0,79.31,88.46  61.19,37.96,46.86  64.29,78.87,70.83  34.48,33.33,33.90

# Group 2, seed: 42
#
# Model      IAV                LVC.cause          LVC.full           MVC                VID                VPC.full           VPC.semi
# -----      ---                ---------          --------           ---                ---                --------           --------
# BERT-base  60.00,33.33,42.86  36.36,13.79,20.00  55.19,58.72,56.90  100.0,82.76,90.57  46.43,36.11,40.63  68.04,76.80,72.15  27.78,16.67,20.83
# BERT-large 42.86,33.33,37.50  12.50,10.34,11.32  52.60,47.09,49.69  100.0,82.76,90.57  60.66,34.26,43.79  64.44,79.38,71.13  46.15,40.00,42.86

# Group 3, seed: 84
#
# Model      IAV                LVC.cause          LVC.full           MVC                VID                VPC.full           VPC.semi
# -----      ---                ---------          --------           ---                ---                --------           --------
# BERT-base  66.67,38.89,49.12  60.00,20.69,30.77  54.04,62.21,57.84  92.00,79.31,85.19  54.05,37.04,43.96  62.75,79.90,70.29  100.0,3.330,6.450
# BERT-large 50.00,44.44,47.06  30.43,24.14,26.92  55.56,61.05,58.17  100.0,79.31,88.46  69.64,36.11,47.56  64.32,79.90,71.26  35.71,33.33,34.48

# creating a simple data - set
sample_bert_base = [55.56, 60.00, 66.67]
sample_bert_base = [41.67, 33.33, 38.89]
sample_bert_base = [47.62, 42.86, 49.12]

sample_bert_base = [42.86, 36.36, 60.00]
sample_bert_base = [20.69, 13.79, 20.69]
sample_bert_base = [27.91, 20.00, 30.77]

sample_bert_base = [47.14, 55.19, 54.04]
sample_bert_base = [62.21, 58.72, 62.21]
sample_bert_base = [53.63, 56.90, 57.84]

sample_bert_base = [95.83, 100.0, 92.00]
sample_bert_base = [79.31, 82.76, 79.31]
sample_bert_base = [86.79, 90.57, 85.19]

sample_bert_base = [56.72, 46.43, 54.05]
sample_bert_base = [35.19, 36.11, 37.04]
sample_bert_base = [43.43, 40.63, 43.96]

sample_bert_base = [62.04, 68.04, 62.75]
sample_bert_base = [78.35, 76.80, 79.90]
sample_bert_base = [69.25, 72.15, 70.29]

sample_bert_base = [40.00, 27.78, 100.0]
sample_bert_base = [6.670, 16.67, 3.330]
sample_bert_base = [11.43, 20.83, 6.450]

sample_all_bert_base = [
    [55.56, 60.00, 66.67],
    [41.67, 33.33, 38.89],
    [47.62, 42.86, 49.12],
    [42.86, 36.36, 60.00],
    [20.69, 13.79, 20.69],
    [27.91, 20.00, 30.77],
    [47.14, 55.19, 54.04],
    [62.21, 58.72, 62.21],
    [53.63, 56.90, 57.84],
    [95.83, 100.0, 92.00],
    [79.31, 82.76, 79.31],
    [86.79, 90.57, 85.19],
    [56.72, 46.43, 54.05],
    [35.19, 36.11, 37.04],
    [43.43, 40.63, 43.96],
    [62.04, 68.04, 62.75],
    [78.35, 76.80, 79.90],
    [69.25, 72.15, 70.29],
    [40.00, 27.78, 100.0],
    [6.670, 16.67, 3.330],
    [11.43, 20.83, 6.450],
    [64.80, 61.07, 63.86],
    [61.87, 60.87, 60.87],
    [63.60, 60.97, 62.33],
]

sample_bert_large = [46.67, 42.86, 50.00]
sample_bert_large = [38.89, 33.33, 44.44]
sample_bert_large = [42.42, 37.50, 47.06]

sample_bert_large = [36.36, 12.50, 30.43]
sample_bert_large = [27.59, 10.34, 24.14]
sample_bert_large = [31.37, 11.32, 26.92]

sample_bert_large = [57.53, 52.60, 55.56]
sample_bert_large = [62.21, 47.09, 61.05]
sample_bert_large = [59.78, 49.69, 58.17]

sample_bert_large = [100.0, 100.0, 100.0]
sample_bert_large = [79.31, 82.76, 79.31]
sample_bert_large = [88.46, 90.57, 88.46]

sample_bert_large = [61.19, 60.66, 69.64]
sample_bert_large = [37.96, 34.26, 36.11]
sample_bert_large = [46.86, 43.79, 47.56]

sample_bert_large = [64.29, 64.44, 64.32]
sample_bert_large = [78.87, 79.38, 79.90]
sample_bert_large = [70.83, 71.13, 71.26]

sample_bert_large = [34.48, 46.15, 35.71]
sample_bert_large = [33.33, 40.00, 33.33]
sample_bert_large = [33.90, 42.86, 34.48]

sample_all_bert_large = [
    [46.67, 42.86, 50.00],
    [38.89, 33.33, 44.44],
    [42.42, 37.50, 47.06],
    [36.36, 12.50, 30.43],
    [27.59, 10.34, 24.14],
    [31.37, 11.32, 26.92],
    [57.53, 52.60, 55.56],
    [62.21, 47.09, 61.05],
    [59.78, 49.69, 58.17],
    [100.0, 100.0, 100.0],
    [79.31, 82.76, 79.31],
    [88.46, 90.57, 88.46],
    [61.19, 60.66, 69.64],
    [37.96, 34.26, 36.11],
    [46.86, 43.79, 47.56],
    [64.29, 64.44, 64.32],
    [78.87, 79.38, 79.90],
    [70.83, 71.13, 71.26],
    [34.48, 46.15, 35.71],
    [33.33, 40.00, 33.33],
    [33.90, 42.86, 34.48],
    [63.85, 64.71, 64.02],
    [59.36, 64.38, 63.38],
    [61.53, 64.54, 63.70],
]

name_all = [
    "IAV-Precision",
    "IAV-Recall",
    "IAV-F1",
    "LVC.cause-Precision",
    "LVC.cause-Recall",
    "LVC.cause-F1",
    "LVC.full-Precision",
    "LVC.full-Recall",
    "LVC.full-F1",
    "MVC-Precision",
    "MVC-Recall",
    "MVC-F1",
    "VID-Precision",
    "VID-Recall",
    "VID-F1",
    "VPC.full-Precision",
    "VPC.full-Recall",
    "VPC.full-F1",
    "VPC.semi-Precision",
    "VPC.semi-Recall",
    "VPC.semi-F1",
    "Global-Micro-Precision",
    "Global-Micro-Recall",
    "Global-Micro-F1",
]

# Prints standard deviation
# xbar is set to default value of 1
for sample_all in [sample_all_bert_base, sample_all_bert_large]:
    for i in range(0, 24):
        print(
            "%s\tMean Value: %s (±%s)"
            % (
                name_all[i],
                round(statistics.fmean(sample_all[i]), ndigits=2),
                round(statistics.stdev(sample_all[i]), ndigits=2),
            )
        )
    print()
