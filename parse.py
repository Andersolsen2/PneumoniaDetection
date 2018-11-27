#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 19:25:21 2018

@author: CecilieAndre
"""

import os 
import numpy as np 
import pandas as pd
from utilities import *


cwd = os.getcwd()
print (cwd)

os.chdir(cwd)

## Read labels
true_labels=pd.read_csv('./stage_2_train_labels.csv')
pred_labels=pd.read_csv('./submission_min_conf=0.95.csv')

## suidID

ID_true = true_labels['patientId']
ID_pred = pred_labels['patientId']


## PARSE
# Test labels
true_parsed = test(true_labels,ID_true,ID_pred)

# Predicted
pred_parsed = predicted(pred_labels, ID_pred)

## Calculate true positiv, true negativ, false positiv, etc. 
#  IOU treshhold
iou = 0.5
sum_of_all, True_positiv, True_negativ, False_positiv, False_negativ = calucalteTP_TN_FP_FN(pred_parsed, true_parsed, iou)


