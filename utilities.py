#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 15:37:29 2018

@author: CecilieAndre
"""

import os 
import numpy as np 
import pandas as pd


## PARSE
# Test labels
def test(true_labels,ID_true,ID_pred):
    true_parsed = []
    for i in range(0,len(ID_pred),1):
        for k in range(0,len(ID_true),1):
            if str(ID_pred[i])==str(ID_true[k]):
                #print('Yes')
                string = true_labels.iloc[k]
                true_parsed.append(
                                   [
                                           string['patientId'],
                                           float(string['x']),
                                           float(string['y']),
                                           float(string['width']),
                                           float(string['height']),
                                           float(string['Target'])
                                   ]
                                  )

    return  true_parsed


# Predicted
            
def predicted(pred_labels, ID_pred):
    dap = pred_labels['PredictionString']
    dip= dap.values.T.tolist()
    pred_parsed = []
    
    for i in range(0,len(ID_pred),1):
        stringf = 0 
        string = pred_labels.iloc[i]
        stringf = dip[i]
        if not str(stringf) =='nan':
            floats = [float(x) for x in stringf.split()]
            pred_parsed.append([string['patientId'],floats])
        else:
            pred_parsed.append([string['patientId'],['nan']])         
    
    
    return pred_parsed

## Define intersection over union


def IOU(x_true, y_true, width_true, height_true, x_pred, y_pred, width_pred, height_pred):
    
    
    # determine the (x1,y1,x2,y2)-coordinates of the intersection rectangle
    x11 = x_true - width_true/2
    y11 = y_true - height_true/2
    x12 = x_true + width_true/2
    y12 = y_true + height_true/2

    x21 = x_pred - width_pred/2
    y21 = y_pred - height_pred/2
    x22 = x_pred + width_pred/2
    y22 = y_pred + height_pred/2

    xA = np.maximum(x11, np.transpose(x21))
    yA = np.maximum(y11, np.transpose(y21))
    xB = np.minimum(x12, np.transpose(x22))
    yB = np.minimum(y12, np.transpose(y22))
    interArea = np.maximum((xB - xA + 1), 0) * np.maximum((yB - yA + 1), 0)

    boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
    boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)
    iou = interArea / (boxAArea + np.transpose(boxBArea) - interArea)
    return iou

## Calculate true positiv, true negativ, false positiv, etc. 
    

def calucalteTP_TN_FP_FN(pred_parsed, true_parsed, iou_conf):

    True_positiv = 0     
    True_negativ = 0
    False_positiv = 0     
    False_negativ = 0      
    
    detect_FP = np.zeros(len(pred_parsed)) 

    for i in range(0,len(pred_parsed),1):
        for k in range(0,len(true_parsed),1):
            if str(pred_parsed[i][0])==str(true_parsed[k][0]):
                if str(pred_parsed[i][1][0])=='nan' and str(true_parsed[k][1])=='nan':
                    detect_FP[i]=1
                    True_negativ +=1
                elif str(pred_parsed[i][1][0])=='nan'and str(true_parsed[k][1])!='nan':
                    detect_FP[i]=1
                    False_negativ +=1                
                else:
                    x_true = true_parsed[k][1]
                    y_true  = true_parsed[k][2]
                    width_true  = true_parsed[k][3]
                    height_true  = true_parsed[k][4]
                                   
                    x_pred = pred_parsed[i][1][1]
                    y_pred = pred_parsed[i][1][2]
                    width_pred = pred_parsed[i][1][3]
                    height_pred = pred_parsed[i][1][4]
                    
                    iou = IOU(x_true, y_true, width_true, height_true, x_pred, y_pred, width_pred, height_pred)
                    if iou > iou_conf :
                        True_positiv += 1
                    else:
                        False_positiv += 1



    sum_of_all = True_positiv+True_negativ+False_positiv+False_negativ
    print('True_positiv: '+ str(True_positiv) + ', True_negativ: '+ str(True_negativ)+ ', False_positiv: '+ str(False_positiv)+ ', False_negativ: '+ str(False_negativ))
    print('All: ' + str(sum_of_all))  
    
    return sum_of_all, True_positiv, True_negativ, False_positiv, False_negativ

