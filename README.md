# PneumoniaDetection
For the deep learning (DTU course 02456) project with Mask-RCNN regarding pneumonia detection and classification

# DATA
The data used is downloaded for the challenge RSNA Pneumonia Detection Challenge at Kaggle. This challenge is a two step process, thus we only use the stage 1 data set. It consist of two .zip files containing the training and test data set. A .csv file is also givin for the ground truth for the training data. Since, the stage 2 traning data set consist of the stage 1 train and test data set plus som more, the ground truth labels givin is parsed with the test data set from stage 1. Thus, the ground truth is obtained. 

## Mask - RCNN 

SETUP:
start with running the requriements.sh in the terminal. 
Download the data from Kaggle.

TRAINNING:
Start train.ipny and set up the paths.
Run train.ipny

TESTING: 
Run parse.ipynb to get main results.
