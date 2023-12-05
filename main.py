import ssl
import os
import math
import random 
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure 
import tkinter as tk
from PIL import ImageTk, Image
import models.spyderNet
from alpha_vantage.timeseries import TimeSeries 
import config.data
import training.train
import plot.plot

model = models.spyderNet
configData = config.data.downloadData()
# datasetTrain = TimeSeriesDataset(dataXTtrain, dataYTrain);
# datasetVal = TimeSeriesDataset(dataXVal, dataYVal)

# print(f'Train data shape X: {datasetTrain.x.shape} Y: {datasetTrain.y.shape}')
# print(f'Validation data shape X: {datasetVal.x.shape} Y: {datasetVal.y.shape}')

#*****************MAY NOT NEED*****************
datasetTrain = training.train.datasetTrain
datasetVal = training.train.datasetVal



#*****************MAY NOT NEED*****************

trainDataLoader = DataLoader(datasetTrain, batch_size=config["training"]["batch_size"], shuffle=True)
valDataLoader = DataLoader(datasetVal, batch_size=config["training"]["batch_size"], shuffle=True)
#reinitialize dataloader so data doesnt shuffle, so you can plot values by date   
trainDataLoader = DataLoader(datasetTrain, batch_size=config["training"]["batch_size"], shuffle=False)
valDataLoader = DataLoader(datasetVal, batch_size=config["training"]["batch_size"], shuffle=False)

model.eval()

#predict on training data to see how the model learned and memorized

predicted_train = np.array([])

for idx, (x,y) in enumerate(trainDataLoader):
    x = x.to(config["training"]["device"])
    out = model(x)
    out = out.cpu().detach().numpy()
    predicted_train = np.concatenate((predicted_train, out))

#predict on val data to see model performance
 
predicted_val = np.array([]) #prob not in plot? preds should be in main
#something like trainPredictionPlot = plot.predTrainPlot() -- 
# valPredictionPlot = plot.predValPlot()

for idc, (x,y) in enumerate(valDataLoader):
    x = x.to(config["training"]["device"])
    out = model(x)
    out = out.cpu().detach().numpy()
    predicted_val = np.concatenate((predicted_val, out))

model.eval()
dataXUnseen = training.train.dataXUnseen
toPlotDataYTestPred = plot.plot.toPlotDataYTestPred
plot_range = plot.plot.plot_range
x = torch.tensor(dataXUnseen).float().to(config["training"]["device"]).unsqueeze(1) #this is dType and shape required [batch, sequence, feature]
x= x.unsqueeze(0)
prediction = model(x)
prediction = prediction.cpu().detach().numpy() # need cuda
print(f'prediction is: {prediction}')


print(f'Predicted close price of next trading day for ${config["alpha_vantage"]["symbol"]}: ${round(toPlotDataYTestPred[plot_range-2], 2)}')





