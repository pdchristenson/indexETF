import models.spyderNet
from alpha_vantage.timeseries import TimeSeries
import torch.nn as nn
from torch import *
from torch.utils.data import Dataset
import numpy as np
import config.data
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.optim as optim
from config import data
from config.data import TimeSeriesDataset
import plot
import matplotlib as plt

etfData = config.data
data_close_price = etfData.data_close_price

def getRandomBatch():

    rnd = random.randrange(2,15)

    x = math.exp2(rnd) 
    return int(x)

class Normalizer():
    def __init__(self):
        self.mu = None
        self.sd = None

    def fit_transform(self, x):
        self.mu = np.mean(x, axis=(0), keepdims=True)
        self.sd = np.std(x, axis=(0), keepdims=True)
        normalized_x = (x-self.mu)/self.sd
        return normalized_x
 
    def inverse_transform(self, x):
        return (x*self.sd) +self.mu
  


def prepare_data_x(x, window_size):
    #windowing
    n_row = x.shape[0] - window_size +1
    output = np.lib.stride_tricks.as_strided(x, shape=(n_row, window_size), strides=(x.strides[0], x.strides[0]))
    return output[:-1], output[-1]


def prepare_data_y(x, window_size):
    #simple moving avg
    # output = np.convolve(x, np.ones(window_size), 'valid') / window_size

    #use next day as label
    output = x[window_size:]
    return output
#Normalize //??
scaler = Normalizer()
normalized_data_close_price = scaler.fit_transform(data_close_price)

dataX, dataXUnseen = prepare_data_x(normalized_data_close_price, window_size=config["data"]["window_size"])
dataY = prepare_data_y(normalized_data_close_price, window_size=config["data"]["window_size"])

# split dataset
splitIndex = int(dataY.shape[0]*config["data"]["train_split_size"])
dataXTtrain = dataX[:splitIndex]
dataXVal = dataX[splitIndex:]
dataYTrain = dataY[:splitIndex].reshape(-1)
dataYVal = dataY[splitIndex:].reshape(-1)
# dataYTrain = dataY[:splitIndex]
# dataYVal = dataY[splitIndex:]
#prep for plotting
toPlotDataYTrain = np.zeros(etfData.num_data_points)
toPlotDataYVal = np.zeros(etfData.num_data_points)

toPlotDataYTrain[etfData.config["data"]["window_size"]:etfData.splitIndex+etfData.config["data"]["window_size"]] = etfData.scaler.inverse_transform(dataYTrain)
toPlotDataYVal[splitIndex+etfData.config["data"]["window_size"]:] = scaler.inverse_transform(dataYVal) # split the index in training, dont need to do it here

toPlotDataYTrain = np.where(toPlotDataYTrain == 0, None, toPlotDataYTrain)
toPlotDataYVal = np.where(toPlotDataYVal == 0, None, toPlotDataYVal)



datasetTrain = TimeSeriesDataset(dataXTtrain, dataYTrain)
datasetVal = TimeSeriesDataset(dataXVal, dataYVal)

print(f'Train data shape X: {datasetTrain.x.shape} Y: {datasetTrain.y.shape}')
print(f'Validation data shape X: {datasetVal.x.shape} Y: {datasetVal.y.shape}')


sPyder = models.spyderNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(sPyder.parameters(), lr=etfData.config["training"]["learning_rate"], betas=(0.9,0.98), eps=1e-9) #try dif optimizer?
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=etfData.config["training"]["scheduler_step_size"], gamma=0.1)
trainDataLoader = DataLoader(datasetTrain, batch_size=config["training"]["batch_size"], shuffle=True)
valDataLoader = DataLoader(datasetVal, batch_size=config["training"]["batch_size"], shuffle=True)

  




def run_epoch(dataloader, is_training=False):
    epochLoss = 0
    print('training')
    if is_training:
        print('training now!!!')
        sPyder.train()
    else:
        sPyder.eval()
    
    for idx, (x,y) in enumerate(dataloader):
        if is_training:
            optimizer.zero_grad()

        batchsize = x.shape[0]

        x = x.to(etfData.config["training"]["device"])
        y = y.to(etfData.config["training"]["device"])
        
        out = sPyder(x)
        loss = criterion(out.contiguous(), y.contiguous())

        if is_training:
            loss.backward()
            optimizer.step()
        
        epochLoss += (loss.detach().item()/batchsize)

    lr = scheduler.get_last_lr()[0]
   

    return epochLoss, lr

# TRAINING
for epoch in range(config["training"]["num_epoch"]):
    
    loss_train, lr_train = run_epoch(trainDataLoader, is_training=True)
    loss_val, lr_val = run_epoch(valDataLoader, is_training=True)
    scheduler.step()

    print('Epoch {}/{} | loss train {:.6f} | lr: {:.6f}'
            .format(epoch+1, config["training"]["num_epoch"], loss_train, loss_val, lr_train))


    if loss_train < bestLoss:
        bestLoss = loss_train
        epNoImprovement = 0
    else:
        print(f'ENI={epNoImprovement}    bestLoss: {bestLoss}')
        epNoImprovement += 1
        
    #if no improvement for a certain number of epochs then
    if epNoImprovement >= config["training"]["patience"]:
        print(f'EARLY STOP TRIGGERED @ {epoch+1}')
        break
  