import models.spyderNet
import torch.nn as nn
from torch import *
import numpy as np
import config.data
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.optim as optim

stockData = config.data.downloadData()
sPyder = models.spyderNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(sPyder.parameters(), lr=stockData["training"]["learning_rate"], betas=(0.9,0.98), eps=1e-9) #try dif optimizer?
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=stockData["training"]["scheduler_step_size"], gamma=0.1)
trainDataLoader = DataLoader(datasetTrain, batch_size=config["training"]["batch_size"], shuffle=True)
valDataLoader = DataLoader(datasetVal, batch_size=config["training"]["batch_size"], shuffle=True)

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
    


#Normalize //??
scaler = Normalizer()
normalized_data_close_price = scaler.fit_transform(data_close_price)


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

        x = x.to(stockData["training"]["device"])
        y = y.to(stockData["training"]["device"])
        
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
  