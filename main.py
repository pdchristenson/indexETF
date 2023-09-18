import ssl
# orig_sslsocket_init = ssl.SSLSocket.__init__
# ssl.SSLSocket.__init__ = lambda *args, cert_reqs=ssl.CERT_NONE, **kwargs: orig_sslsocket_init(*args, cert_reqs=ssl.CERT_NONE, **kwargs)
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
from alpha_vantage.timeseries import TimeSeries 

# responseAPI = requests.get('https://alphavantage.co/support/GF2DSZBYDT4AOOC7')
# print(responseAPI.status_code)
myKey = 'GF2DSZBYDT4AOOC7'

# r = requests.get()
# symbol = input('what would you like to model')
# config = {
#     "alpha_vantage": {
#         "key": myKey,
#         "symbol": "QQQ",#may want to make this a variable toUpper for dynamic screening?
#         "outputsize": "full",
#         "key_adjusted_close": "5. adjusted close",
#     },
#     "data": {
#         "window_size": 20,
#         "train_split_size": 0.9,
#     },
#     "plots": {
#         "xticks_interval": 126, #show a date every 6 trading months
#         "color_actual": "#001f3f",
#         "color_train": "#3D9970",
#         "color_val": "#0074D9",
#         "color_pred_train": "#3D9970",
#         "color_pred_val": "#0074D9",
#         "color_pred_test": "#FF4136",
#     },
#     "model": {
#         "input_size": 1, #b/c youre only using 1 feature, close price
#         "num_lstm_layers": 4, #2,
#         "lstm_size": 128,#32
#         "dropout": 0.15,
#     },
#     "training": {
#         "device": "cpu", #or cuda
#         "batch_size": 128, #5948
#         "num_epoch": 15,
#         "learning_rate": 0.01, #what does adjusting this do?
#         "scheduler_step_size": 20,
#         "patience": 10,
#     }
# }


# def download_data(config):
#     # ts = TimeSeries(key=config["alpha_vantage"]["key"])
#     ts = TimeSeries(key=myKey)#, treat_info_as_error=False)
#     # data, meta_data = ts.get_intraday(config["alpha_vantage"]["symbol"], interval='15min', outputsize='full')
#     # data['4. close'].plot()
#     # plt.title('TEST')
#     # plt.show()
#     data, meta_data = ts.get_daily_adjusted(config["alpha_vantage"]["symbol"], outputsize=config["alpha_vantage"]["outputsize"])

#     data_date = [date for date in data.keys()]
#     print("data_date="+str(data_date)) #testing
#     data_date.reverse()

#     data_close_price = [float(data[date][config["alpha_vantage"]["key_adjusted_close"]]) for date in data.keys()]
#     data_close_price.reverse()
#     data_close_price = np.array(data_close_price)

#     num_data_points = len(data_date)
#     display_date_range = "from " + data_date[0] + " to " + data_date[num_data_points-1]
#     print("Number data points", num_data_points, display_date_range)

#     return data_date, data_close_price, num_data_points, display_date_range

# data_date, data_close_price, num_data_points, display_date_range = download_data(config)

# plot

fig = figure(figsize=(25, 5), dpi=80)
# fig.patch.set_facecolor((1.0, 1.0, 1.0))
fig.set_facecolor((1.0, 1.0, 1.0))
plt.plot(data_date, data_close_price, color=config["plots"]["color_actual"])
xticks = [data_date[i] if ((i%config["plots"]["xticks_interval"]==0 and (num_data_points-i) > config["plots"]["xticks_interval"]) or i==num_data_points-1) else None for i in range(num_data_points)] # make x ticks nice
x = np.arange(0,len(xticks))
plt.xticks(x, xticks, rotation='vertical')
plt.title("Daily close price for " + config["alpha_vantage"]["symbol"] + ", " + display_date_range)
# plt.grid(b=None, which='major', axis='y', linestyle='--')
plt.grid(visible=True, which='major', axis='y', linestyle='--') #might wanna make false?

print('through')

# plt.show()


# class Normalizer():
#     def __init__(self):
#         self.mu = None
#         self.sd = None

#     def fit_transform(self, x):
#         self.mu = np.mean(x, axis=(0), keepdims=True)
#         self.sd = np.std(x, axis=(0), keepdims=True)
#         normalized_x = (x-self.mu)/self.sd
#         return normalized_x
 
#     def inverse_transform(self, x):
#         return (x*self.sd) +self.mu
    


#Normalize //??
scaler = Normalizer()
normalized_data_close_price = scaler.fit_transform(data_close_price)

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
toPlotDataYTrain = np.zeros(num_data_points)
toPlotDataYVal = np.zeros(num_data_points)

toPlotDataYTrain[config["data"]["window_size"]:splitIndex+config["data"]["window_size"]] = scaler.inverse_transform(dataYTrain)
toPlotDataYVal[splitIndex+config["data"]["window_size"]:] = scaler.inverse_transform(dataYVal)

toPlotDataYTrain = np.where(toPlotDataYTrain == 0, None, toPlotDataYTrain)
toPlotDataYVal = np.where(toPlotDataYVal == 0, None, toPlotDataYVal)

#plots

fig = figure(figsize=(25,5), dpi=80)
fig.set_facecolor((1.0,1.0,1.0))
plt.plot(data_date, toPlotDataYTrain, label="Prices (train)", color=config["plots"]["color_train"])
plt.plot(data_date, toPlotDataYVal, label="Prices (validation)", color=config["plots"]["color_val"])
xticks = [data_date[i] if((i%config["plots"]["xticks_interval"]==0 and (num_data_points-i) > config["plots"]["xticks_interval"]) or i==num_data_points-1) else None for i in range(num_data_points)]
# ^make xticks nice??
x = np.arange(0,len(xticks))
plt.xticks(x, xticks, rotation='vertical')
plt.title(f'Daily close prices for {config["alpha_vantage"]["symbol"]} - showing train and val data')
# plt.grid(b=None, which='major', axis='y', linestyle='--')
plt.grid(visible=True, which='major', axis='y', linestyle='--')

plt.legend()
# plt.show()

class TimeSeriesDataset(Dataset):

    def __init__(self, x, y):
        x = np.expand_dims(x,
                           2) # only have one feature so you need to convert 'x' into [batch, sequence, features] for LSTM
        self.x = x.astype(np.float32)
        self.y = x.astype(np.float32)

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return (self.x[idx], self.y[idx])
    
datasetTrain = TimeSeriesDataset(dataXTtrain, dataYTrain)
datasetVal = TimeSeriesDataset(dataXVal, dataYVal)

print(f'Train data shape X: {datasetTrain.x.shape} Y: {datasetTrain.y.shape}')
print(f'Validation data shape X: {datasetVal.x.shape} Y: {datasetVal.y.shape}')

trainDataLoader = DataLoader(datasetTrain, batch_size=config["training"]["batch_size"], shuffle=True)
valDataLoader = DataLoader(datasetVal, batch_size=config["training"]["batch_size"], shuffle=True)

# class LSTMMODEL(nn.Module):
#     def __init__(self, input_size=1, hidden_layer_size=64, num_layers=2, output_size=1, dropout = 0.2):
#         super().__init__()
#         self.hidden_layer_size = hidden_layer_size

#         self.linear_1 = nn.Linear(input_size, hidden_layer_size)
#         self.relu = nn.ReLU()
#         self.lstm = nn.LSTM(hidden_layer_size, hidden_size = hidden_layer_size, num_layers=num_layers,
#                             batch_first=True)
#         self.dropout = nn.Dropout(dropout)
#         self.linear_2 = nn.Linear(num_layers*hidden_layer_size, output_size)
#         #self.hidden = (torch.zeros(1,1,hidden_size), torch.zeros(1,1,hidden_size)) 
#         # self.hidden uses both h_n and c_n
#         self.init_weights()

#     def init_weights(self):
#         for name, param in self.lstm.named_parameters():
#             if 'bias' in name:
#                 nn.init.constant_(param, 0.0)
#             elif 'weight_ih' in name:
#                 nn.init.kaiming_normal_(param)
#             elif 'weight_hh' in name:
#                 nn.init.orthogonal_(param)
    
#     def forward(self, x):
#         batchsize = x.shape[0]
#         # print(f' LINE 232: batchSize: {batchsize}')
        
        
#         #layer1
#         x =self.linear_1(x)
#         x = self.relu(x)
#         # x = self.F.relu(x)

#         #LSTM layer'
#         lstm_out, (h_n, c_n) = self.lstm(x)

#         #reshape output from hidden cell into [batch, features] for linear_2
#         x = h_n.permute(1,0,2).reshape(batchsize, -1)
        
#         #layer 2
#         x = self.dropout(x)
#         predictions = self.linear_2(x)
#         return predictions[:,-1]
    
# def run_epoch(dataloader, is_training=False):
#     epochLoss = 0
#     print('training')
#     if is_training:
#         print('training now!!!')
#         model.train()
#     else:
#         model.eval()
    
#     for idx, (x,y) in enumerate(dataloader):
#         if is_training:
#             optimizer.zero_grad()

#         batchsize = x.shape[0]

#         x = x.to(config["training"]["device"])
#         y = y.to(config["training"]["device"])
        
#         out = model(x)
#         loss = criterion(out.contiguous(), y.contiguous())

#         if is_training:
#             loss.backward()
#             optimizer.step()
        
#         epochLoss += (loss.detach().item()/batchsize)

#     lr = scheduler.get_last_lr()[0]
   

#     return epochLoss, lr

trainDataLoader = DataLoader(datasetTrain, batch_size=config["training"]["batch_size"], shuffle=True)
valDataLoader = DataLoader(datasetVal, batch_size=config["training"]["batch_size"], shuffle=True)

model = LSTMMODEL(input_size=config["model"]["input_size"], hidden_layer_size=config["model"]["lstm_size"],
                  num_layers=config["model"]["num_lstm_layers"], output_size=1, dropout=config["model"]["dropout"])
model = model.to(config["training"]["device"])

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"], betas=(0.9,0.98), eps=1e-9) #try dif optimizer?
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config["training"]["scheduler_step_size"], gamma=0.1)
bestLoss = float(1000000000.0000000000)
epNoImprovement = 0


# # TRAINING
# for epoch in range(config["training"]["num_epoch"]):
    
#     loss_train, lr_train = run_epoch(trainDataLoader, is_training=True)
#     loss_val, lr_val = run_epoch(valDataLoader, is_training=True)
#     scheduler.step()

#     print('Epoch {}/{} | loss train {:.6f} | lr: {:.6f}'
#             .format(epoch+1, config["training"]["num_epoch"], loss_train, loss_val, lr_train))


#     if loss_train < bestLoss:
#         bestLoss = loss_train
#         epNoImprovement = 0
#     else:
#         print(f'ENI={epNoImprovement}    bestLoss: {bestLoss}')
#         epNoImprovement += 1
        
#     #if no improvement for a certain number of epochs then
#     if epNoImprovement >= config["training"]["patience"]:
#         print(f'EARLY STOP TRIGGERED @ {epoch+1}')
#         break
   

   

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
 
predicted_val = np.array([])

for idc, (x,y) in enumerate(valDataLoader):
    x = x.to(config["training"]["device"])
    out = model(x)
    out = out.cpu().detach().numpy()
    predicted_val = np.concatenate((predicted_val, out))

#prep for plotting

toPlotDataYTrainPred = np.zeros(num_data_points)
toPlotDataYValPred = np.zeros(num_data_points)

toPlotDataYTrainPred[config["data"]["window_size"]:splitIndex+config["data"]["window_size"]]= scaler.inverse_transform(predicted_train)
toPlotDataYValPred[splitIndex+config["data"]["window_size"]:] = scaler.inverse_transform(predicted_val)

toPlotDataYTrainPred = np.where(toPlotDataYTrainPred == 0, None, toPlotDataYTrainPred)
toPlotDataYValPred = np.where(toPlotDataYValPred == 0, None, toPlotDataYValPred)

#plots
fig = figure(figsize=(25,5), dpi=80)
fig.set_facecolor((1.0,1.0,1.0))
plt.plot(data_date, data_close_price, label="Actual Prices", color=config["plots"]["color_actual"])
plt.plot(data_date, toPlotDataYTrainPred, label="Predicted prices (train)", color=config["plots"]["color_pred_train"])
plt.plot(data_date, toPlotDataYValPred, label="Predicted prices (val)", color=config["plots"]["color_pred_val"])
plt.title("Compare pred prices to actual prices")
xticks = [data_date[i] if ((i%config["plots"]["xticks_interval"]==0 and (num_data_points-i)>config["plots"]["xticks_interval"]) or i==num_data_points-1) else None for i in range(num_data_points)] #make x ticks nice?
x = np.arange(0,len(xticks))
plt.xticks(x, xticks, rotation='vertical')
# plt.grid(b=None, which='major', axis='y', linestyle='--')
plt.grid(visible=True, which='major', axis='y', linestyle='--')

plt.legend()
plt.show()

#prep data for zoomed plot

toPlotDataYValSubset = scaler.inverse_transform(dataYVal)
toPlotPredVal = scaler.inverse_transform(predicted_val)
toPlotDataDate = data_date[splitIndex+config["data"]["window_size"]:]

#plots
fig = figure(figsize=(25,5), dpi=80)
fig.set_facecolor((1.0,1.0,1.0))
plt.plot(toPlotDataDate, toPlotDataYValSubset, label="Actual Prices", color=config["plots"]["color_actual"])
plt.plot(toPlotDataDate, toPlotPredVal, label="Predicted prices (val)", color=config["plots"]["color_pred_val"])
plt.title("Zoom into predictions - compare with actuals")
xticks = [toPlotDataDate[i] if ((i%int(config["plots"]["xticks_interval"]/5)==0 and (len(toPlotDataDate)-i)> config["plots"]["xticks_interval"]/6) or i==len(toPlotDataDate)-1)else None for i in range(len(toPlotDataDate))] #MAKE X TICKS NICE
xs = np.arange(0,len(xticks))
plt.xticks(xs, xticks, rotation='vertical')
# plt.grid(b=None, which='major', axis='y', linestyle='--')
plt.grid(visible=True, which='major', axis='y', linestyle='--')

# print("Training loss: ", train)
plt.legend()
plt.show()

#prdict closing price of next trading day

model.eval()

x = torch.tensor(dataXUnseen).float().to(config["training"]["device"]).unsqueeze(1) #this is dType and shape required [batch, sequence, feature]
x= x.unsqueeze(0)
prediction = model(x)
prediction = prediction.cpu().detach().numpy()
print(f'prediction is: {prediction}')

#prep plots
plot_range = 10
toPlotDataYVal = np.zeros(plot_range)
toPlotDataYValPred = np.zeros(plot_range)
toPlotDataYTestPred = np.zeros(plot_range)

toPlotDataYVal[:plot_range-1] = scaler.inverse_transform(dataYVal)[-plot_range+1:]
toPlotDataYValPred[:plot_range-1] = scaler.inverse_transform(predicted_val)[-plot_range+1:]

toPlotDataYTestPred[:plot_range-1] = scaler.inverse_transform(prediction)

toPlotDataYVal = np.where(toPlotDataYVal == 0, None, toPlotDataYVal)
toPlotDataYValPred = np.where(toPlotDataYValPred == 0, None, toPlotDataYValPred)
toPlotDataYTestPred = np.where(toPlotDataYTestPred == 0, None, toPlotDataYTestPred)

#PLOTS
plotDateTest = data_date[-plot_range+1:]
plotDateTest.append("tomorrow")

fig = figure(figsize=(25,5), dpi=80)
fig.set_facecolor((1.0,1.0,1.0))
plt.plot(plotDateTest, toPlotDataYVal, label="Actual prices", marker=".", markersize=10, color=config["plots"]["color_actual"])
plt.plot(plotDateTest, toPlotDataYValPred, label="Past predicted prices", marker=".", markersize=10, color=config["plots"]["color_pred_val"])
plt.plot(plotDateTest, toPlotDataYTestPred, label="Predicted price for tm", marker=".", markersize=20, color=config["plots"]["color_pred_test"])
plt.title("Predicting close price of next trading day")
plt.grid(visible=True, which='major', axis='y', linestyle='--')

plt.legend()
plt.show()

print(f'Predicted close price of next trading day for ${config["alpha_vantage"]["symbol"]}: ${round(toPlotDataYTestPred[plot_range-2], 2)}')





