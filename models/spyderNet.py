import torch.nn as nn
import numpy as np


class LSTMMODEL(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=64, num_layers=2, output_size=1, dropout = 0.2):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.linear_1 = nn.Linear(input_size, hidden_layer_size)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(hidden_layer_size, hidden_size = hidden_layer_size, num_layers=num_layers,
                            batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(num_layers*hidden_layer_size, output_size)
        #self.hidden = (torch.zeros(1,1,hidden_size), torch.zeros(1,1,hidden_size)) 
        # self.hidden uses both h_n and c_n
        self.init_weights()

    def init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
    
    def forward(self, x):
        batchsize = x.shape[0]
        # print(f' LINE 232: batchSize: {batchsize}')
        
        
        #layer1
        x =self.linear_1(x)
        x = self.relu(x)
        # x = self.F.relu(x)

        #LSTM layer'
        lstm_out, (h_n, c_n) = self.lstm(x)

        #reshape output from hidden cell into [batch, features] for linear_2
        x = h_n.permute(1,0,2).reshape(batchsize, -1)
        
        #layer 2
        x = self.dropout(x)
        predictions = self.linear_2(x)
        return predictions[:,-1]


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