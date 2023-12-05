import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np
from config import data
import training.train
#need to fix to import config data
#will abstract more, still need to fix order of code, and further initialize into 
#classes/scripts with more defined functions
splitIndex = training.train.splitIndex

configData = data.config
etfData = data.downloadData(configData)
scaler = training.train.Normalizer()

dataYVal = training.train.dataYVal # plot shouldn't need these variables in it's script.
dataYTrain = training.train.dataYTrain # ^^^^^^^^^^^^^^^^^^^^^

fig = figure(figsize=(25, 5), dpi=80)
# fig.patch.set_facecolor((1.0, 1.0, 1.0))
fig.set_facecolor((1.0, 1.0, 1.0))
plt.plot(etfData.data_date, etfData.data_close_price, color=etfData.config["plots"]["color_actual"])
xticks = [etfData.data_date[i] if ((i%etfData.config["plots"]["xticks_interval"]==0 and (etfData.num_data_points-i) > etfData.config["plots"]["xticks_interval"]) or i==etfData.num_data_points-1) else None for i in range(etfData.num_data_points)] # make x ticks nice
x = np.arange(0,len(xticks))
plt.xticks(x, xticks, rotation='vertical')
plt.title("Daily close price for " + etfData.config["alpha_vantage"]["symbol"] + ", " + etfData.display_date_range)
# plt.grid(b=None, which='major', axis='y', linestyle='--')
plt.grid(visible=True, which='major', axis='y', linestyle='--') #might wanna make false?

             
#prep for plotting
toPlotDataYTrain = np.zeros(etfData.num_data_points)
toPlotDataYVal = np.zeros(etfData.num_data_points)

toPlotDataYTrain[etfData.config["data"]["window_size"]:etfData.splitIndex+etfData.config["data"]["window_size"]] = scaler.inverse_transform(dataYTrain)
toPlotDataYVal[splitIndex+etfData.config["data"]["window_size"]:] = scaler.inverse_transform(dataYVal) # split the index in training, dont need to do it here

toPlotDataYTrain = np.where(toPlotDataYTrain == 0, None, toPlotDataYTrain)
toPlotDataYVal = np.where(toPlotDataYVal == 0, None, toPlotDataYVal)

#plots

fig = figure(figsize=(25,5), dpi=80)
fig.set_facecolor((1.0,1.0,1.0))
plt.plot(etfData.data_date, toPlotDataYTrain, label="Prices (train)", color=etfData.config["plots"]["color_train"])
plt.plot(etfData.data_date, toPlotDataYVal, label="Prices (validation)", color=etfData.config["plots"]["color_val"])
xticks = [etfData.data_date[i] if((i%etfData.config["plots"]["xticks_interval"]==0 and (etfData.num_data_points-i) > etfData.config["plots"]["xticks_interval"]) or i==etfData.num_data_points-1) else None for i in range(etfData.num_data_points)]
# ^make xticks nice??
x = np.arange(0,len(xticks))
plt.xticks(x, xticks, rotation='vertical')
plt.title(f'Daily close prices for {etfData.config["alpha_vantage"]["symbol"]} - showing train and val data')
# plt.grid(b=None, which='major', axis='y', linestyle='--')
plt.grid(visible=True, which='major', axis='y', linestyle='--')

plt.legend()
# plt.show()


#prep for plotting
###*************************move to main, and/or multiple plot objects in main calling a plot function on each instance

toPlotDataYTrainPred = np.zeros(etfData.num_data_points)
toPlotDataYValPred = np.zeros(etfData.num_data_points)

toPlotDataYTrainPred[etfData.config["data"]["window_size"]:splitIndex+etfData.config["data"]["window_size"]]= scaler.inverse_transform(predicted_train)
toPlotDataYValPred[splitIndex+etfData.config["data"]["window_size"]:] = scaler.inverse_transform(predicted_val)

toPlotDataYTrainPred = np.where(toPlotDataYTrainPred == 0, None, toPlotDataYTrainPred)
toPlotDataYValPred = np.where(toPlotDataYValPred == 0, None, toPlotDataYValPred)
###*************************move to main, and/or multiple plot objects in main calling a plot function on each instance



#plots
fig = figure(figsize=(25,5), dpi=80)
fig.set_facecolor((1.0,1.0,1.0))
plt.plot(etfData.data_date, etfData.data_close_price, label="Actual Prices", color=etfData.config["plots"]["color_actual"])
plt.plot(etfData.data_date, toPlotDataYTrainPred, label="Predicted prices (train)", color=etfData.config["plots"]["color_pred_train"])
plt.plot(etfData.data_date, toPlotDataYValPred, label="Predicted prices (val)", color=etfData.config["plots"]["color_pred_val"])
plt.title("Compare pred prices to actual prices")
xticks = [etfData.data_date[i] if ((i%etfData.config["plots"]["xticks_interval"]==0 and (etfData.num_data_points-i)>etfData.config["plots"]["xticks_interval"]) or i==etfData.num_data_points-1) else None for i in range(etfData.num_data_points)] #make x ticks nice?
x = np.arange(0,len(xticks))
plt.xticks(x, xticks, rotation='vertical')
# plt.grid(b=None, which='major', axis='y', linestyle='--')
plt.grid(visible=True, which='major', axis='y', linestyle='--')

plt.legend()
plt.show()

#prep data for zoomed plot

toPlotDataYValSubset = scaler.inverse_transform(dataYVal)
toPlotPredVal = scaler.inverse_transform(predicted_val)
toPlotDataDate = etfData.data_date[splitIndex+etfData.config["data"]["window_size"]:]

#plots
fig = figure(figsize=(25,5), dpi=80)
fig.set_facecolor((1.0,1.0,1.0))
plt.plot(toPlotDataDate, toPlotDataYValSubset, label="Actual Prices", color=etfData.config["plots"]["color_actual"])
plt.plot(toPlotDataDate, toPlotPredVal, label="Predicted prices (val)", color=etfData.config["plots"]["color_pred_val"])
plt.title("Zoom into predictions - compare with actuals")
xticks = [toPlotDataDate[i] if ((i%int(etfData.config["plots"]["xticks_interval"]/5)==0 and (len(toPlotDataDate)-i)> etfData.config["plots"]["xticks_interval"]/6) or i==len(toPlotDataDate)-1)else None for i in range(len(toPlotDataDate))] #MAKE X TICKS NICE
xs = np.arange(0,len(xticks))
plt.xticks(xs, xticks, rotation='vertical')
# plt.grid(b=None, which='major', axis='y', linestyle='--')
plt.grid(visible=True, which='major', axis='y', linestyle='--')

# print("Training loss: ", train)
plt.legend()
plt.show()

#prdict closing price of next trading day

#prep plots
plot_range = 10
toPlotDataYVal = np.zeros(plot_range)
toPlotDataYValPred = np.zeros(plot_range)
toPlotDataYTestPred = np.zeros(plot_range)
dataYVal = training.train.dataYVal

toPlotDataYVal[:plot_range-1] = scaler.inverse_transform(dataYVal)[-plot_range+1:]
toPlotDataYValPred[:plot_range-1] = scaler.inverse_transform(predicted_val)[-plot_range+1:]

toPlotDataYTestPred[:plot_range-1] = scaler.inverse_transform(prediction)

toPlotDataYVal = np.where(toPlotDataYVal == 0, None, toPlotDataYVal)
toPlotDataYValPred = np.where(toPlotDataYValPred == 0, None, toPlotDataYValPred)
toPlotDataYTestPred = np.where(toPlotDataYTestPred == 0, None, toPlotDataYTestPred)

#PLOTS
plotDateTest = etfData.data_date[-plot_range+1:]
plotDateTest.append("tomorrow")

fig = figure(figsize=(25,5), dpi=80)
fig.set_facecolor((1.0,1.0,1.0))
plt.plot(plotDateTest, toPlotDataYVal, label="Actual prices", marker=".", markersize=10, color=etfData.config["plots"]["color_actual"])
plt.plot(plotDateTest, toPlotDataYValPred, label="Past predicted prices", marker=".", markersize=10, color=etfData.config["plots"]["color_pred_val"])
plt.plot(plotDateTest, toPlotDataYTestPred, label="Predicted price for tm", marker=".", markersize=20, color=etfData.config["plots"]["color_pred_test"])
plt.title("Predicting close price of next trading day")
plt.grid(visible=True, which='major', axis='y', linestyle='--')

plt.legend()
plt.show()
