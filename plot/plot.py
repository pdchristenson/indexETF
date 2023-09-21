import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np

#need to fix to import config data
#will abstract more, still need to fix order of code, and further initialize into 
#classes/scripts with more defined functions


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
