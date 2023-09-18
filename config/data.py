import alpha_vantage.timeseries as TimeSeries
import numpy as np


myKey = ''

config = {
    "alpha_vantage": {
        "key": myKey,
        "symbol": "SPY",#may want to make this a variable toUpper for dynamic screening?
        "outputsize": "full",
        "key_adjusted_close": "4. close",
        "key_adjusted_open": "1. open",
    },
    "data": {
        "window_size": 20,
        "train_split_size": 0.7,
    },
    "plots": {
        "xticks_interval": 126, #show a date every 6 trading months
        "color_actual": "#001f3f",
        "color_train": "#3D9970",
        "color_val": "#0074D9",
        "color_pred_train": "#3D9970",
        "color_pred_val": "#0074D9",
        "color_pred_test": "#FF4136",
    },
    "model": {
        "input_size": 2, #b/c youre only using 1 feature, close price
        "num_lstm_layers": 4, #2,
        "lstm_size": 128,#32
        "dropout": 0.15,
    },
    "training": {
        "device": "cpu", #or cuda
        # "batch_size": x, #5948
        "num_epoch": 15,
        "learning_rate": 0.01, #what does adjusting this do?
        "scheduler_step_size": 20,
        "patience": 10,
    }
}

def downloadData():
    # ts = TimeSeries(key=config["alpha_vantage"]["key"])
    # ts = TimeSeries(key=myKey)#, treat_info_as_error=False)
    ts = TimeSeries.TimeSeries(key=myKey)
    # data, meta_data = ts.get_intraday(config["alpha_vantage"]["symbol"], interval='15min', outputsize='full')
    # data['4. close'].plot()
    # plt.title('TEST')
    # plt.show()
    
    
    # data, meta_data = ts.get_daily_adjusted(config["alpha_vantage"]["symbol"], outputsize=config["alpha_vantage"]["outputsize"])

    data, meta_data = ts.get_daily(config["alpha_vantage"]["symbol"], outputsize=config["alpha_vantage"]["outputsize"])

    print(data)
    data_date = [date for date in data.keys()]
    print("data_date="+str(data_date)) #testing
    data_date.reverse()

    data_close_price = [float(data[date][config["alpha_vantage"]["key_adjusted_close"]]) for date in data.keys()]
    data_open_price = [float(data[date][config["alpha_vantage"]["key_adjusted_open"]]) for date in data.keys()]
    data_open_price.reverse()
    data_open_price = np.array(data_open_price)
    data_close_price.reverse()
    data_close_price = np.array(data_close_price)

    num_data_points = len(data_date)
    display_date_range = "from " + data_date[0] + " to " + data_date[num_data_points-1]
    print("Number data points", num_data_points, display_date_range)

    return data_date, data_open_price, data_close_price, num_data_points, display_date_range

data_date, data_close_price, num_data_points, display_date_range = downloadData(config)