import pandas as pd
import datetime
import matplotlib.pyplot as plt
import numpy as np
from icecream import ic


def plot_data(df, x_label, y_label, x_col_name, y_col_name):
    fig, ax = plt.subplots()
    ax.set_ylabel(x_label, fontsize=12)
    ax.set_xlabel(y_label, fontsize=12)
    ax.plot(data_frame[x_col_name], data_frame[y_col_name])
    plt.show()

def process_crypto_data(source):

    df = pd.read_csv(source)
    df = df[["time_close", "price_close", "volume_traded", "trades_count"]]
    
    df['date_time'] = df.apply(lambda row: row.time_close.replace('T', ' ').split('.')[0], axis=1)
    print(df.head())
    # df = df[['date_time', 'sin_hour', 'cos_hour', 'sin_day', 'cos_day', 'sin_month', 'cos_month', 'price_close', 'volume_traded', 'trades_count']]
    
    # encoding the timestamp data cyclically
    timestamps = df['date_time']
    timestamps_hour = np.array([float(datetime.datetime.strptime(t, '%Y-%m-%d %H:%M:%S').hour) for t in timestamps])
    timestamps_day = np.array([float(datetime.datetime.strptime(t, '%Y-%m-%d %H:%M:%S').day) for t in timestamps])
    timestamps_month = np.array([float(datetime.datetime.strptime(t, '%Y-%m-%d %H:%M:%S').month) for t in timestamps])

    hours_in_day = 24
    days_in_month = 30
    month_in_year = 12

    df['sin_hour'] = np.sin(2*np.pi*timestamps_hour/hours_in_day)
    df['cos_hour'] = np.cos(2*np.pi*timestamps_hour/hours_in_day)
    df['sin_day'] = np.sin(2*np.pi*timestamps_day/days_in_month)
    df['cos_day'] = np.cos(2*np.pi*timestamps_day/days_in_month)
    df['sin_month'] = np.sin(2*np.pi*timestamps_month/month_in_year)
    df['cos_month'] = np.cos(2*np.pi*timestamps_month/month_in_year)
    
    df = df.fillna(-1) # specifically for weather_precip_path10min which has NaN
    ic(df)
    return df

if __name__ == '__main__':
    df = process_crypto_data('./Data/BTC_3hrs_History_20150101_20210427.csv')
    df.to_csv('./Data/processed_btc_3hrs_data.csv')
#     df = pd.read_csv('./Data/BTC_3hrs_History_20150101_20210427.csv')
#     print(df.shape[0])
#     print(df.shape)
#     print(df.size)