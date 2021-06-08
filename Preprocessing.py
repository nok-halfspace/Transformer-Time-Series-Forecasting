import pandas as pd
import time
import numpy as np
import datetime
from icecream import ic

# encoding the timestamp data cyclically. See Medium Article.
def process_data(source):

    df = pd.read_csv(source)
        
    timestamps = [ts.split('+')[0] for ts in  df['timestamp']]
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

    return df

train_dataset = process_data('Data/train_raw.csv')
test_dataset = process_data('Data/test_raw.csv')

train_dataset.to_csv(r'Data/train_dataset.csv', index=False)
test_dataset.to_csv(r'Data/test_dataset.csv', index=False)
