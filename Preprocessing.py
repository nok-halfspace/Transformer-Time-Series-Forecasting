import pandas as pd
import time
import numpy as np
import datetime
from icecream import ic

# the original features of the input data
"""
0 - sensor_id
1 - timestamp
2 - temperature
3 - humidity
4 - ohms
5 - moisture
6 - weather_humidity
7 - weather_pressure
8 - weather_temp_dew
9 - weather_temp_dry
10 - weather_wind_dir
11 - weather_wind_speed
12 - weather_wind_max
13 - weather_wind_min
14 - weather_precip_past10min
"""

source = "Data/sensor2.csv"

# Here I selected a subset of the data. As the original dataset had a lot of unusable samples, I selected a subset of sensors for training and testing.
sensors_train = [1,14,29,30,32,41,43,47]
sensors_test = [33, 44]
# I additionally chose intervals from these samples that I would use for training/testing.
sensor_interval = {1:(300,600), 14:(270,800), 29:(0,400), 30:(200, 600), 32:(100, 400), 33:(500,900), 41:(500, 750), 43: (1600, 2900), 44:(500,1400), 47:(900,1200)}


def process_data(source, sensors):

    df = pd.read_csv(source)
    df_filtered = pd.DataFrame()
    
    for i, sensor in enumerate(sensors):
        
        # triming each sensor to the desired start and end point of the chosen interval, as shown above.
        (start, end) = sensor_interval[sensor]
        new_sensor = df[df['sensor_id'] == sensor][start:end]
        
        """ Choose sensors to be removed from the list above """
        # Here I drop the features that I do not want to use, from the features 0-14 shown above. 
        # From my exploratory data analysis, I chose a subset of features to use as input to my model. 
        # Then I performed experiments adding/dropping features to get a better model.
        new_sensor = new_sensor.drop(new_sensor.columns[[2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]], axis=1) 
        # re-indexing the sensor with incrementing ids
        new_sensor['filtered_id'] = i+1

        # encoding the timestamp data cyclically. See Medium Article.
        timestamps = [ts.split('+')[0] for ts in  new_sensor['timestamp']]
        timestamps_hour = np.array([float(datetime.datetime.strptime(t, '%Y-%m-%d %H:%M:%S').hour) for t in timestamps])
        timestamps_day = np.array([float(datetime.datetime.strptime(t, '%Y-%m-%d %H:%M:%S').day) for t in timestamps])
        timestamps_month = np.array([float(datetime.datetime.strptime(t, '%Y-%m-%d %H:%M:%S').month) for t in timestamps])

        hours_in_day = 24
        days_in_month = 30
        month_in_year = 12

        new_sensor['sin_hour'] = np.sin(2*np.pi*timestamps_hour/hours_in_day)
        new_sensor['cos_hour'] = np.cos(2*np.pi*timestamps_hour/hours_in_day)
        new_sensor['sin_day'] = np.sin(2*np.pi*timestamps_day/days_in_month)
        new_sensor['cos_day'] = np.cos(2*np.pi*timestamps_day/days_in_month)
        new_sensor['sin_month'] = np.sin(2*np.pi*timestamps_month/month_in_year)
        new_sensor['cos_month'] = np.cos(2*np.pi*timestamps_month/month_in_year)

        new_sensor = new_sensor.fillna(-1) # specifically for weather_precip_path10min which has NaN
        # ic(new_sensor)
        df_filtered = df_filtered.append(new_sensor)

    return df_filtered

train_dataset = process_data(source, sensors_train)
test_dataset = process_data(source, sensors_test)

train_dataset.to_csv(r'Data/train_dataset.csv', index=False)
test_dataset.to_csv(r'Data/test_dataset.csv', index=False)
