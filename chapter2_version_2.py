from read_clean_functions import (read_clean, read_time_series, data_departure)
import glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.utils import resample
import matplotlib.pyplot as plt

vehicle_names = ["P_1092"]


for vehicle_name in vehicle_names:

    all_files = glob.glob(f'D:\\LSTM\\Data\\{vehicle_name}\\*.csv')

    df_timeseries1, df_timeseries2 = read_time_series(all_files)
    df_timeseries_clean, df_full_trips = read_clean(df_timeseries2, vehicle_name)
    departure_input = data_departure(df_timeseries_clean)



