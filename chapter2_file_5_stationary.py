import pandas as pd
import numpy as np
from datetime import timedelta
from parking import (charging_dataframe, box_plot_with_stats_for_three, v2g_tou_cap, total_storage_tou, v2g_participate, storage_cap_tou_sta, extra_extra_kwh_sta)
import warnings
warnings.filterwarnings("ignore")
##################################################################################################################
##################################################################################################################
# Section 1
# Reading the cleaned CSV file which was saved earlier
# final_dataframes = pd.read_csv("data.csv")
# V2G_cap_charging_rate = pd.read_csv("V2G_cap_charging_rate.csv")
# final_dataframes_charging = charging_dataframe(final_dataframes, 0)
##################################################################################################################
##################################################################################################################
# max V2G stationary
##################################################################################################################
##################################################################################################################
v2g_tou_stationary = V2G_cap_charging_rate.copy()
v2g_tou_stationary = v2g_tou_cap(v2g_tou_stationary)

v2g_tou_stationary["discharge_start"] = v2g_tou_stationary['discharge_start'].dt.tz_convert(None).dt.strftime('%Y-%m-%d 16:00:00')
v2g_tou_stationary["discharge_end"] = v2g_tou_stationary['discharge_end'].dt.tz_convert(None).dt.strftime('%Y-%m-%d 20:59:00')
v2g_tou_stationary["charge_start"] = (v2g_tou_stationary['charge_start'].dt.tz_convert(None).dt.strftime('%Y-%m-%d 21:00:00'))
v2g_tou_stationary["charge_end"] = (v2g_tou_stationary['charge_end'].dt.tz_convert(None).dt.strftime('%Y-%m-%d 15:59:00'))
v2g_tou_stationary["charge_start"] = pd.to_datetime(v2g_tou_stationary["charge_start"]) + timedelta(days=1)
v2g_tou_stationary["charge_end"] = pd.to_datetime(v2g_tou_stationary["charge_end"]) + timedelta(days=1)
v2g_tou_stationary['discharge_start'] = pd.to_datetime(v2g_tou_stationary['discharge_start'])
v2g_tou_stationary['discharge_end'] = pd.to_datetime(v2g_tou_stationary['discharge_end'])
v2g_tou_stationary['charge_start'] = pd.to_datetime(v2g_tou_stationary['charge_start'])
v2g_tou_stationary['charge_end'] = pd.to_datetime(v2g_tou_stationary['charge_end'])

v2g_tou_stationary['V2G_cap_6k'] = v2g_tou_stationary["bat_cap"]
v2g_tou_stationary['V2G_cap_12k'] = v2g_tou_stationary["bat_cap"]
v2g_tou_stationary['V2G_cap_19k'] = v2g_tou_stationary["bat_cap"]

v2g_tou_stationary['V2G_SOC_half_6k'] = 100
v2g_tou_stationary['V2G_SOC_half_12k'] = 100
v2g_tou_stationary['V2G_SOC_half_19k'] = 100

v2g_tou_stationary = v2g_tou_stationary.groupby("vehicle_name").tail(n=1)
v2g_tou_stationary["day"] = 1

v2g_tou_stationary = pd.concat([v2g_tou_stationary] * 365, ignore_index=True)
v2g_tou_stationary['day'] = np.repeat(np.arange(1, 366), len(v2g_tou_stationary) // 365)

V2G_hourly_12_tou_sta, V2G_hourly_6_tou_sta, V2G_hourly_19_tou_sta, V2G_hourly_12_tou_sum_sta, V2G_hourly_6_tou_sum_sta, V2G_hourly_19_tou_sum_sta = storage_cap_tou_sta(v2g_tou_stationary)

v2g_tou_stationary = v2g_participate(v2g_tou_stationary)


v2g_participate_dataframe_sta = v2g_tou_stationary

df_summary_tou_sta = total_storage_tou(V2G_hourly_6_tou_sum_sta, V2G_hourly_12_tou_sum_sta, V2G_hourly_19_tou_sum_sta)

df_box1_tou_sta = [V2G_hourly_12_tou_sum_sta, V2G_hourly_6_tou_sum_sta, V2G_hourly_19_tou_sum_sta]
labels1_tou_sta = ["12 kW", "6.6 kW", "19 kW"]

V2G_hourly_6_tou_sta["vehicle_name"] = v2g_tou_stationary["vehicle_name"]
V2G_hourly_12_tou_sta["vehicle_name"] = v2g_tou_stationary["vehicle_name"]
V2G_hourly_19_tou_sta["vehicle_name"] = v2g_tou_stationary["vehicle_name"]

V2G_hourly_6_tou_sta_dropped = V2G_hourly_6_tou_sta.drop(columns=['day']).groupby('vehicle_name').sum().sum(axis=1)
V2G_hourly_12_tou_sta_dropped = V2G_hourly_12_tou_sta.drop(columns=['day']).groupby('vehicle_name').sum().sum(axis=1)
V2G_hourly_19_tou_sta_dropped = V2G_hourly_19_tou_sta.drop(columns=['day']).groupby('vehicle_name').sum().sum(axis=1)

box_plot_with_stats_for_three(df_box1_tou_sta, labels1_tou_sta, 0, 2000)


v2g_tou_stationary_max = pd.DataFrame({"6.6 kW": V2G_hourly_6_tou_sta_dropped, "12 kW": V2G_hourly_12_tou_sta_dropped, "19 kW": V2G_hourly_19_tou_sta_dropped})

extra_extra_kwh_sta(v2g_tou_stationary_max)
##################################################################################################################
##################################################################################################################
