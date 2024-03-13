import pandas as pd
from parking import (box_plot_with_stats_for_three, storage_cap_tou, total_storage_tou, v2g_participate, v2g_fail, v2g_tou_trip_buffer, v2g_tou_charging_buffer, extra_extra_kwh_parking)
import warnings
warnings.filterwarnings("ignore")
##################################################################################################################
##################################################################################################################
# Section 1
# # Reading the cleaned CSV file which was saved earlier
# final_dataframes = pd.read_csv("data.csv")
# V2G_cap_charging_rate = pd.read_csv("V2G_cap_charging_rate.csv")
# final_dataframes_charging = charging_dataframe(final_dataframes, 0)
##################################################################################################################
##################################################################################################################
# TOU V2G G next Trip
##################################################################################################################
##################################################################################################################
v2g_tou_nt = V2G_cap_charging_rate.copy()
v2g_tou_nt = v2g_tou_trip_buffer(v2g_tou_nt)

V2G_hourly_12_tou_nt, V2G_hourly_6_tou_nt, V2G_hourly_19_tou_nt, V2G_hourly_12_tou_nt_sum, V2G_hourly_6_tou_nt_sum, V2G_hourly_19_tou_nt_sum = storage_cap_tou(v2g_tou_nt)

v2g_tou_nt = v2g_participate(v2g_tou_nt)


v2g_participate_dataframe_nt = v2g_tou_nt[v2g_tou_nt["V2G_participate"] == True]

df_summary_tou_nt = total_storage_tou(V2G_hourly_6_tou_nt_sum, V2G_hourly_12_tou_nt_sum, V2G_hourly_19_tou_nt_sum)

df_box1_tou = [V2G_hourly_6_tou_nt_sum, V2G_hourly_12_tou_nt_sum, V2G_hourly_19_tou_nt_sum]
labels1_tou = ["6.6 kW", "12 kW", "19 kW"]

box_plot_with_stats_for_three(df_box1_tou, labels1_tou)
data_failure_nt = v2g_fail(v2g_participate_dataframe_nt)

##################################################################################################################
##################################################################################################################
# TOU V2G Next Charging
##################################################################################################################
##################################################################################################################
v2g_tou_nc = V2G_cap_charging_rate.copy()
v2g_tou_nc = v2g_tou_charging_buffer(v2g_tou_nc)
v2g_tou_nc = v2g_tou_nc[v2g_tou_nc["vehicle_name"] != "P_1294"]
V2G_hourly_12_tou_nc, V2G_hourly_6_tou_nc, V2G_hourly_19_tou_nc, V2G_hourly_12_tou_nc_sum, V2G_hourly_6_tou_nc_sum, V2G_hourly_19_tou_nc_sum = storage_cap_tou(v2g_tou_nc)

v2g_tou_nc = v2g_participate(v2g_tou_nc)


v2g_participate_dataframe_nc = v2g_tou_nc[v2g_tou_nc["V2G_participate"] == True]

df_summary_tou_nc = total_storage_tou(V2G_hourly_6_tou_nc_sum, V2G_hourly_12_tou_nc_sum, V2G_hourly_19_tou_nc_sum)

df_box1_tou = [V2G_hourly_6_tou_nc_sum, V2G_hourly_12_tou_nc_sum, V2G_hourly_19_tou_nc_sum]
labels1_tou = ["6.6 kW", "12 kW", "19 kW"]


box_plot_with_stats_for_three(df_box1_tou, labels1_tou, 0, 2000)

data_failure_nc = v2g_fail(v2g_participate_dataframe_nc)

test0_tou3 = charging_c_k_tou_real(v2g_tou_nc).reset_index(drop=False)

test0_tou3 = pd.merge(test0_tou3, travel_distance_mean, how="left", on="vehicle_name")
test0_tou3.iloc[:, 2:5] = test0_tou3.iloc[:, 2:5].sub(test0_tou3['charging_cap'], axis=0)
test0_tou3.iloc[:, 1] = test0_tou3.iloc[:, 1].div(test0_tou3['Driving days'], axis=0)
test0_tou3.iloc[:, 2:5] = test0_tou3.iloc[:, 2:5].div(test0_tou3['Charging days'], axis=0)
test0_tou3.iloc[:, 1:5] = test0_tou3.iloc[:, 1:5].div(test0_tou3['bat_cap']*0.01, axis=0)
test0_tou3.rename(columns={"distance": "Average Daily VMT"}, inplace=True)
test0_tou3.rename(columns={"charging_cap": "Driving"}, inplace=True)
test0_tou3.rename(columns={"charging_v2g_energy_6k": "V2G - 6.6 kW"}, inplace=True)
test0_tou3.rename(columns={"charging_v2g_energy_12k": "V2G - 12 kW"}, inplace=True)
test0_tou3.rename(columns={"charging_v2g_energy_19k": "V2G - 19 kW"}, inplace=True)

extra_extra_kwh_parking(test0_tou3)
