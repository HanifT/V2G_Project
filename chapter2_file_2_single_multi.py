import pandas as pd
from parking import (charging_dataframe, total_storage, v2g_normal, v2g_r5, v2g_r10, v2g_normal_mc, v2g_r5_mc, v2g_r10_mc, failure_estimation, total_capacity, box_plot_with_stats_for_three)
import warnings
warnings.filterwarnings("ignore")
##################################################################################################################
##################################################################################################################
# Section 1
# # Reading the raw data and clean it
# final_dataframes = clean_data

# # Saving cleaned data as a csv file
# final_dataframes.to_csv("data.csv")
##################################################################################################################
##################################################################################################################
# Section 2
# Reading the cleaned CSV file which was saved earlier
final_dataframes = pd.read_csv("data.csv")
final_dataframes_charging = charging_dataframe(final_dataframes, 0)
##################################################################################################################
##################################################################################################################
# Section 3
# Single Cycle V2G
##################################################################################################################
##################################################################################################################
# calculating the storage capacity based on the different charging discharging speed
(V2G_cap_charging_rate, V2G_hourly_12, V2G_hourly_6, V2G_hourly_19, V2G_hourly_12_sum, V2G_hourly_6_sum, V2G_hourly_19_sum, V2G_hourly_12_sum_reset, V2G_hourly_6_sum_reset, V2G_hourly_19_sum_reset) = v2g_normal(final_dataframes_charging)

(V2G_cap_soc_rate5, V2G_hourly_12_s5, V2G_hourly_6_s5, V2G_hourly_19_s5, V2G_hourly_12_sum_s5, V2G_hourly_6_sum_s5, V2G_hourly_19_sum_s5, V2G_hourly_12_sum_reset_s5, V2G_hourly_6_sum_reset_s5, V2G_hourly_19_sum_reset_s5) = v2g_r5(final_dataframes_charging)

(V2G_cap_soc_rate10, V2G_hourly_12_s10, V2G_hourly_6_s10, V2G_hourly_19_s10, V2G_hourly_12_sum_s10, V2G_hourly_6_sum_s10, V2G_hourly_19_sum_s10,
 V2G_hourly_12_sum_reset_s10, V2G_hourly_6_sum_reset_s10, V2G_hourly_19_sum_reset_s10) = v2g_r10(final_dataframes_charging)

df_summary = total_storage(V2G_hourly_12_sum, V2G_hourly_12_sum_s5, V2G_hourly_12_sum_s10, V2G_hourly_6_sum, V2G_hourly_6_sum_s5, V2G_hourly_6_sum_s10, V2G_hourly_19_sum, V2G_hourly_19_sum_s5, V2G_hourly_19_sum_s10)

df_summary_failure = failure_estimation(V2G_cap_soc_rate5, V2G_cap_soc_rate10)

total_cap = total_capacity(final_dataframes)
V2G_cap_charging_rate.to_csv("V2G_cap_charging_rate.csv")

# Draw final Graph

df_box1 = [V2G_hourly_12_sum, V2G_hourly_6_sum, V2G_hourly_19_sum]
labels1 = ["12 kW", "6.6 kW", "19 kW"]
df_box2 = [V2G_hourly_12_sum_s5, V2G_hourly_6_sum_s5, V2G_hourly_19_sum_s5]
labels2 = ["12 kW", "6.6 kW", "19 kW"]
df_box3 = [V2G_hourly_12_sum_s10, V2G_hourly_6_sum_s10, V2G_hourly_19_sum_s10]
labels3 = ["12 kW", "6.6 kW", "19 kW"]

box_plot_with_stats_for_three(df_box1, labels1)
box_plot_with_stats_for_three(df_box2, labels2)
box_plot_with_stats_for_three(df_box3, labels3)

##################################################################################################################
##################################################################################################################
# Multi Cycle V2G
##################################################################################################################
##################################################################################################################
V2G_cap_charging_rate_mc, V2G_hourly_12_mc, V2G_hourly_6_mc, V2G_hourly_19_mc, V2G_hourly_12_sum_mc, V2G_hourly_6_sum_mc, V2G_hourly_19_sum_mc, V2G_hourly_12_sum_reset_mc, V2G_hourly_6_sum_reset_mc, V2G_hourly_19_sum_reset_mc = \
    v2g_normal_mc(final_dataframes_charging)
V2G_cap_soc_rate5_mc, V2G_hourly_12_s5_mc, V2G_hourly_6_s5_mc, V2G_hourly_19_s5_mc, V2G_hourly_12_sum_s5_mc, V2G_hourly_6_sum_s5_mc, V2G_hourly_19_sum_s5_mc, V2G_hourly_12_sum_reset_s5_mc, V2G_hourly_6_sum_reset_s5_mc, V2G_hourly_19_sum_reset_s5_mc =\
    v2g_r5_mc(final_dataframes_charging)
V2G_cap_soc_rate10_mc, V2G_hourly_12_s10_mc, V2G_hourly_6_s10_mc, V2G_hourly_19_s10_mc, V2G_hourly_12_sum_s10_mc, V2G_hourly_6_sum_s10_mc, V2G_hourly_19_sum_s10_mc, V2G_hourly_12_sum_reset_s10_mc, V2G_hourly_6_sum_reset_s10_mc, V2G_hourly_19_sum_reset_s10_mc =\
    v2g_r10_mc(final_dataframes_charging)

df_summary_mc = total_storage(V2G_hourly_12_sum_mc, V2G_hourly_12_sum_s5_mc, V2G_hourly_12_sum_s10_mc,
                              V2G_hourly_6_sum_mc, V2G_hourly_6_sum_s5_mc, V2G_hourly_6_sum_s10_mc,
                              V2G_hourly_19_sum_mc, V2G_hourly_19_sum_s5_mc, V2G_hourly_19_sum_s10_mc)

df_summary_failure_mc = failure_estimation(V2G_cap_soc_rate5_mc, V2G_cap_soc_rate10_mc)


df_box1_mc = [V2G_hourly_12_sum_mc, V2G_hourly_6_sum_mc, V2G_hourly_19_sum_mc]
labels1_mc = ["12 kW", "6.6 kW", "19 kW"]
df_box2_mc = [V2G_hourly_12_sum_s5_mc, V2G_hourly_6_sum_s5_mc, V2G_hourly_19_sum_s5_mc]
labels2_mc = ["12 kW", "6.6 kW", "19 kW"]
df_box3_mc = [V2G_hourly_12_sum_s10_mc, V2G_hourly_6_sum_s10_mc, V2G_hourly_19_sum_s10_mc]
labels3_mc = ["12 kW", "6.6 kW", "19 kW"]


box_plot_with_stats_for_three(df_box1_mc, labels1_mc)
box_plot_with_stats_for_three(df_box2_mc, labels2_mc)
box_plot_with_stats_for_three(df_box3_mc, labels3_mc)


travel_distance = final_dataframes.groupby(["vehicle_name", 'year', 'month', 'day'])["distance"].sum().reset_index()

# Count the number of rows for each group
observation_day = travel_distance.groupby('vehicle_name').size().reset_index(name='count')
charging_day = final_dataframes_charging.groupby(["vehicle_name", 'year', 'month', 'day'])["distance"].sum().reset_index()
charging_day = charging_day.groupby('vehicle_name').size().reset_index(name='count')
# Rename the column to 'observation_day'
observation_day.rename(columns={'count': 'Driving days'}, inplace=True)
charging_day.rename(columns={'count': 'Charging days'}, inplace=True)
# Concatenate observation_day with travel_distance

travel_distance_mean = travel_distance.groupby(["vehicle_name"]).agg({'distance': 'mean'}).reset_index()
travel_distance_mean = pd.merge(travel_distance_mean, observation_day, how="left", on="vehicle_name")
travel_distance_mean = pd.merge(travel_distance_mean, charging_day, how="left", on="vehicle_name")