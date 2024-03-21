import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from parking import (v2g_normal, box_plot_with_stats_for_three,
                     charging_c_k_tou, storage_cap_tou, total_storage_tou, v2g_participate, parking_sessions, charging_selection, charging_speed, extra_extra_kwh_parking, range_indicator, v2g_draw, v2g_tou_parking_function,
                     conditionally_update_start_time, conditionally_update_end_time, conditionally_update_start_time_charging, conditionally_update_end_time_charging)
import warnings
warnings.filterwarnings("ignore")
# %%
##################################################################################################################
##################################################################################################################
# Section 1
# Reading the cleaned CSV file which was saved earlier
# final_dataframes = pd.read_csv("data.csv")
# V2G_cap_charging_rate = pd.read_csv("V2G_cap_charging_rate.csv")
# v2g_tou = pd.read_csv("v2g_tou.csv")
# final_dataframes_charging = charging_dataframe(final_dataframes, 0)
# %%
##################################################################################################################
##################################################################################################################
parking_dataframe_extra = parking_sessions(final_dataframes)
# Function to conditionally update the start_time to match the day column
parking_dataframe_extra = parking_dataframe_extra.apply(conditionally_update_start_time, axis=1)
parking_dataframe_extra = parking_dataframe_extra.apply(conditionally_update_end_time, axis=1)
parking_dataframe_extra = parking_dataframe_extra.apply(conditionally_update_start_time_charging, axis=1)
parking_dataframe_extra = parking_dataframe_extra.apply(conditionally_update_end_time_charging, axis=1)

parking_dataframe_extra = charging_selection(parking_dataframe_extra)
# determine teh charging speed based on the parking time, charging time and SOC before and after charging
parking_dataframe_extra = charging_speed(parking_dataframe_extra)
# range indicator is indicating if the trip will fail or not
parking_dataframe_extra = range_indicator(parking_dataframe_extra)
parking_dataframe_extra = v2g_draw(parking_dataframe_extra)
parking_dataframe_extra = parking_dataframe_extra.loc[parking_dataframe_extra["V2G_time_min"] >= 0]


(V2G_cap_charging_rate_p, V2G_hourly_12_p, V2G_hourly_6_p, V2G_hourly_19_p, V2G_hourly_12_sum_p, V2G_hourly_6_sum_p, V2G_hourly_19_sum_p, V2G_hourly_12_sum_reset_p, V2G_hourly_6_sum_reset_p, V2G_hourly_19_sum_reset_p) = v2g_normal(parking_dataframe_extra)

v2g_tou_p = v2g_tou_parking_function(V2G_cap_charging_rate_p)

v2g_tou_normal = v2g_tou_nc.copy()
v2g_tou_normal.discharge_start = v2g_tou_normal.discharge_start.dt.tz_localize(None)
v2g_tou_normal.discharge_end = v2g_tou_normal.discharge_end.dt.tz_localize(None)

v2g_tou_parking = pd.concat([v2g_tou_normal, v2g_tou_p], axis=0).reset_index(drop=True)

v2g_tou_parking = v2g_tou_parking[v2g_tou_parking["vehicle_name"] != "P_1294"]
v2g_tou_parking = v2g_tou_parking.apply(conditionally_adjust_charging_end, axis=1)
v2g_tou_parking.loc[(v2g_tou_parking["energy[charge_type][type]"] != "Parking") & ((v2g_tou_parking["discharge_end"] - v2g_tou_parking["discharge_start"]).dt.seconds == 0), "V2G_cap_6k_tou"] = 0
v2g_tou_parking.loc[(v2g_tou_parking["energy[charge_type][type]"] != "Parking") & ((v2g_tou_parking["discharge_end"] - v2g_tou_parking["discharge_start"]).dt.seconds == 0), "V2G_cap_12k_tou"] = 0
v2g_tou_parking.loc[(v2g_tou_parking["energy[charge_type][type]"] != "Parking") & ((v2g_tou_parking["discharge_end"] - v2g_tou_parking["discharge_start"]).dt.seconds == 0), "V2G_cap_19k_tou"] = 0
test = v2g_tou_parking[["vehicle_name", "start_time_local", "end_time_local", "next_departure_time", "start_time_charging", "end_time_charging", "energy[charge_type][type]", "year", "month", "day", "discharge_start", "discharge_end", "indicator_column",
                        "V2G_cap_6k_tou", "V2G_cap_12k_tou", "V2G_cap_19k_tou"]].sort_values(by=["vehicle_name", "year", "month", "day"])

V2G_hourly_12_tou_p, V2G_hourly_6_tou_p, V2G_hourly_19_tou_p, V2G_hourly_12_tou_sum_p, V2G_hourly_6_tou_sum_p, V2G_hourly_19_tou_sum_p = storage_cap_tou(v2g_tou_parking)

v2g_tou_p = v2g_participate(v2g_tou_p)

v2g_participate_dataframe_p = v2g_tou_p[v2g_tou_p["V2G_participate"] == True]

# df_summary_tou_parking_nt = total_storage_tou(V2G_hourly_6_tou_sum_p, V2G_hourly_12_tou_sum_p, V2G_hourly_19_tou_sum_p)
df_summary_tou_parking_nc = total_storage_tou(V2G_hourly_6_tou_sum_p, V2G_hourly_12_tou_sum_p, V2G_hourly_19_tou_sum_p)

df_box1_tou = [V2G_hourly_12_tou_sum_p, V2G_hourly_6_tou_sum_p, V2G_hourly_19_tou_sum_p]
labels1_tou = ["12 kW", "6.6 kW", "19 kW"]

box_plot_with_stats_for_three(df_box1_tou, labels1_tou, 0, 2000)

test0 = charging_c_k_tou(v2g_tou_parking).reset_index(drop=False)
test0 = pd.merge(test0, travel_distance_mean, how="left", on="vehicle_name")
test0.iloc[:, 2:5] = test0.iloc[:, 2:5].sub(test0['charging_cap'], axis=0)
test0.iloc[:, 1] = test0.iloc[:, 1].div(test0['Driving days'], axis=0)
test0.iloc[:, 2:5] = test0.iloc[:, 2:5].div(test0['observation_day'], axis=0)
test0.iloc[:, 1:5] = test0.iloc[:, 1:5].div(test0['bat_cap']*0.01, axis=0)
test0.rename(columns={"distance": "Average Daily VMT"}, inplace=True)
test0.rename(columns={"charging_cap": "Driving"}, inplace=True)
test0.rename(columns={"charging_v2g_energy_6k": "V2G - 6.6kW"}, inplace=True)
test0.rename(columns={"charging_v2g_energy_12k": "V2G 12 - kW"}, inplace=True)
test0.rename(columns={"charging_v2g_energy_19k": "V2G 19 - kW"}, inplace=True)
extra_extra_kwh_parking(test0)
############################################################################################################
# %%
v2g_test = pd.DataFrame({"year": v2g_tou_parking["year"]})
v2g_test_12 = pd.concat([v2g_test, V2G_hourly_12_tou_p], axis=1)
v2g_test_6 = pd.concat([v2g_test, V2G_hourly_6_tou_p], axis=1)
v2g_test_19 = pd.concat([v2g_test, V2G_hourly_19_tou_p], axis=1)
# v2g_test = v2g_test.drop(columns=[col for col in ["year", "month", "day"] if col in v2g_test.columns], axis=1)

v2g_test_12_s = V2G_hourly_12_tou_sta.drop(columns="vehicle_name")
v2g_test_6_s = V2G_hourly_6_tou_sta.drop(columns="vehicle_name")
v2g_test_19_s = V2G_hourly_19_tou_sta.drop(columns="vehicle_name")

v2g_test_6 = v2g_test_6.groupby(["year", "month", "day"]).sum() / 64
v2g_test_12 = v2g_test_12.groupby(["year", "month", "day"]).sum() / 64
v2g_test_19 = v2g_test_19.groupby(["year", "month", "day"]).sum() / 64

v2g_test_6_s = v2g_test_6_s.groupby(["day"]).sum() / 64
v2g_test_12_s = v2g_test_12_s.groupby(["day"]).sum() / 64
v2g_test_19_s = v2g_test_19_s.groupby(["day"]).sum() / 64

# Assuming 'daily_kwh' contains the sum of daily kWh available for each vehicle
daily_kwh_6 = v2g_test_6.sum(axis=1)
daily_kwh_12 = v2g_test_12.sum(axis=1)
daily_kwh_19 = v2g_test_19.sum(axis=1)
# Step 1: Sort the daily kWh values in ascending order
sorted_kwh_6 = np.sort(daily_kwh_6)
sorted_kwh_12 = np.sort(daily_kwh_12)
sorted_kwh_19 = np.sort(daily_kwh_19)
# Step 2: Calculate the CDF


daily_kwh_6_s = v2g_test_6_s.sum(axis=1)
daily_kwh_12_s = v2g_test_12_s.sum(axis=1)
daily_kwh_19_s = v2g_test_19_s.sum(axis=1)
# Step 1: Sort the daily kWh values in ascending order
sorted_kwh_6_s = np.sort(daily_kwh_6_s)
sorted_kwh_12_s = np.sort(daily_kwh_12_s)
sorted_kwh_19_s = np.sort(daily_kwh_19_s)



# Calculate the cumulative distribution
cdf6 = (np.arange(1, len(sorted_kwh_6)+1) / len(sorted_kwh_6))
cdf12 = (np.arange(1, len(sorted_kwh_12)+1) / len(sorted_kwh_12))
cdf19 = (np.arange(1, len(sorted_kwh_19)+1) / len(sorted_kwh_19))

cdf6_s = (np.arange(1, len(sorted_kwh_6_s)+1) / len(sorted_kwh_6_s))
cdf12_s = (np.arange(1, len(sorted_kwh_12_s)+1) / len(sorted_kwh_12_s))
cdf19_s = (np.arange(1, len(sorted_kwh_19_s)+1) / len(sorted_kwh_19_s))


# Step 3: Plot the CDF
plt.figure(figsize=(10, 6))
plt.plot(cdf6, sorted_kwh_6, marker='.', linestyle='none', label="6.6 kW")
plt.plot(cdf12, sorted_kwh_12, marker='.', linestyle='none', label="12 kW")
plt.plot(cdf19, sorted_kwh_19, marker='.', linestyle='none', label="19 kW")

plt.xlabel('% Share of the EV in V2G program', fontsize=14)  # Increase fontsize for x-axis title
plt.ylabel('Average kWh Available per Day per Vehicle', fontsize=14)  # Increase fontsize for y-axis title
plt.title('CDF of kWh Available per Day per Vehicle')
plt.grid(True)
plt.legend(fontsize=12)

# Set x-axis ticks from 0 to 100 with a step of 10
plt.xticks(ticks=[i/100 for i in range(0, 101, 10)], labels=[str(i) for i in range(0, 101, 10)], fontsize=12, rotation=0)

# Increase fontsize for y-axis ticks
plt.yticks(fontsize=12)

plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()

# Step 3: Plot the CDF
plt.figure(figsize=(10, 6))
plt.plot((1 - cdf6), sorted_kwh_6, marker='.', linestyle='none', label="6.6 kW")
plt.plot((1 - cdf12), sorted_kwh_12, marker='.', linestyle='none', label="12 kW")
plt.plot((1 - cdf19), sorted_kwh_19, marker='.', linestyle='none', label="19 kW")

plt.xlabel('% Share of the EV in V2G program', fontsize=14)
plt.ylabel('Average kWh Available per Day per Vehicle', fontsize=14)
plt.title('Survival plot of kWh Available per Day per Vehicle')
plt.grid(True)
plt.legend(fontsize=12)

# Set x-axis ticks from 0 to 100 with a step of 10
plt.xticks(ticks=[i/100 for i in range(0, 101, 10)], labels=[str(i) for i in range(0, 101, 10)], fontsize=12, rotation=0)

# Increase fontsize for y-axis ticks
plt.yticks(fontsize=12)

plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()

################################################################
# %%
average_energy = v2g_tou_parking[["vehicle_name", "discharge_start", "discharge_end", "year", "month", "day", "V2G_cap_6k_tou", "V2G_cap_12k_tou", "V2G_cap_19k_tou"]]

average_energy_final1 = average_energy.groupby(["vehicle_name", "year", "month", "day"])[["V2G_cap_6k_tou", "V2G_cap_12k_tou", "V2G_cap_19k_tou"]].sum().reset_index().sort_values(by=["vehicle_name", "year", "month", "day"])
average_energy_final2 = average_energy_final1.groupby(["vehicle_name", "year", "month", "day"])[["V2G_cap_6k_tou", "V2G_cap_12k_tou", "V2G_cap_19k_tou"]].mean().reset_index()
# Calculate the 10th percentile for each column
percentiles_10th = average_energy_final2.groupby(["vehicle_name"])[["V2G_cap_6k_tou", "V2G_cap_12k_tou", "V2G_cap_19k_tou"]].quantile(0.1)

# Show the 10th percentile values for each column
print(percentiles_10th)


# Create a figure and three subplots
fig, axes = plt.subplots(3, 1, figsize=(8, 12))

# Plotting histograms for observation_day, Driving days, and Charging days
for i, column in enumerate(['observation_day', 'Driving days', 'Charging days']):
    axes[i].hist(test0[column], bins=20, range=(0, 500), alpha=0.5, color=['blue', 'green', 'red'][i])
    if i == 0:
        axes[i].set_title('Observation Days', fontsize=16)  # Set title for the first subplot
    else:
        axes[i].set_title(column, fontsize=16)
    axes[i].set_xlabel('Days', fontsize=14)
    axes[i].set_ylabel('Frequency', fontsize=14)
    axes[i].tick_params(axis='both', which='major', labelsize=12)  # Increase size of ticks
    axes[i].grid(axis='y', linewidth=0.5)  # Add vertical grid lines and increase grid line width

# Adjust layout to prevent overlap
plt.tight_layout()

# Display the plot
plt.show()

# Create a figure and three subplots
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Plotting histograms for Driving and Average Daily VMT
for i, column in enumerate(['Driving', 'Average Daily VMT']):
    axes[i].hist(test0[column], bins=20, range=(0, 150), alpha=0.5, color=['blue', 'green'][i])

    axes[i].set_title(column, fontsize=22)
    axes[i].set_xlabel('Values', fontsize=20)
    axes[i].set_ylabel('Frequency', fontsize=20)
    axes[i].tick_params(axis='both', which='major', labelsize=18)  # Increase size of ticks
    axes[i].grid(axis='y', linewidth=0.5)  # Add vertical grid lines and increase grid line width

# Set X-axis titles
axes[0].set_xlabel('Average energy consumption during driving per vehicle (kWh)', fontsize=20)
axes[1].set_xlabel('Average daily VMT per vehicle (mile)', fontsize=20)

# Adjust layout to prevent overlap
plt.tight_layout()

# Display the plot
plt.show()