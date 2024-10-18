# %%
import os
import json
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from parking import xlsx_read, plotting, draw_RT, draw_util, draw_util_rt, draw_profile, demand_response, draw_parking, draw_charging, draw_combined, draw_multilevel_pie, draw_rose_chart_parking, draw_rose_chart_charging
# %% Reading Files
GHG_data = pd.read_csv("CISO.csv")
GHG_dict = dict(enumerate(GHG_data.iloc[:, 0]))

with open("combined_price_PGE_average.json", "r") as json_file:
    combined_price_PGE_average = json.load(json_file)

with open("tou_prices.json", "r") as json_file:
    tou_prices = json.load(json_file)

with open("merged_dict.json", "r") as json_file:
    merged_dict = json.load(json_file)

# Flatten the dictionary
flattened_data = []
for vehicle, hours in merged_dict.items():
    for hour, values in hours.items():
        entry = {'Vehicle': vehicle, 'Hour': int(hour)}
        entry.update(values)
        flattened_data.append(entry)

# Create DataFrame
flatten_veh_data = pd.DataFrame(flattened_data)
final_dataframes = pd.read_csv("data.csv")

# %%
charging_cost_NR = tou_prices
directory_norm = "/Users/haniftayarani/V2G_Project/Result_normal/V2G_locations/Home_Work"

total_costs_df_norm, costs_df_norm = xlsx_read(directory_norm)
total_costs_df_norm["V2G Location"] = " "
total_costs_df_norm["Charging Type"] = "Normal"
plotting(total_costs_df_norm, 50)
draw_RT(combined_price_PGE_average)
# draw_RT(combined_price_PGE_gen)
result_NR, hourly_data_NR, hourly_data_charging_NR, hourly_data_discharging_NR = json_file(directory_norm, flatten_veh_data)
draw_util_rt(result_NR)
result_RTH_DR1, result_RTH_DR10, result_RTH_DR20 = demand_response(result_NR, ["Home"])
draw_profile(charging_cost_NR, hourly_data_NR)

# %%
charging_cost_RT = combined_price_PGE_average
directory_RTH = "/Users/haniftayarani/V2G_Project/Result_RT/V2G_locations/Home"

total_costs_df_RTH, costs_df_RTH = xlsx_read(directory_RTH)
total_costs_df_RTH["V2G Location"] = "Home"
plotting(total_costs_df_RTH, 50)
draw_RT(combined_price_PGE_average)
# draw_RT(combined_price_PGE_gen)
result_RTH, hourly_data_RTH, hourly_data_charging_RTH, hourly_data_discharging_RTH = json_file(directory_RTH, flatten_veh_data)
draw_util_rt(result_RTH)
result_RTH_DR1, result_RTH_DR10, result_RTH_DR20 = demand_response(result_RTH, ["Home"])
draw_profile(charging_cost_RT, hourly_data_RTH)

directory_RTHW = "/Users/haniftayarani/V2G_Project/Result_RT/V2G_locations/Home_Work"

total_costs_df_RTHW, costs_df_RTHW = xlsx_read(directory_RTHW)
total_costs_df_RTHW["V2G Location"] = "Home & Work"
plotting(total_costs_df_RTHW, 50)
draw_RT(combined_price_PGE_average)
result_RTHW, hourly_data_RTHW, hourly_data_charging_RTHW, hourly_data_discharging_RTHW = json_file(directory_RTHW, flatten_veh_data)
draw_util_rt(result_RTHW)
result_RTHW_DR1, result_RTHW_DR10, result_RTHW_DR20 = demand_response(result_RTHW, ["Home", "Work"])
draw_profile(charging_cost_RT, hourly_data_RTHW)

total_costs_df_RTH_HW = pd.concat([total_costs_df_RTH, total_costs_df_RTHW[total_costs_df_RTHW["Charging Type"] != "smart"]], axis=0).reset_index(drop=True)
total_costs_df_RTH_HW.loc[total_costs_df_RTH_HW["V2G Location"] == "Home", "group"] = 1
total_costs_df_RTH_HW.loc[total_costs_df_RTH_HW["V2G Location"] != "Home", "group"] = 2
plotting(pd.concat([total_costs_df_RTH_HW, total_costs_df_norm]), 50)

# %%
charging_cost_TOU = tou_prices
directory_TOUH = "/Users/haniftayarani/V2G_Project/Result_TOU/V2G_locations/Home1"

total_costs_df_TOUH, costs_df_TOUH = xlsx_read(directory_TOUH)
total_costs_df_TOUH["V2G Location"] = "Home"
plotting(total_costs_df_TOUH, 50)
draw_RT(combined_price_PGE_average)
result_TOUH, hourly_data_TOUH, hourly_data_charging_TOUH, hourly_data_discharging_TOUH = json_file(directory_TOUH, flatten_veh_data)
draw_util(result_TOUH)
result_TOUH_DR1, result_TOUH_DR10, result_TOUH_DR20 = demand_response(result_TOUH, ["Home"])
draw_profile(charging_cost_TOU, hourly_data_TOUH)

directory_TOUHW = "/Users/haniftayarani/V2G_Project/Result_TOU/V2G_locations/Home_Work"

total_costs_df_TOUHW, costs_df_TOUHW = xlsx_read(directory_TOUHW)
total_costs_df_TOUHW["V2G Location"] = "Home & Work"
plotting(total_costs_df_TOUHW, 50)
draw_RT(combined_price_PGE_average)
result_TOUHW, hourly_data_TOUHW, hourly_data_charging_TOUHW, hourly_data_discharging_TOUHW = json_file(directory_TOUHW, flatten_veh_data)
draw_util(result_TOUHW)
result_TOUHW_DR1, result_TOUHW_DR10, result_TOUHW_DR20 = demand_response(result_TOUHW, ["Home", "Work"])
draw_profile(charging_cost_TOU, hourly_data_TOUHW)


total_costs_df_TOUH_HW = pd.concat([total_costs_df_TOUH, total_costs_df_TOUHW[total_costs_df_TOUHW["Charging Type"] != "smart"]], axis=0).reset_index(drop=True)
total_costs_df_TOUH_HW.loc[total_costs_df_TOUH_HW["V2G Location"] == "Home", "group"] = 1
total_costs_df_TOUH_HW.loc[total_costs_df_TOUH_HW["V2G Location"] != "Home", "group"] = 2
total_costs_df_TOUH_HW = total_costs_df_TOUH_HW.sort_values(by=['V2G Location', 'Charging Type', 'Charging Speed', 'GHG Cost'])

plotting(pd.concat([total_costs_df_TOUH_HW, total_costs_df_norm]), 50)

# %%
import matplotlib.pyplot as plt

# Calculate the total energy injected back to the grid for each pricing method
hourly_data_discharging_RT_df = hourly_data_discharging_RTH[hourly_data_discharging_RTH["Charging Speed"] == 12]
hourly_data_discharging_TOU_df = hourly_data_discharging_TOUH[hourly_data_discharging_TOUH["Charging Speed"] == 12]


total_energy_RT = hourly_data_discharging_RT_df["X_CHR"].sum()
total_energy_TOU = hourly_data_discharging_TOU_df["X_CHR"].sum()

# Data for plotting
categories = ['Real-Time Pricing', 'TOU Pricing']
totals = [abs(total_energy_RT), abs(total_energy_TOU)]

# Create the bar chart
fig, ax = plt.subplots(figsize=(8, 6))
bars = ax.bar(categories, totals, color=['blue', 'green'])

# Add labels and title
ax.set_xlabel('Pricing Method', fontsize=14)
ax.set_ylabel('Total Energy Injected (kWh)', fontsize=14)
ax.set_title('Total Energy Injected Back to the Grid via V2G', fontsize=16)

# Add values on top of the bars
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.2f}', va='bottom', ha='center', fontsize=12)

# Increase the size of tick labels
ax.tick_params(axis='both', which='major', labelsize=12)
ax.set_ylim([0, 2e06])
plt.tight_layout()
plt.show()


# %%
draw_parking(final_dataframes)
draw_charging(final_dataframes)
draw_combined(final_dataframes)
draw_multilevel_pie(final_dataframes)
draw_rose_chart_parking(final_dataframes, text_size=18)
draw_rose_chart_charging(final_dataframes, text_size=18)
# # %%
# # Plot the grid of stacked bar charts
# dr_plot(result_RTH_DR1, result_RTHW_DR1)
# dr_plot(result_RTH_DR10, result_RTHW_DR10)
# dr_plot(result_RTH_DR20, result_RTHW_DR20)
#
# dr_plot(result_TOUH_DR1, result_TOUHW_DR1)
# dr_plot(result_TOUH_DR10, result_TOUHW_DR10)
# dr_plot(result_TOUH_DR20, result_TOUHW_DR20)
# %%



