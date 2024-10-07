# %%
import pandas as pd
import warnings
import json
import os
import logging
from parking import (xlsx_read, json_file_read, plot_price_chart_EVRATE, plot_price_chart_TOU, plot_cost_comparison_EV, plot_cost_comparison_TOU, 
                     plot_cost_comparison_RT, stacked_violin_plot, add_tariff_name, plot_ghg_distribution_seasons, violin_input, filter_rows_by_word, add_tariff_name2,add_tariff_name3,
                     process_charging_data, plotting_demand_heatmap, demand_response, dr_plot, process_charging_data1)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.interpolate import griddata
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
# %%
GHG_data = pd.read_csv("CISO.csv")
Power_data = pd.read_csv("CISO_power.csv")

GHG_dict = dict(enumerate(GHG_data.iloc[:, 0]))
Power_dict = dict(enumerate(Power_data.iloc[:, 0]))

df_GHG = pd.DataFrame(list(GHG_dict.items()), columns=['Hour', 'GHG_value'])

with open("merged_dict.json", "r") as json_file:
    merged_dict = json.load(json_file)

with open("merged_dict_parking.json", "r") as json_file:
    merged_dict_parking = json.load(json_file)

# Flatten the dictionary
flattened_data = []
for vehicle, hours in merged_dict.items():
    for hour, values in hours.items():
        entry = {'Vehicle': vehicle, 'Hour': int(hour)}
        entry.update(values)
        flattened_data.append(entry)

# Create DataFrame
flatten_veh_data = pd.DataFrame(flattened_data)

# Flatten the dictionary
flattened_data_parking = []
for vehicle, hours in merged_dict.items():
    for hour, values in hours.items():
        entry = {'Vehicle': vehicle, 'Hour': int(hour)}
        entry.update(values)
        flattened_data_parking.append(entry)

# Create DataFrame
flatten_veh_data_parking = pd.DataFrame(flattened_data_parking)

# %% reading json file

# Normal Behavior
directory_N_ng_EV_rate_H = "/Users/haniftayarani/V2G_Project/Results/Normal_Result_no_ghg/EV_Rate_Result/Home"
total_costs_N_ng_EV_rate_H, costs_N_ng_EV_rate_H = xlsx_read(directory_N_ng_EV_rate_H)
total_costs_N_ng_EV_rate_H["Plugged-in Sessions"] = "Actual"
total_costs_N_ng_EV_rate_H["V2G_Location"] = "Home"
costs_N_ng_EV_rate_H["Plugged-in Sessions"] = "Actual"
costs_N_ng_EV_rate_H["V2G_Location"] = "Home"
total_costs_N_ng_EV_rate_H["GHG Cost"] = 0
costs_N_ng_EV_rate_H["GHG Cost"] = 0

directory_N_ng_EV_rate_HW = "/Users/haniftayarani/V2G_Project/Results/Normal_Result_no_ghg/EV_Rate_Result/Home_Work"
total_costs_N_ng_EV_rate_HW, costs_N_ng_EV_rate_HW = xlsx_read(directory_N_ng_EV_rate_HW)
total_costs_N_ng_EV_rate_HW["Plugged-in Sessions"] = "Actual"
total_costs_N_ng_EV_rate_HW["V2G_Location"] = "Home_Work"
costs_N_ng_EV_rate_HW["Plugged-in Sessions"] = "Actual"
costs_N_ng_EV_rate_HW["V2G_Location"] = "Home_Work"
total_costs_N_ng_EV_rate_HW["GHG Cost"] = 0
costs_N_ng_EV_rate_HW["GHG Cost"] = 0

directory_N_ng_RT_rate_H = "/Users/haniftayarani/V2G_Project/Results/Normal_Result_no_ghg/RT_Result/Home"
total_costs_N_ng_RT_rate_H, costs_N_ng_RT_rate_H = xlsx_read(directory_N_ng_RT_rate_H)
total_costs_N_ng_RT_rate_H["Plugged-in Sessions"] = "Actual"
total_costs_N_ng_RT_rate_H["V2G_Location"] = "Home"
costs_N_ng_RT_rate_H["Plugged-in Sessions"] = "Actual"
costs_N_ng_RT_rate_H["V2G_Location"] = "Home"
total_costs_N_ng_RT_rate_H["GHG Cost"] = 0
costs_N_ng_RT_rate_H["GHG Cost"] = 0

directory_N_ng_RT_rate_HW = "/Users/haniftayarani/V2G_Project/Results/Normal_Result_no_ghg/RT_Result/Home_Work"
total_costs_N_ng_RT_rate_HW, costs_N_ng_RT_rate_HW = xlsx_read(directory_N_ng_RT_rate_HW)
total_costs_N_ng_RT_rate_HW["Plugged-in Sessions"] = "Actual"
total_costs_N_ng_RT_rate_HW["V2G_Location"] = "Home_Work"
costs_N_ng_RT_rate_HW["Plugged-in Sessions"] = "Actual"
costs_N_ng_RT_rate_HW["V2G_Location"] = "Home_Work"
total_costs_N_ng_RT_rate_HW["GHG Cost"] = 0
costs_N_ng_RT_rate_HW["GHG Cost"] = 0

directory_N_ng_TOU_rate_H = "/Users/haniftayarani/V2G_Project/Results/Normal_Result_no_ghg/TOU_Result/Home"
total_costs_N_ng_TOU_rate_H, costs_N_ng_TOU_rate_H = xlsx_read(directory_N_ng_TOU_rate_H)
total_costs_N_ng_TOU_rate_H["Plugged-in Sessions"] = "Actual"
total_costs_N_ng_TOU_rate_H["V2G_Location"] = "Home"
costs_N_ng_TOU_rate_H["Plugged-in Sessions"] = "Actual"
costs_N_ng_TOU_rate_H["V2G_Location"] = "Home"
total_costs_N_ng_TOU_rate_H["GHG Cost"] = 0
costs_N_ng_TOU_rate_H["GHG Cost"] = 0

directory_N_ng_TOU_rate_HW = "/Users/haniftayarani/V2G_Project/Results/Normal_Result_no_ghg/TOU_Result/Home_Work"
total_costs_N_ng_TOU_rate_HW, costs_N_ng_TOU_rate_HW = xlsx_read(directory_N_ng_TOU_rate_HW)
total_costs_N_ng_TOU_rate_HW["Plugged-in Sessions"] = "Actual"
total_costs_N_ng_TOU_rate_HW["V2G_Location"] = "Home_Work"
costs_N_ng_TOU_rate_HW["Plugged-in Sessions"] = "Actual"
costs_N_ng_TOU_rate_HW["V2G_Location"] = "Home_Work"
total_costs_N_ng_TOU_rate_HW["GHG Cost"] = 0
costs_N_ng_TOU_rate_HW["GHG Cost"] = 0

# 50 gram
directory_N_50g_EV_rate_H = "/Users/haniftayarani/V2G_Project/Results/Normal_Result_50_ghg/EV_Rate_Result/Home"
total_costs_N_50g_EV_rate_H, costs_N_50g_EV_rate_H = xlsx_read(directory_N_50g_EV_rate_H)
total_costs_N_50g_EV_rate_H["Plugged-in Sessions"] = "Actual"
total_costs_N_50g_EV_rate_H["V2G_Location"] = "Home"
costs_N_50g_EV_rate_H["Plugged-in Sessions"] = "Actual"
costs_N_50g_EV_rate_H["V2G_Location"] = "Home"


directory_N_50g_EV_rate_HW = "/Users/haniftayarani/V2G_Project/Results/Normal_Result_50_ghg/EV_Rate_Result/Home_Work"
total_costs_N_50g_EV_rate_HW, costs_N_50g_EV_rate_HW = xlsx_read(directory_N_50g_EV_rate_HW)
total_costs_N_50g_EV_rate_HW["Plugged-in Sessions"] = "Actual"
total_costs_N_50g_EV_rate_HW["V2G_Location"] = "Home_Work"
costs_N_50g_EV_rate_HW["Plugged-in Sessions"] = "Actual"
costs_N_50g_EV_rate_HW["V2G_Location"] = "Home_Work"


directory_N_50g_RT_rate_H = "/Users/haniftayarani/V2G_Project/Results/Normal_Result_50_ghg/RT_Result/Home"
total_costs_N_50g_RT_rate_H, costs_N_50g_RT_rate_H = xlsx_read(directory_N_50g_RT_rate_H)
total_costs_N_50g_RT_rate_H["Plugged-in Sessions"] = "Actual"
total_costs_N_50g_RT_rate_H["V2G_Location"] = "Home"
costs_N_50g_RT_rate_H["Plugged-in Sessions"] = "Actual"
costs_N_50g_RT_rate_H["V2G_Location"] = "Home"

directory_N_50g_RT_rate_HW = "/Users/haniftayarani/V2G_Project/Results/Normal_Result_50_ghg/RT_Result/Home_Work"
total_costs_N_50g_RT_rate_HW, costs_N_50g_RT_rate_HW = xlsx_read(directory_N_50g_RT_rate_HW)
total_costs_N_50g_RT_rate_HW["Plugged-in Sessions"] = "Actual"
total_costs_N_50g_RT_rate_HW["V2G_Location"] = "Home_Work"
costs_N_50g_RT_rate_HW["Plugged-in Sessions"] = "Actual"
costs_N_50g_RT_rate_HW["V2G_Location"] = "Home_Work"

directory_N_50g_TOU_rate_H = "/Users/haniftayarani/V2G_Project/Results/Normal_Result_50_ghg/TOU_Result/Home"
total_costs_N_50g_TOU_rate_H, costs_N_50g_TOU_rate_H = xlsx_read(directory_N_50g_TOU_rate_H)
total_costs_N_50g_TOU_rate_H["Plugged-in Sessions"] = "Actual"
total_costs_N_50g_TOU_rate_H["V2G_Location"] = "Home"
costs_N_50g_TOU_rate_H["Plugged-in Sessions"] = "Actual"
costs_N_50g_TOU_rate_H["V2G_Location"] = "Home"


directory_N_50g_TOU_rate_HW = "/Users/haniftayarani/V2G_Project/Results/Normal_Result_50_ghg/TOU_Result/Home_Work"
total_costs_N_50g_TOU_rate_HW, costs_N_50g_TOU_rate_HW = xlsx_read(directory_N_50g_TOU_rate_HW)
total_costs_N_50g_TOU_rate_HW["Plugged-in Sessions"] = "Actual"
total_costs_N_50g_TOU_rate_HW["V2G_Location"] = "Home_Work"
costs_N_50g_TOU_rate_HW["Plugged-in Sessions"] = "Actual"
costs_N_50g_TOU_rate_HW["V2G_Location"] = "Home_Work"

# 191 gram

directory_N_191g_EV_rate_H = "/Users/haniftayarani/V2G_Project/Results/Normal_Result_191_ghg/EV_Rate_Result/Home"
total_costs_N_191g_EV_rate_H, costs_N_191g_EV_rate_H = xlsx_read(directory_N_191g_EV_rate_H)
total_costs_N_191g_EV_rate_H["Plugged-in Sessions"] = "Actual"
total_costs_N_191g_EV_rate_H["V2G_Location"] = "Home"
costs_N_191g_EV_rate_H["Plugged-in Sessions"] = "Actual"
costs_N_191g_EV_rate_H["V2G_Location"] = "Home"

directory_N_191g_EV_rate_HW = "/Users/haniftayarani/V2G_Project/Results/Normal_Result_191_ghg/EV_Rate_Result/Home_Work"
total_costs_N_191g_EV_rate_HW, costs_N_191g_EV_rate_HW = xlsx_read(directory_N_191g_EV_rate_HW)
total_costs_N_191g_EV_rate_HW["Plugged-in Sessions"] = "Actual"
total_costs_N_191g_EV_rate_HW["V2G_Location"] = "Home_Work"
costs_N_191g_EV_rate_HW["Plugged-in Sessions"] = "Actual"
costs_N_191g_EV_rate_HW["V2G_Location"] = "Home_Work"

directory_N_191g_RT_rate_H = "/Users/haniftayarani/V2G_Project/Results/Normal_Result_191_ghg/RT_Result/Home"
total_costs_N_191g_RT_rate_H, costs_N_191g_RT_rate_H = xlsx_read(directory_N_191g_RT_rate_H)
total_costs_N_191g_RT_rate_H["Plugged-in Sessions"] = "Actual"
total_costs_N_191g_RT_rate_H["V2G_Location"] = "Home"
costs_N_191g_RT_rate_H["Plugged-in Sessions"] = "Actual"
costs_N_191g_RT_rate_H["V2G_Location"] = "Home"

directory_N_191g_RT_rate_HW = "/Users/haniftayarani/V2G_Project/Results/Normal_Result_191_ghg/RT_Result/Home_Work"
total_costs_N_191g_RT_rate_HW, costs_N_191g_RT_rate_HW = xlsx_read(directory_N_191g_RT_rate_HW)
total_costs_N_191g_RT_rate_HW["Plugged-in Sessions"] = "Actual"
total_costs_N_191g_RT_rate_HW["V2G_Location"] = "Home_Work"
costs_N_191g_RT_rate_HW["Plugged-in Sessions"] = "Actual"
costs_N_191g_RT_rate_HW["V2G_Location"] = "Home_Work"

directory_N_191g_TOU_rate_H = "/Users/haniftayarani/V2G_Project/Results/Normal_Result_191_ghg/TOU_Result/Home"
total_costs_N_191g_TOU_rate_H, costs_N_191g_TOU_rate_H = xlsx_read(directory_N_191g_TOU_rate_H)
total_costs_N_191g_TOU_rate_H["Plugged-in Sessions"] = "Actual"
total_costs_N_191g_TOU_rate_H["V2G_Location"] = "Home"
costs_N_191g_TOU_rate_H["Plugged-in Sessions"] = "Actual"
costs_N_191g_TOU_rate_H["V2G_Location"] = "Home"

directory_N_191g_TOU_rate_HW = "/Users/haniftayarani/V2G_Project/Results/Normal_Result_191_ghg/TOU_Result/Home_Work"
total_costs_N_191g_TOU_rate_HW, costs_N_191g_TOU_rate_HW = xlsx_read(directory_N_191g_TOU_rate_HW)
total_costs_N_191g_TOU_rate_HW["Plugged-in Sessions"] = "Actual"
total_costs_N_191g_TOU_rate_HW["V2G_Location"] = "Home_Work"
costs_N_191g_TOU_rate_HW["Plugged-in Sessions"] = "Actual"
costs_N_191g_TOU_rate_HW["V2G_Location"] = "Home_Work"


####

# Potential  Behavior
directory_P_ng_EV_rate_H = "/Users/haniftayarani/V2G_Project/Results/Parking_Result_no_ghg/EV_Rate_Result/Home"
total_costs_P_ng_EV_rate_H, costs_P_ng_EV_rate_H = xlsx_read(directory_P_ng_EV_rate_H)
total_costs_P_ng_EV_rate_H["Plugged-in Sessions"] = "Potential"
total_costs_P_ng_EV_rate_H["V2G_Location"] = "Home"
total_costs_P_ng_EV_rate_H["GHG Cost"] = 0
costs_P_ng_EV_rate_H["Plugged-in Sessions"] = "Potential"
costs_P_ng_EV_rate_H["V2G_Location"] = "Home"
costs_P_ng_EV_rate_H["GHG Cost"] = 0


directory_P_ng_EV_rate_HW = "/Users/haniftayarani/V2G_Project/Results/Parking_Result_no_ghg/EV_Rate_Result/Home_Work"
total_costs_P_ng_EV_rate_HW, costs_P_ng_EV_rate_HW = xlsx_read(directory_P_ng_EV_rate_HW)
total_costs_P_ng_EV_rate_HW["Plugged-in Sessions"] = "Potential"
total_costs_P_ng_EV_rate_HW["V2G_Location"] = "Home_Work"
total_costs_P_ng_EV_rate_HW["GHG Cost"] = 0
costs_P_ng_EV_rate_HW["Plugged-in Sessions"] = "Potential"
costs_P_ng_EV_rate_HW["V2G_Location"] = "Home_Work"
costs_P_ng_EV_rate_HW["GHG Cost"] = 0


directory_P_ng_RT_rate_H = "/Users/haniftayarani/V2G_Project/Results/Parking_Result_no_ghg/RT_Result/Home"
total_costs_P_ng_RT_rate_H, costs_P_ng_RT_rate_H = xlsx_read(directory_P_ng_RT_rate_H)
total_costs_P_ng_RT_rate_H["Plugged-in Sessions"] = "Potential"
total_costs_P_ng_RT_rate_H["V2G_Location"] = "Home"
total_costs_P_ng_RT_rate_H["GHG Cost"] = 0
costs_P_ng_RT_rate_H["Plugged-in Sessions"] = "Potential"
costs_P_ng_RT_rate_H["V2G_Location"] = "Home"
costs_P_ng_RT_rate_H["GHG Cost"] = 0


directory_P_ng_RT_rate_HW = "/Users/haniftayarani/V2G_Project/Results/Parking_Result_no_ghg/RT_Result/Home_Work"
total_costs_P_ng_RT_rate_HW, costs_P_ng_RT_rate_HW = xlsx_read(directory_P_ng_RT_rate_HW)
total_costs_P_ng_RT_rate_HW["Plugged-in Sessions"] = "Potential"
total_costs_P_ng_RT_rate_HW["V2G_Location"] = "Home_Work"
total_costs_P_ng_RT_rate_HW["GHG Cost"] = 0
costs_P_ng_RT_rate_HW["Plugged-in Sessions"] = "Potential"
costs_P_ng_RT_rate_HW["V2G_Location"] = "Home_Work"
costs_P_ng_RT_rate_HW["GHG Cost"] = 0


directory_P_ng_TOU_rate_H = "/Users/haniftayarani/V2G_Project/Results/Parking_Result_no_ghg/TOU_Result/Home"

total_costs_P_ng_TOU_rate_H, costs_P_ng_TOU_rate_H = xlsx_read(directory_P_ng_TOU_rate_H)
total_costs_P_ng_TOU_rate_H["Plugged-in Sessions"] = "Potential"
total_costs_P_ng_TOU_rate_H["V2G_Location"] = "Home"
total_costs_P_ng_TOU_rate_H["GHG Cost"] = 0
costs_P_ng_TOU_rate_H["Plugged-in Sessions"] = "Potential"
costs_P_ng_TOU_rate_H["V2G_Location"] = "Home"
costs_P_ng_TOU_rate_H["GHG Cost"] = 0

directory_P_ng_TOU_rate_HW = "/Users/haniftayarani/V2G_Project/Results/Parking_Result_no_ghg/TOU_Result/Home_Work"
total_costs_P_ng_TOU_rate_HW, costs_P_ng_TOU_rate_HW = xlsx_read(directory_P_ng_TOU_rate_HW)
total_costs_P_ng_TOU_rate_HW["Plugged-in Sessions"] = "Potential"
total_costs_P_ng_TOU_rate_HW["V2G_Location"] = "Home_Work"
total_costs_P_ng_TOU_rate_HW["GHG Cost"] = 0
costs_P_ng_TOU_rate_HW["Plugged-in Sessions"] = "Potential"
costs_P_ng_TOU_rate_HW["V2G_Location"] = "Home_Work"
costs_P_ng_TOU_rate_HW["GHG Cost"] = 0

# 50 gram
directory_P_50g_EV_rate_H = "/Users/haniftayarani/V2G_Project/Results/Parking_Result_50_ghg/EV_Rate_Result/Home"
total_costs_P_50g_EV_rate_H, costs_P_50g_EV_rate_H = xlsx_read(directory_P_50g_EV_rate_H)
total_costs_P_50g_EV_rate_H["Plugged-in Sessions"] = "Potential"
total_costs_P_50g_EV_rate_H["V2G_Location"] = "Home"
costs_P_50g_EV_rate_H["Plugged-in Sessions"] = "Potential"
costs_P_50g_EV_rate_H["V2G_Location"] = "Home"

directory_P_50g_EV_rate_HW = "/Users/haniftayarani/V2G_Project/Results/Parking_Result_50_ghg/EV_Rate_Result/Home_Work"
total_costs_P_50g_EV_rate_HW, costs_P_50g_EV_rate_HW = xlsx_read(directory_P_50g_EV_rate_HW)
total_costs_P_50g_EV_rate_HW["Plugged-in Sessions"] = "Potential"
total_costs_P_50g_EV_rate_HW["V2G_Location"] = "Home_Work"
costs_P_50g_EV_rate_HW["Plugged-in Sessions"] = "Potential"
costs_P_50g_EV_rate_HW["V2G_Location"] = "Home_Work"

directory_P_50g_RT_rate_H = "/Users/haniftayarani/V2G_Project/Results/Parking_Result_50_ghg/RT_Result/Home"
total_costs_P_50g_RT_rate_H, costs_P_50g_RT_rate_H = xlsx_read(directory_P_50g_RT_rate_H)
total_costs_P_50g_RT_rate_H["Plugged-in Sessions"] = "Potential"
total_costs_P_50g_RT_rate_H["V2G_Location"] = "Home"
costs_P_50g_RT_rate_H["Plugged-in Sessions"] = "Potential"
costs_P_50g_RT_rate_H["V2G_Location"] = "Home"

directory_P_50g_RT_rate_HW = "/Users/haniftayarani/V2G_Project/Results/Parking_Result_50_ghg/RT_Result/Home_Work"
total_costs_P_50g_RT_rate_HW, costs_P_50g_RT_rate_HW = xlsx_read(directory_P_50g_RT_rate_HW)
total_costs_P_50g_RT_rate_HW["Plugged-in Sessions"] = "Potential"
total_costs_P_50g_RT_rate_HW["V2G_Location"] = "Home_Work"
costs_P_50g_RT_rate_HW["Plugged-in Sessions"] = "Potential"
costs_P_50g_RT_rate_HW["V2G_Location"] = "Home_Work"

directory_P_50g_TOU_rate_H = "/Users/haniftayarani/V2G_Project/Results/Parking_Result_50_ghg/TOU_Result/Home"
total_costs_P_50g_TOU_rate_H, costs_P_50g_TOU_rate_H = xlsx_read(directory_P_50g_TOU_rate_H)
total_costs_P_50g_TOU_rate_H["Plugged-in Sessions"] = "Potential"
total_costs_P_50g_TOU_rate_H["V2G_Location"] = "Home"
costs_P_50g_TOU_rate_H["Plugged-in Sessions"] = "Potential"
costs_P_50g_TOU_rate_H["V2G_Location"] = "Home"

directory_P_50g_TOU_rate_HW = "/Users/haniftayarani/V2G_Project/Results/Parking_Result_50_ghg/TOU_Result/Home_Work"
total_costs_P_50g_TOU_rate_HW, costs_P_50g_TOU_rate_HW = xlsx_read(directory_P_50g_TOU_rate_HW)
total_costs_P_50g_TOU_rate_HW["Plugged-in Sessions"] = "Potential"
total_costs_P_50g_TOU_rate_HW["V2G_Location"] = "Home_Work"
costs_P_50g_TOU_rate_HW["Plugged-in Sessions"] = "Potential"
costs_P_50g_TOU_rate_HW["V2G_Location"] = "Home_Work"

# 191 gram

directory_P_191g_EV_rate_H = "/Users/haniftayarani/V2G_Project/Results/Parking_Result_191_ghg/EV_Rate_Result/Home"
total_costs_P_191g_EV_rate_H, costs_P_191g_EV_rate_H = xlsx_read(directory_P_191g_EV_rate_H)
total_costs_P_191g_EV_rate_H["Plugged-in Sessions"] = "Potential"
total_costs_P_191g_EV_rate_H["V2G_Location"] = "Home"
costs_P_191g_EV_rate_H["Plugged-in Sessions"] = "Potential"
costs_P_191g_EV_rate_H["V2G_Location"] = "Home"


directory_P_191g_EV_rate_HW = "/Users/haniftayarani/V2G_Project/Results/Parking_Result_191_ghg/EV_Rate_Result/Home_Work"
total_costs_P_191g_EV_rate_HW, costs_P_191g_EV_rate_HW = xlsx_read(directory_P_191g_EV_rate_HW)
total_costs_P_191g_EV_rate_HW["Plugged-in Sessions"] = "Potential"
total_costs_P_191g_EV_rate_HW["V2G_Location"] = "Home_Work"
costs_P_191g_EV_rate_HW["Plugged-in Sessions"] = "Potential"
costs_P_191g_EV_rate_HW["V2G_Location"] = "Home_Work"

directory_P_191g_RT_rate_H = "/Users/haniftayarani/V2G_Project/Results/Parking_Result_191_ghg/RT_Result/Home"
total_costs_P_191g_RT_rate_H, costs_P_191g_RT_rate_H = xlsx_read(directory_P_191g_RT_rate_H)
total_costs_P_191g_RT_rate_H["Plugged-in Sessions"] = "Potential"
total_costs_P_191g_RT_rate_H["V2G_Location"] = "Home"
costs_P_191g_RT_rate_H["Plugged-in Sessions"] = "Potential"
costs_P_191g_RT_rate_H["V2G_Location"] = "Home"

directory_P_191g_RT_rate_HW = "/Users/haniftayarani/V2G_Project/Results/Parking_Result_191_ghg/RT_Result/Home_Work"
total_costs_P_191g_RT_rate_HW, costs_P_191g_RT_rate_HW = xlsx_read(directory_P_191g_RT_rate_HW)
total_costs_P_191g_RT_rate_HW["Plugged-in Sessions"] = "Potential"
total_costs_P_191g_RT_rate_HW["V2G_Location"] = "Home_Work"
costs_P_191g_RT_rate_HW["Plugged-in Sessions"] = "Potential"
costs_P_191g_RT_rate_HW["V2G_Location"] = "Home_Work"


directory_P_191g_TOU_rate_H = "/Users/haniftayarani/V2G_Project/Results/Parking_Result_191_ghg/TOU_Result/Home"
total_costs_P_191g_TOU_rate_H, costs_P_191g_TOU_rate_H = xlsx_read(directory_P_191g_TOU_rate_H)
total_costs_P_191g_TOU_rate_H["Plugged-in Sessions"] = "Potential"
total_costs_P_191g_TOU_rate_H["V2G_Location"] = "Home"
costs_P_191g_TOU_rate_H["Plugged-in Sessions"] = "Potential"
costs_P_191g_TOU_rate_H["V2G_Location"] = "Home"


directory_P_191g_TOU_rate_HW = "/Users/haniftayarani/V2G_Project/Results/Parking_Result_191_ghg/TOU_Result/Home_Work"
total_costs_P_191g_TOU_rate_HW, costs_P_191g_TOU_rate_HW = xlsx_read(directory_P_191g_TOU_rate_HW)
total_costs_P_191g_TOU_rate_HW["Plugged-in Sessions"] = "Potential"
total_costs_P_191g_TOU_rate_HW["V2G_Location"] = "Home_Work"
costs_P_191g_TOU_rate_HW["Plugged-in Sessions"] = "Potential"
costs_P_191g_TOU_rate_HW["V2G_Location"] = "Home_Work"


# Actual
# Define the file paths
rt_cost_file = '/Users/haniftayarani/V2G_Project/Results/Actual/Actual_RT_cost.xlsx'
tou_cost_file = '/Users/haniftayarani/V2G_Project/Results/Actual/Actual_TOU_cost.xlsx'
ev_rate_cost_file = '/Users/haniftayarani/V2G_Project/Results/Actual/Actual_EV_rate_cost.xlsx'

# Read the Excel files into DataFrames
costs_A_RT_rate = pd.read_excel(rt_cost_file, sheet_name='Individual Costs')
costs_A_TOU_rate = pd.read_excel(tou_cost_file, sheet_name='Individual Costs')
costs_A_EV_rate = pd.read_excel(ev_rate_cost_file, sheet_name='Individual Costs')

costs_A_RT_rate = costs_A_RT_rate.drop(costs_A_RT_rate.columns[0], axis=1)
costs_A_TOU_rate = costs_A_TOU_rate.drop(costs_A_TOU_rate.columns[0], axis=1)
costs_A_EV_rate = costs_A_EV_rate.drop(costs_A_EV_rate.columns[0], axis=1)

costs_A_RT_rate["Charging Type"] = "Actual Behavior - RT"
costs_A_TOU_rate["Charging Type"] = "Actual - TOU"
costs_A_EV_rate["Charging Type"] = "Actual - EV Rate"

costs_A_RT_rate["GHG Cost"] = 0.05
costs_A_TOU_rate["GHG Cost"] = 0.05
costs_A_EV_rate["GHG Cost"] = 0.05

costs_A_RT_rate["V2G_Location"] = "None"
costs_A_TOU_rate["V2G_Location"] = "None"
costs_A_EV_rate["V2G_Location"] = "None"

costs_A_RT_rate["Plugged-in Sessions"] = "Actual"
costs_A_TOU_rate["Plugged-in Sessions"] = "Actual"
costs_A_EV_rate["Plugged-in Sessions"] = "Actual"

costs_A_RT_rate_191 = costs_A_RT_rate.copy()
costs_A_TOU_rate_191 = costs_A_TOU_rate.copy()
costs_A_EV_rate_191 = costs_A_EV_rate.copy()

costs_A_RT_rate_191["GHG Cost"] = 0.191
costs_A_TOU_rate_191["GHG Cost"] = 0.191
costs_A_EV_rate_191["GHG Cost"] = 0.191

costs_A_RT_rate_191["GHG_Cost"] = (costs_A_RT_rate_191["GHG_Cost"]/0.05) * 0.191
costs_A_TOU_rate_191["GHG_Cost"] = (costs_A_TOU_rate_191["GHG_Cost"]/0.05) * 0.191
costs_A_EV_rate_191["GHG_Cost"] = (costs_A_EV_rate_191["GHG_Cost"]/0.05) * 0.191

# Create total costs DataFrames with index
total_costs_A_RT_rate = pd.DataFrame({
    "Electricity_Cost": [costs_A_RT_rate["Electricity_Cost"].sum()],
    "Degradation_Cost": [costs_A_RT_rate["Degradation_Cost"].sum()],
    "GHG_Cost": [costs_A_RT_rate["GHG_Cost"].sum()],
    "X_CHR": [costs_A_RT_rate["Total Charge"].sum()],
    "Charging Type": ["Actual Behavior"],
    "Charging Speed": [0],
    "GHG Cost": [0]
})

total_costs_A_TOU_rate = pd.DataFrame({
    "Electricity_Cost": [costs_A_TOU_rate["Electricity_Cost"].sum()],
    "Degradation_Cost": [costs_A_TOU_rate["Degradation_Cost"].sum()],
    "GHG_Cost": [costs_A_TOU_rate["GHG_Cost"].sum()],
    "X_CHR": [costs_A_TOU_rate["Total Charge"].sum()],
    "Charging Type": ["Actual Behavior - TOU Rate"],
    "Charging Speed": [0],
    "GHG Cost": [0]
})

total_costs_A_EV_rate = pd.DataFrame({
    "Electricity_Cost": [costs_A_EV_rate["Electricity_Cost"].sum()],
    "Degradation_Cost": [costs_A_EV_rate["Degradation_Cost"].sum()],
    "GHG_Cost": [costs_A_EV_rate["GHG_Cost"].sum()],
    "X_CHR": [costs_A_EV_rate["Total Charge"].sum()],
    "Charging Type": ["Actual Behavior - EV Rate"],
    "Charging Speed": [0],
    "GHG Cost": [0]
})


rt_cost_file_hourly = '/Users/haniftayarani/V2G_Project/Results/Actual/Actual_RT_cost_hourly.xlsx'
tou_cost_file_hourly = '/Users/haniftayarani/V2G_Project/Results/Actual/Actual_TOU_cost_hourly.xlsx'
ev_rate_cost_file_hourly = '/Users/haniftayarani/V2G_Project/Results/Actual/Actual_EV_rate_cost_hourly.xlsx'

costs_A_RT_rate_hourly = pd.read_excel(rt_cost_file_hourly, sheet_name='Individual Costs')
costs_A_TOU_rate_hourly = pd.read_excel(tou_cost_file_hourly, sheet_name='Individual Costs')
costs_A_EV_rate_hourly = pd.read_excel(ev_rate_cost_file_hourly, sheet_name='Individual Costs')

costs_A_RT_rate_hourly = costs_A_RT_rate_hourly.drop(costs_A_RT_rate_hourly.columns[0], axis=1)
costs_A_TOU_rate_hourly = costs_A_TOU_rate_hourly.drop(costs_A_TOU_rate_hourly.columns[0], axis=1)
costs_A_EV_rate_hourly = costs_A_EV_rate_hourly.drop(costs_A_EV_rate_hourly.columns[0], axis=1)


rt_cost_file_hourly_in = '/Users/haniftayarani/V2G_Project/Results/Actual/Actual_RT_cost_hourly_in.xlsx'
tou_cost_file_hourly_in = '/Users/haniftayarani/V2G_Project/Results/Actual/Actual_TOU_cost_hourly_in.xlsx'
ev_rate_cost_file_hourly_in = '/Users/haniftayarani/V2G_Project/Results/Actual/Actual_EV_rate_cost_hourly_in.xlsx'

costs_A_RT_rate_hourly_in = pd.read_excel(rt_cost_file_hourly_in, sheet_name='Individual Costs')
costs_A_TOU_rate_hourly_in = pd.read_excel(tou_cost_file_hourly_in, sheet_name='Individual Costs')
costs_A_EV_rate_hourly_in = pd.read_excel(ev_rate_cost_file_hourly_in, sheet_name='Individual Costs')

costs_A_RT_rate_hourly_in = costs_A_RT_rate_hourly_in.drop(costs_A_RT_rate_hourly_in.columns[0], axis=1)
costs_A_TOU_rate_hourly_in = costs_A_TOU_rate_hourly_in.drop(costs_A_TOU_rate_hourly_in.columns[0], axis=1)
costs_A_EV_rate_hourly_in = costs_A_EV_rate_hourly_in.drop(costs_A_EV_rate_hourly_in.columns[0], axis=1)
# %%
result_N_ng_EV_rate_H, hourly_data_N_ng_EV_rate_H = json_file_read(directory_N_ng_EV_rate_H, flatten_veh_data)
# result_N_ng_EV_rate_HW, hourly_data_N_ng_EV_rate_HW = json_file_read(directory_N_ng_EV_rate_HW, flatten_veh_data)
result_N_ng_RT_rate_H, hourly_data_N_ng_RT_rate_H = json_file_read(directory_N_ng_RT_rate_H, flatten_veh_data)
# result_N_ng_RT_rate_HW, hourly_data_N_ng_RT_rate_HW = json_file_read(directory_N_ng_RT_rate_HW, flatten_veh_data)
result_N_ng_TOU_rate_H, hourly_data_N_ng_TOU_rate_H = json_file_read(directory_N_ng_TOU_rate_H, flatten_veh_data)
# result_N_ng_TOU_rate_HW, hourly_data_N_ng_TOU_rate_HW = json_file_read(directory_N_ng_TOU_rate_HW, flatten_veh_data)
result_N_50g_EV_rate_H, hourly_data_N_50g_EV_rate_H = json_file_read(directory_N_50g_EV_rate_H, flatten_veh_data)
# result_N_50g_EV_rate_HW, hourly_data_N_50g_EV_rate_HW = json_file_read(directory_N_50g_EV_rate_HW, flatten_veh_data)
result_N_50g_RT_rate_H, hourly_data_N_50g_RT_rate_H = json_file_read(directory_N_50g_RT_rate_H, flatten_veh_data)
# result_N_50g_RT_rate_HW, hourly_data_N_50g_RT_rate_HW = json_file_read(directory_N_50g_RT_rate_HW, flatten_veh_data)
result_N_50g_TOU_rate_H, hourly_data_N_50g_TOU_rate_H = json_file_read(directory_N_50g_TOU_rate_H, flatten_veh_data)
# result_N_50g_TOU_rate_HW, hourly_data_N_50g_TOU_rate_HW = json_file_read(directory_N_50g_TOU_rate_HW, flatten_veh_data)
result_N_191g_EV_rate_H, hourly_data_N_191g_EV_rate_H = json_file_read(directory_N_191g_EV_rate_H, flatten_veh_data)
# result_N_191g_EV_rate_HW, hourly_data_N_191g_EV_rate_HW = json_file_read(directory_N_191g_EV_rate_HW, flatten_veh_data)
result_N_191g_RT_rate_H, hourly_data_N_191g_RT_rate_H = json_file_read(directory_N_191g_RT_rate_H, flatten_veh_data)
# result_N_191g_RT_rate_HW, hourly_data_N_191g_RT_rate_HW = json_file_read(directory_N_191g_RT_rate_HW, flatten_veh_data)
result_N_191g_TOU_rate_H, hourly_data_N_191g_TOU_rate_H = json_file_read(directory_N_191g_TOU_rate_H, flatten_veh_data)
# result_N_191g_TOU_rate_HW, hourly_data_N_191g_TOU_rate_HW = json_file_read(directory_N_191g_TOU_rate_HW, flatten_veh_data)
result_P_ng_EV_rate_H, hourly_data_P_ng_EV_rate_H = json_file_read(directory_P_ng_EV_rate_H, flatten_veh_data_parking)
# result_P_ng_EV_rate_HW, hourly_data_P_ng_EV_rate_HW = json_file_read(directory_P_ng_EV_rate_HW, flatten_veh_data_parking)
result_P_ng_RT_rate_H, hourly_data_P_ng_RT_rate_H = json_file_read(directory_P_ng_RT_rate_H, flatten_veh_data_parking)
# result_P_ng_RT_rate_HW, hourly_data_P_ng_RT_rate_HW = json_file_read(directory_P_ng_RT_rate_HW, flatten_veh_data_parking)
result_P_ng_TOU_rate_H, hourly_data_P_ng_TOU_rate_H = json_file_read(directory_P_ng_TOU_rate_H, flatten_veh_data_parking)
# result_P_ng_TOU_rate_HW, hourly_data_P_ng_TOU_rate_HW = json_file_read(directory_P_ng_TOU_rate_HW, flatten_veh_data_parking)
result_P_50g_EV_rate_H, hourly_data_P_50g_EV_rate_H = json_file_read(directory_P_50g_EV_rate_H, flatten_veh_data_parking)
# result_P_50g_EV_rate_HW, hourly_data_P_50g_EV_rate_HW = json_file_read(directory_P_50g_EV_rate_HW, flatten_veh_data_parking)
result_P_50g_RT_rate_H, hourly_data_P_50g_RT_rate_H = json_file_read(directory_P_50g_RT_rate_H, flatten_veh_data_parking)
# result_P_50g_RT_rate_HW, hourly_data_P_50g_RT_rate_HW = json_file_read(directory_P_50g_RT_rate_HW, flatten_veh_data_parking)
result_P_50g_TOU_rate_H, hourly_data_P_50g_TOU_rate_H, = json_file_read(directory_P_50g_TOU_rate_H, flatten_veh_data_parking)
# result_P_50g_TOU_rate_HW, hourly_data_P_50g_TOU_rate_HW = json_file_read(directory_P_50g_TOU_rate_HW, flatten_veh_data_parking)
result_P_191g_EV_rate_H, hourly_data_P_191g_EV_rate_H = json_file_read(directory_P_191g_EV_rate_H, flatten_veh_data_parking)
# result_P_191g_EV_rate_HW, hourly_data_P_191g_EV_rate_HW = json_file_read(directory_P_191g_EV_rate_HW, flatten_veh_data_parking)
result_P_191g_RT_rate_H, hourly_data_P_191g_RT_rate_H = json_file_read(directory_P_191g_RT_rate_H, flatten_veh_data_parking)
# result_P_191g_RT_rate_HW, hourly_data_P_191g_RT_rate_HW = json_file_read(directory_P_191g_RT_rate_HW, flatten_veh_data_parking)
result_P_191g_TOU_rate_H, hourly_data_P_191g_TOU_rate_H = json_file_read(directory_P_191g_TOU_rate_H, flatten_veh_data_parking)
# result_P_191g_TOU_rate_HW, hourly_data_P_191g_TOU_rate_HW = json_file_read(directory_P_191g_TOU_rate_HW, flatten_veh_data_parking)
# %%

all_hourly_charging_N_data = pd.concat([
    add_tariff_name3(hourly_data_N_ng_EV_rate_H, 'EV Rate - Home', "Actual"),
    # add_tariff_name3(hourly_data_N_ng_EV_rate_HW, 'EV Rate - Home&Work', "Actual"),

    add_tariff_name3(hourly_data_N_ng_RT_rate_H, 'RT Rate - Home', "Actual"),
    # add_tariff_name3(hourly_data_N_ng_RT_rate_HW, 'RT Rate - Home&Work', "Actual"),

    add_tariff_name3(hourly_data_N_ng_TOU_rate_H, 'TOU Rate - Home', "Actual"),
    # add_tariff_name3(hourly_data_N_ng_TOU_rate_HW, 'TOU Rate - Home&Work', "Actual"),

    add_tariff_name2(hourly_data_N_50g_EV_rate_H, 'EV Rate - Home', "Actual"),
    # add_tariff_name2(hourly_data_N_50g_EV_rate_HW, 'EV Rate - Home&Work', "Actual"),

    add_tariff_name2(hourly_data_N_50g_RT_rate_H, 'RT Rate - Home', "Actual"),
    # add_tariff_name2(hourly_data_N_50g_RT_rate_HW, 'RT Rate - Home&Work', "Actual"),

    add_tariff_name2(hourly_data_N_50g_TOU_rate_H, 'TOU Rate - Home', "Actual"),
    # add_tariff_name2(hourly_data_N_50g_TOU_rate_HW, 'TOU Rate - Home&Work', "Actual"),

    add_tariff_name2(hourly_data_N_191g_EV_rate_H, 'EV Rate - Home', "Actual"),
    # add_tariff_name2(hourly_data_N_191g_EV_rate_HW, 'EV Rate - Home&Work', "Actual"),

    add_tariff_name2(hourly_data_N_191g_RT_rate_H, 'RT Rate - Home', "Actual"),
    # add_tariff_name2(hourly_data_N_191g_RT_rate_HW, 'RT Rate - Home&Work', "Actual"),

    add_tariff_name2(hourly_data_N_191g_TOU_rate_H, 'TOU Rate - Home', "Actual"),
    # add_tariff_name2(hourly_data_N_191g_TOU_rate_HW, 'TOU Rate - Home&Work', "Actual")
    ],ignore_index=True)

all_hourly_charging_P_data = pd.concat([
    add_tariff_name3(hourly_data_P_ng_EV_rate_H, 'EV Rate - Home', "Potential"),
    # add_tariff_name3(hourly_data_P_ng_EV_rate_HW, 'EV Rate - Home&Work', "Potential"),

    add_tariff_name3(hourly_data_P_ng_RT_rate_H, 'RT Rate - Home', "Potential"),
    # add_tariff_name3(hourly_data_P_ng_RT_rate_HW, 'RT Rate - Home&Work', "Potential"),

    add_tariff_name3(hourly_data_P_ng_TOU_rate_H, 'TOU Rate - Home', "Potential"),
    # add_tariff_name3(hourly_data_P_ng_TOU_rate_HW, 'TOU Rate - Home&Work', "Potential"),

    add_tariff_name2(hourly_data_P_50g_EV_rate_H, 'EV Rate - Home', "Potential"),
    # add_tariff_name2(hourly_data_P_50g_EV_rate_HW, 'EV Rate - Home&Work', "Potential"),

    add_tariff_name2(hourly_data_P_50g_RT_rate_H, 'RT Rate - Home', "Potential"),
    # add_tariff_name2(hourly_data_P_50g_RT_rate_HW, 'RT Rate - Home&Work', "Potential"),

    add_tariff_name2(hourly_data_P_50g_TOU_rate_H, 'TOU Rate - Home', "Potential"),
    # add_tariff_name2(hourly_data_P_50g_TOU_rate_HW, 'TOU Rate - Home&Work', "Potential"),

    add_tariff_name2(hourly_data_P_191g_EV_rate_H, 'EV Rate - Home', "Potential"),
    # add_tariff_name2(hourly_data_P_191g_EV_rate_HW, 'EV Rate - Home&Work', "Potential"),

    add_tariff_name2(hourly_data_P_191g_RT_rate_H, 'RT Rate - Home', "Potential"),
    # add_tariff_name2(hourly_data_P_191g_RT_rate_HW, 'RT Rate - Home&Work', "Potential"),

    add_tariff_name2(hourly_data_P_191g_TOU_rate_H, 'TOU Rate - Home', "Potential"),
    # add_tariff_name2(hourly_data_P_191g_TOU_rate_HW, 'TOU Rate - Home&Work', "Potential"),
    ], ignore_index=True)


# %%

# Concatenate all DataFrames into one
EV_rates_total = pd.concat([
    total_costs_N_ng_EV_rate_H,    total_costs_N_ng_EV_rate_HW,
    total_costs_P_ng_EV_rate_H,    total_costs_P_ng_EV_rate_HW,
    total_costs_A_EV_rate
], ignore_index=True)
EV_rates_total["Charging Speed"] = EV_rates_total["Charging Speed"].astype(float)


# Concatenate all DataFrames into one
TOU_rates_total = pd.concat([
    total_costs_N_ng_TOU_rate_H,    total_costs_N_ng_TOU_rate_HW,
    total_costs_P_ng_TOU_rate_H,    total_costs_P_ng_TOU_rate_HW,
    total_costs_A_TOU_rate,
], ignore_index=True)
TOU_rates_total["Charging Speed"] = TOU_rates_total["Charging Speed"].astype(float)


# Concatenate all DataFrames into one
RT_rates_total = pd.concat([
    total_costs_N_ng_RT_rate_H,    total_costs_N_ng_RT_rate_HW,
    total_costs_P_ng_RT_rate_H,    total_costs_P_ng_RT_rate_HW,
    total_costs_A_TOU_rate,    total_costs_A_EV_rate],
    ignore_index=True)
RT_rates_total["Charging Speed"] = RT_rates_total["Charging Speed"].astype(float)


# Concatenate all dataframes
All_rates_total = pd.concat([
    add_tariff_name(costs_N_50g_EV_rate_H, 'EV Rate'),  # add_tariff_name(costs_N_50g_EV_rate_HW, 'EV Rate'),
    add_tariff_name(costs_N_50g_RT_rate_H, 'RT Rate'),   # add_tariff_name(costs_N_50g_RT_rate_HW, 'RT Rate'),
    add_tariff_name(costs_N_50g_TOU_rate_H, 'TOU Rate'),  # add_tariff_name(costs_N_50g_TOU_rate_HW, 'TOU Rate'),
    add_tariff_name(costs_P_50g_EV_rate_H, 'EV Rate'),  # add_tariff_name(costs_P_50g_EV_rate_HW, 'EV Rate'),
    add_tariff_name(costs_P_50g_RT_rate_H, 'RT Rate'),   # add_tariff_name(costs_P_50g_RT_rate_HW, 'RT Rate'),
    add_tariff_name(costs_P_50g_TOU_rate_H, 'TOU Rate'),  # add_tariff_name(costs_P_50g_TOU_rate_HW, 'TOU Rate'),
    add_tariff_name(costs_N_191g_EV_rate_H, 'EV Rate'),  # add_tariff_name(costs_N_191g_EV_rate_HW, 'EV Rate'),
    add_tariff_name(costs_N_191g_RT_rate_H, 'RT Rate'),  # add_tariff_name(costs_N_191g_RT_rate_HW, 'RT Rate'),
    add_tariff_name(costs_N_191g_TOU_rate_H, 'TOU Rate'),  # add_tariff_name(costs_N_191g_TOU_rate_HW, 'TOU Rate'),
    add_tariff_name(costs_P_191g_EV_rate_H, 'EV Rate'),  # add_tariff_name(costs_P_191g_EV_rate_HW, 'EV Rate'),
    add_tariff_name(costs_P_191g_RT_rate_H, 'RT Rate'),  # add_tariff_name(costs_P_191g_RT_rate_HW, 'RT Rate'),
    add_tariff_name(costs_P_191g_TOU_rate_H, 'TOU Rate'),  # add_tariff_name(costs_P_191g_TOU_rate_HW, 'TOU Rate'),
    add_tariff_name(costs_A_TOU_rate, 'TOU Rate'),    add_tariff_name(costs_A_EV_rate, 'EV Rate'),
    add_tariff_name(costs_A_TOU_rate_191, 'TOU Rate'),    add_tariff_name(costs_A_EV_rate_191, 'EV Rate')], ignore_index=True)


All_rates_total_6 = All_rates_total.loc[(All_rates_total["Charging Speed"].isna()) | (All_rates_total["Charging Speed"] == 6.6)]
All_rates_total_12 = All_rates_total.loc[(All_rates_total["Charging Speed"].isna()) | (All_rates_total["Charging Speed"] == 12)]
All_rates_total_19 = All_rates_total.loc[(All_rates_total["Charging Speed"].isna()) | (All_rates_total["Charging Speed"] == 19)]
# %%

# EV Rate Pricing Plot
colors = {'Off-Peak': '#00a4de', 'Part-Peak': '#fdba0b', 'Peak': '#f2651c'}
plot_price_chart_EVRATE(off_peak_price=31, part_peak_price=51, peak_price=62, bar_width=1, colors=colors, font_size=14)
plot_price_chart_EVRATE(off_peak_price=31, part_peak_price=48, peak_price=49, bar_width=1, colors=colors, font_size=14)

# %%
# TOU Pricing Plot
colors = {'Off-Peak': '#00a4de', 'Peak': '#f2651c'}
plot_price_chart_TOU(off_peak_price=49, peak_price=59, bar_width=1, colors=colors, font_size=14)
plot_price_chart_TOU(off_peak_price=45, peak_price=48, bar_width=1, colors=colors, font_size=14)
# %%

# Example usage
plot_cost_comparison_EV(EV_rates_total, num_vehicles=50, title_size=14, axis_text_size=12, y_axis_title='Cost / Revenue ($)', barhight=-2500)
plot_cost_comparison_TOU(TOU_rates_total, num_vehicles=50, title_size=14, axis_text_size=12, y_axis_title='Cost ($)', barhight=500)
plot_cost_comparison_RT(RT_rates_total, num_vehicles=50, title_size=14, axis_text_size=12, y_axis_title='Cost / Revenue ($)', barhight=-7500)

# %%

voilin_input_data = violin_input(All_rates_total)
voilin_input_data_smart = filter_rows_by_word(voilin_input_data, 'Scenario', ['Actual', 'Smart'])
voilin_input_data_V2G = filter_rows_by_word(voilin_input_data, 'Scenario', ['Actual', 'V2G'])
# %%
# Example usage with df_long
stacked_violin_plot(voilin_input_data_smart)
stacked_violin_plot(voilin_input_data_V2G)
# %%

# Process both datasets
all_hourly_charging_N_data_grouped = process_charging_data(all_hourly_charging_N_data)
all_hourly_charging_P_data_grouped = process_charging_data(all_hourly_charging_P_data)
# %%

plotting_demand_heatmap(all_hourly_charging_N_data_grouped, all_hourly_charging_P_data_grouped, charging_type="smart", color_palette="inferno")
plotting_demand_heatmap(all_hourly_charging_N_data_grouped, all_hourly_charging_P_data_grouped, charging_type="v2g", color_palette="inferno")

# %%
plot_ghg_distribution_seasons(GHG_dict, Power_dict)
# %%

# test_dataframe = all_hourly_charging_N_data[(all_hourly_charging_N_data["Vehicle"] == "P_1087") & (all_hourly_charging_N_data["Tariff"] == "EV Rate - Home") & (all_hourly_charging_N_data["Charging Speed"] == 6.6) &
#                                             (all_hourly_charging_N_data["Charging Type"] == 'v2g') & (all_hourly_charging_N_data["GHG Cost"] == 0)].reset_index(drop=True)

test = process_charging_data1(all_hourly_charging_N_data)

# Concatenate DataFrames
result = pd.concat([test, costs_A_RT_rate_hourly], ignore_index=True)

def plot_filtered_data(df, ghg_value, chtype):
    # Fill NaN values in all columns (if any) to consider them in one group
    df.fillna('Actual', inplace=True)

    # Filter data based on GHG value
    filtered_df = df[((df['GHG Cost'] == ghg_value) & (df['Charging Type'] == chtype)) | (df['Charging Type'] == 'Actual')]
    filtered_df = filtered_df[(filtered_df['Tariff'].str.contains('Home') & ~filtered_df['Tariff'].str.contains('Home&Work')) | ((filtered_df['Charging Type'] == 'Actual'))]
    # Group the data based on the columns except 'daily_hour' and 'X_CHR'
    grouped = filtered_df.groupby(['Charging Type', 'Charging Speed', 'GHG Cost', 'Tariff', 'Charging_Behavior'])

    # Plot the data
    plt.figure(figsize=(12, 6))

    for name, group in grouped:
        plt.plot(group['daily_hour'], group['X_CHR'], label=str(name))

    plt.xlabel('Hour', fontsize=14)
    plt.ylabel('Power kW', fontsize=14)
    plt.title(f'Charging Demand Curve (GHG Cost: {ghg_value})')
    plt.legend(title='Groups', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.grid(True)
    plt.show()


# Example usage
plot_filtered_data(result, 0.05000, "smart")
# %% Bttery health

all_hourly_charging_N_data_battery = all_hourly_charging_N_data[["Vehicle", "Electricity_Cost", "Degradation_Cost", "GHG_Cost", "X_CHR", "Charging Type", "Charging Speed", "GHG Cost", 'Tariff', 'Charging_Behavior']]
all_hourly_charging_N_data_battery.to_pickle('all_hourly_charging_N_data_battery.pkl')

all_hourly_charging_P_data_battery = all_hourly_charging_P_data[["Vehicle", "Electricity_Cost", "Degradation_Cost","GHG_Cost", "X_CHR", "Charging Type", "Charging Speed", "GHG Cost", 'Tariff', 'Charging_Behavior']]
all_hourly_charging_P_data_battery.to_pickle('all_hourly_charging_P_data_battery.pkl')

# %%

CDF_N = all_hourly_charging_N_data[(all_hourly_charging_N_data["X_CHR"] != 0) & (all_hourly_charging_N_data["X_CHR"] <= 12) & (all_hourly_charging_N_data["X_CHR"] >= -12) & (all_hourly_charging_N_data["Charging Speed"] != 19)
                                   & (all_hourly_charging_N_data["GHG Cost"] == 0.191) & (all_hourly_charging_N_data["Tariff"] != "TOU Rate - Home")]
CDF_P = all_hourly_charging_P_data[(all_hourly_charging_P_data["X_CHR"] != 0) & (all_hourly_charging_P_data["X_CHR"] <= 12) & (all_hourly_charging_P_data["X_CHR"] >= -12) & (all_hourly_charging_P_data["Charging Speed"] != 19)
                                   & (all_hourly_charging_P_data["GHG Cost"] != 0.191) & (all_hourly_charging_P_data["Tariff"] != "TOU Rate - Home")]

CDF_N["Electricity_Paid"] = CDF_N["Electricity_Cost"] / abs(CDF_N["X_CHR"])
CDF_P["Electricity_Paid"] = CDF_P["Electricity_Cost"] / abs(CDF_P["X_CHR"])

CDF_N = pd.merge(CDF_N, df_GHG, on="Hour", how="left")
CDF_P = pd.merge(CDF_P, df_GHG, on="Hour", how="left")

CDF_N['GHG_value'] = CDF_N.apply(lambda row: row['GHG_value'] * -1 if row['X_CHR'] < -1 else row['GHG_value'], axis=1)
CDF_P['GHG_value'] = CDF_P.apply(lambda row: row['GHG_value'] * -1 if row['X_CHR'] < -1 else row['GHG_value'], axis=1)

costs_A_TOU_rate_hourly_in["Electricity_Paid"] = costs_A_TOU_rate_hourly_in["Electricity_Cost"] / abs(costs_A_TOU_rate_hourly_in["X_CHR"])
costs_A_TOU_rate_hourly_in = pd.merge(costs_A_TOU_rate_hourly_in, df_GHG, on="Hour", how="left")
# %%


def plot_cdf_by_group(df, df2, column_to_plot, xlabel, figsize=(10, 6)):
    def plot_cdf(data, label, **kwargs):
        """Helper function to plot the CDF of a data series."""
        sorted_data = np.sort(data)
        cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        plt.plot(sorted_data, cdf, label=label, linewidth=2, **kwargs)
    group_by_columns = ['Charging Type', 'Charging Speed', 'Tariff']
    # Group the DataFrame by specified columns
    grouped = df.groupby(group_by_columns)
    plt.figure(figsize=figsize)
    # Loop through each group and plot the CDF for the specified column
    for name, group in grouped:
        # Assuming 'name' is a tuple with the same order as 'group_by_columns'
        if len(name) == 3:  # Ensure there are exactly three components to unpack
            charging_type, charging_speed, tariff = name

            # Properly capitalize 'smart' to 'Smart' and 'v2g' to 'V2G'
            if charging_type.lower() == 'smart':
                charging_type = 'Smart'
            elif charging_type.lower() == 'v2g':
                charging_type = 'V2G'
            # Remove ' - Home' from the tariff name
            tariff = tariff.replace(' - Home', '')
            # Create the label
            label = f'{charging_type}, {charging_speed} kW, {tariff}'
        else:
            label = ', '.join([str(val) for val in name])  # Fallback if the structure is unexpected
        # Plot the CDF for this group
        plot_cdf(group[column_to_plot], label)
        # Plot the baseline DataFrame (df2)
    if df2 is not None and column_to_plot in df2.columns:
        plot_cdf(df2[column_to_plot], label='Baseline', linestyle='--', color='black')  # Different style for the baseline

    # Set the x and y labels with font size
    plt.xlabel(xlabel, fontsize=18)
    plt.ylabel("CDF", fontsize=18)
    # Adjust x-ticks and y-ticks font size
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    # Add legend with increased font size
    plt.legend(loc='best', fontsize=14)  # Increase the font size of the legend
    plt.grid(True)
    # Display the plot
    plt.show()


plot_cdf_by_group(CDF_N, costs_A_TOU_rate_hourly_in, "Electricity_Paid", 'Electricity Cost ($/kWh)', figsize=(10, 6))
plot_cdf_by_group(CDF_N, costs_A_TOU_rate_hourly_in, "GHG_value", 'Carbon Intensity (g/kWh)', figsize=(10, 6))

plot_cdf_by_group(CDF_P, costs_A_TOU_rate_hourly_in, "Electricity_Paid", 'Electricity Cost ($/kWh)', figsize=(10, 6))
plot_cdf_by_group(CDF_P, costs_A_TOU_rate_hourly_in, "GHG_value", 'Carbon Intensity (g/kWh)', figsize=(10, 6))