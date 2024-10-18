# %%
import pandas as pd
import warnings
import json
import os
import logging
from parking import (xlsx_read, json_file_read, plot_price_chart_EVRATE, plot_price_chart_TOU, plot_cost_comparison_EV, plot_cost_comparison_TOU, 
                     plot_cost_comparison_RT, stacked_violin_plot, add_tariff_name, plot_ghg_distribution_seasons, violin_input, filter_rows_by_word, add_tariff_name2,add_tariff_name3,
                     process_charging_data, plotting_demand_heatmap,plot_benefit_by_scenario, calculate_ghg_difference,plot_styled_box_by_scenarioP, calculate_cost_difference,
                     plot_styled_box_by_scenario, plot_cdf_by_group, plot_benefit_vs_degradation, calculate_charge_difference, plot_box_by_tariff, draw_box_plot)

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
# %%
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
# %%
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
    add_tariff_name(costs_N_50g_EV_rate_H, 'EV Rate'),
    add_tariff_name(costs_N_50g_RT_rate_H, 'RT Rate'),
    add_tariff_name(costs_N_50g_TOU_rate_H, 'TOU Rate'),
    add_tariff_name(costs_P_50g_EV_rate_H, 'EV Rate'),
    add_tariff_name(costs_P_50g_RT_rate_H, 'RT Rate'),
    add_tariff_name(costs_P_50g_TOU_rate_H, 'TOU Rate'),
    add_tariff_name(costs_N_191g_EV_rate_H, 'EV Rate'),
    add_tariff_name(costs_N_191g_RT_rate_H, 'RT Rate'),
    add_tariff_name(costs_N_191g_TOU_rate_H, 'TOU Rate'),
    add_tariff_name(costs_P_191g_EV_rate_H, 'EV Rate'),
    add_tariff_name(costs_P_191g_RT_rate_H, 'RT Rate'),
    add_tariff_name(costs_P_191g_TOU_rate_H, 'TOU Rate'),
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
stacked_violin_plot(voilin_input_data_smart)
stacked_violin_plot(voilin_input_data_V2G)
# %%
plot_ghg_distribution_seasons(GHG_dict, Power_dict)

# %% Bttery health

# all_hourly_charging_N_data_battery = all_hourly_charging_N_data[["Vehicle", "Electricity_Cost", "Degradation_Cost", "GHG_Cost", "X_CHR", "Charging Type", "Charging Speed", "GHG Cost", 'Tariff', 'Charging_Behavior']]
# all_hourly_charging_N_data_battery.to_pickle('all_hourly_charging_N_data_battery.pkl')
#
# all_hourly_charging_P_data_battery = all_hourly_charging_P_data[["Vehicle", "Electricity_Cost", "Degradation_Cost","GHG_Cost", "X_CHR", "Charging Type", "Charging Speed", "GHG Cost", 'Tariff', 'Charging_Behavior']]
# all_hourly_charging_P_data_battery.to_pickle('all_hourly_charging_P_data_battery.pkl')

# %%
CDF_N = all_hourly_charging_N_data[(all_hourly_charging_N_data["X_CHR"] != 0) &
                                   (all_hourly_charging_N_data["GHG Cost"] == 0.191) &
                                   (all_hourly_charging_N_data["Tariff"] != "TOU Rate - Home")]

CDF_N["Electricity_Paid"] = CDF_N["Electricity_Cost"] / abs(CDF_N["X_CHR"])
CDF_N = pd.merge(CDF_N, df_GHG, on="Hour", how="left")
CDF_N['GHG_value'] = CDF_N.apply(lambda row: row['GHG_value'] * -1 if row['X_CHR'] < 0 else row['GHG_value'], axis=1)
# %%
CDF_P = all_hourly_charging_P_data[(all_hourly_charging_P_data["X_CHR"] != 0) &
                                   (all_hourly_charging_P_data["GHG Cost"] != 0.191) &
                                   (all_hourly_charging_P_data["Tariff"] != "TOU Rate - Home")]
CDF_P["Electricity_Paid"] = CDF_P["Electricity_Cost"] / abs(CDF_P["X_CHR"])
CDF_P = pd.merge(CDF_P, df_GHG, on="Hour", how="left")
CDF_P['GHG_value'] = CDF_P.apply(lambda row: row['GHG_value'] * -1 if row['X_CHR'] < 0 else row['GHG_value'], axis=1)
# %%
costs_A_TOU_rate_hourly_in["Electricity_Paid"] = costs_A_TOU_rate_hourly_in["Electricity_Cost"] / abs(costs_A_TOU_rate_hourly_in["X_CHR"])
costs_A_TOU_rate_hourly_in = pd.merge(costs_A_TOU_rate_hourly_in, df_GHG, on="Hour", how="left")
# %%

charged_N = calculate_charge_difference(CDF_N)
charged_P = calculate_charge_difference(CDF_P)

charged = pd.concat([charged_N, charged_P], axis=0)


draw_box_plot(charged, text_size=14)
# %%
# Run the function using your data with charging speed consideration
# plot_styled_box_by_scenario(CDF_N,costs_A_TOU_rate_hourly_in, xaxis="Hourly Cost of Electric Vehicle Charging/Discharging ($)", x_column='Electricity_Cost', y_column='Scenarios',
#                      charge_type_column='Charging Type', tariff_column='Tariff', behavior_column='X_CHR', speed_column='Charging Speed', xlimit=30)
#
# plot_styled_box_by_scenario(CDF_N,costs_A_TOU_rate_hourly_in,  xaxis="Hourly Emission of Electric Vehicle Charging/Discharging (g CO2)",x_column='GHG_value', y_column='Scenarios',
#                      charge_type_column='Charging Type', tariff_column='Tariff', behavior_column='X_CHR', speed_column='Charging Speed', xlimit=7000)

CDF_N_grouped_g = calculate_ghg_difference(CDF_N, costs_A_TOU_rate_hourly_in)
CDF_N_grouped_c = calculate_cost_difference(CDF_N, costs_A_TOU_rate_hourly_in)
# %%
# Run the function using your data with charging speed consideration
# plot_styled_box_by_scenarioP(CDF_P,costs_A_TOU_rate_hourly_in, xaxis="Hourly Cost of Electric Vehicle Charging/Discharging ($)" , x_column='Electricity_Cost', y_column='Scenarios',
#                      charge_type_column='Charging Type', tariff_column='Tariff', behavior_column='X_CHR', speed_column='Charging Speed', xlimit=30)
#
# plot_styled_box_by_scenarioP(CDF_P,costs_A_TOU_rate_hourly_in,  xaxis="Hourly Emission of Electric Vehicle Charging/Discharging (g CO2)" ,x_column='GHG_value', y_column='Scenarios',
#                      charge_type_column='Charging Type', tariff_column='Tariff', behavior_column='X_CHR', speed_column='Charging Speed', xlimit=7000)

CDF_P_grouped_g = calculate_ghg_difference(CDF_P, costs_A_TOU_rate_hourly_in)
CDF_P_grouped_c = calculate_cost_difference(CDF_P, costs_A_TOU_rate_hourly_in)
#%%
# plot_cdf_by_group(CDF_N, costs_A_TOU_rate_hourly_in, "Electricity_Cost", "Hourly Cost of Electric Vehicle Charging/Discharging ($)", figsize=(10, 6))
# plot_cdf_by_group(CDF_N, costs_A_TOU_rate_hourly_in, "GHG_value", 'Carbon Intensity (g/kWh)', figsize=(10, 6))

#%%
# plot_cdf_by_group(CDF_P, costs_A_TOU_rate_hourly_in, "Electricity_Paid", 'Electricity Cost ($/kWh)', figsize=(10, 6))
# plot_cdf_by_group(CDF_P, costs_A_TOU_rate_hourly_in, "GHG_value", 'Carbon Intensity (g/kWh)', figsize=(10, 6))

# %%

plot_benefit_vs_degradation(EV_rates_total, num_vehicles=50, baseline_cost_scenario='last',title="EV-Rate",  lb=1500, ub=5200, title_size=12, axis_text_size=12)
plot_benefit_vs_degradation(TOU_rates_total, num_vehicles=50, baseline_cost_scenario='last',title="TOU-Rate", lb=900, ub=1800, title_size=12, axis_text_size=12)

plot_benefit_vs_degradation(RT_rates_total, num_vehicles=50, baseline_cost_scenario='second_last',title="RTvsTOU",lb=3000, ub=11500, title_size=12, axis_text_size=12)
plot_benefit_vs_degradation(RT_rates_total, num_vehicles=50, baseline_cost_scenario='last',title="RTvsEV", lb=3000, ub=11500, title_size=12, axis_text_size=12)

# %%
plot_benefit_by_scenario(voilin_input_data, scenario_filter='Actual', charging_speed=19, fz=18)
plot_benefit_by_scenario(voilin_input_data, scenario_filter='Potential', charging_speed=19, fz=18)

# %%

plot_box_by_tariff(CDF_N_grouped_g, CDF_P_grouped_g, figtitle="Annual CO$_2$ Emissions Reduction per\nVehicle from V1G and V2G Participation (tonne)",  fz=16, show_dollar=False)
plot_box_by_tariff(CDF_N_grouped_c, CDF_P_grouped_c, figtitle="Annual Financial Savings per Vehicle\nfrom V1G and V2G Participation ($)",  fz=16, show_dollar=True)
