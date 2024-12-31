# %%
import pandas as pd
import warnings
import json

from parking import (plot_price_chart_EVRATE, plot_price_chart_TOU, read_all_results, read_combined_costs,
                     plot_ghg_distribution_seasons, plot_2x2_utility_panels_with_adjustments, plot_box_by_tariff,
                     process_and_plot_utility_data, all_rates, plot_benefit_vs_degradation_panel_combined_fixed, plot_box_by_tariff_panel)

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


# Define inputs
base_dir = "/Users/haniftayarani/V2G_Project/Results/Result"
rate_types = ["EV_Rate", "TOU", "RT_Rate"]
utilities = ["pge", "sce", "sdge", "smud"]
locations = ["Home", "Home_Work"]  # Adjusted location format
# Run the function
total_optimal_combined, optimal_individual_aggregated = read_all_results(base_dir, rate_types, utilities, locations)
# Define inputs
base_dir1 = "/Users/haniftayarani/V2G_Project/Results/Actual"
# Run the function
actual_hourly, actual_individual_aggregated, total_actual_combined = read_combined_costs(base_dir1)
# %% reading the hourly files from json file

# base_dir = "/Users/haniftayarani/V2G_Project/Results/Result"
# process_and_save_all(base_dir, flatten_veh_data, flatten_veh_data_parking)
# %%

combined_hourly_data_pge_normal = pd.read_pickle("/Users/haniftayarani/V2G_Project/Hourly_data/combined_hourly_data_normal_PGE.pkl")
combined_hourly_data_pge_parking = pd.read_pickle("/Users/haniftayarani/V2G_Project/Hourly_data/combined_hourly_data_parking_PGE.pkl")
combined_hourly_data_sce_normal = pd.read_pickle("/Users/haniftayarani/V2G_Project/Hourly_data/combined_hourly_data_normal_SCE.pkl")
combined_hourly_data_sce_parking = pd.read_pickle("/Users/haniftayarani/V2G_Project/Hourly_data/combined_hourly_data_parking_SCE.pkl")
combined_hourly_data_sdge_normal = pd.read_pickle("/Users/haniftayarani/V2G_Project/Hourly_data/combined_hourly_data_normal_SDGE.pkl")
combined_hourly_data_sdge_parking = pd.read_pickle("/Users/haniftayarani/V2G_Project/Hourly_data/combined_hourly_data_parking_SDGE.pkl")
combined_hourly_data_smud_normal = pd.read_pickle("/Users/haniftayarani/V2G_Project/Hourly_data/combined_hourly_data_normal_SMUD.pkl")
combined_hourly_data_smud_parking = pd.read_pickle("/Users/haniftayarani/V2G_Project/Hourly_data/combined_hourly_data_parking_SMUD.pkl")


# %%
All_rates_total, TOU_rates_total, EV_rates_total, RT_rates_total, RT_rates_total_TOU, RT_rates_total_EV, All_rates_total_6, All_rates_total_12, All_rates_total_19 = all_rates(total_optimal_combined, total_actual_combined)
# %%
# EV Rate Pricing Plot
colors = {'Off-Peak': '#00a4de', 'Part-Peak': '#fdba0b', 'Peak': '#f2651c'}
plot_price_chart_EVRATE(off_peak_price=31, part_peak_price=51, peak_price=62, bar_width=1, colors=colors, font_size=14)
plot_price_chart_EVRATE(off_peak_price=31, part_peak_price=48, peak_price=49, bar_width=1, colors=colors, font_size=14)
# TOU Pricing Plot
colors = {'Off-Peak': '#00a4de', 'Peak': '#f2651c'}
plot_price_chart_TOU(off_peak_price=49, peak_price=59, bar_width=1, colors=colors, font_size=14)
plot_price_chart_TOU(off_peak_price=45, peak_price=48, bar_width=1, colors=colors, font_size=14)
# %%
plot_ghg_distribution_seasons(GHG_dict, Power_dict)
# %%
CDF_N_grouped_g_pge, CDF_P_grouped_g_pge, VM_PGE = process_and_plot_utility_data("PGE", actual_hourly, GHG_data)
CDF_N_grouped_g_sce, CDF_P_grouped_g_sce, VM_SCE = process_and_plot_utility_data("SCE", actual_hourly, GHG_data)
CDF_N_grouped_g_sdge, CDF_P_grouped_g_sdge, VM_SDGE = process_and_plot_utility_data("SDGE", actual_hourly, GHG_data)
CDF_N_grouped_g_smud, CDF_P_grouped_g_smud, VM_SMUD = process_and_plot_utility_data("SMUD", actual_hourly, GHG_data)

# %%

plot_box_by_tariff(CDF_N_grouped_g_pge, CDF_P_grouped_g_pge, figtitle="Annual CO$_2$ Emissions Reduction per\nVehicle from V1G and V2G Participation (tonne)", Rate_type="RT_Rate", fz=16, show_dollar=False)
plot_box_by_tariff(CDF_N_grouped_g_sce, CDF_P_grouped_g_sce, figtitle="Annual CO$_2$ Emissions Reduction per\nVehicle from V1G and V2G Participation (tonne)", Rate_type="RT_Rate", fz=16, show_dollar=False)
plot_box_by_tariff(CDF_N_grouped_g_sdge, CDF_P_grouped_g_sdge, figtitle="Annual CO$_2$ Emissions Reduction per\nVehicle from V1G and V2G Participation (tonne)", Rate_type="RT_Rate", fz=16, show_dollar=False)
plot_box_by_tariff(CDF_N_grouped_g_smud, CDF_P_grouped_g_smud, figtitle="Annual CO$_2$ Emissions Reduction per\nVehicle from V1G and V2G Participation (tonne)", Rate_type="RT_Rate", fz=16, show_dollar=False)

# %%

plot_benefit_vs_degradation_panel_combined_fixed(
    data_list_sets=[
        [TOU_rates_total, EV_rates_total, RT_rates_total],  # PGE
        [TOU_rates_total, EV_rates_total, RT_rates_total],  # SCE
        [TOU_rates_total, EV_rates_total, RT_rates_total],  # SDGE
        [TOU_rates_total, EV_rates_total, RT_rates_total],  # SMUD
    ],
    num_vehicles_list=[50, 50, 50, 50],  # Number of vehicles for each utility
    utilities_list=[
        ["PGE", "PGE", "PGE"],  # Utilities for PGE
        ["SCE", "SCE", "SCE"],  # Utilities for SCE
        ["SDGE", "SDGE", "SDGE"],  # Utilities for SDGE
        ["SMUD", "SMUD", "SMUD"],  # Utilities for SMUD
    ],
    titles_list=[
        ["TOU Rate", "EV Rate", "RT Rate"],  # Titles for PGE
        ["", "", ""],  # Titles for SCE
        ["", "", ""],  # Titles for SDGE
        ["", "", ""],  # Titles for SMUD
    ],
    lbs_list=[
        [2500, 2500, 2500],  # Lower bounds for PGE
        [2500, 2500, 2500],  # Lower bounds for SCE
        [2500, 2500, 2500],  # Lower bounds for SDGE
        [2500, 2500, 2500],  # Lower bounds for SMUD
    ],
    ubs_list=[
        [10500, 10500, 10500],  # Upper bounds for PGE
        [10500, 10500, 10500],  # Upper bounds for SCE
        [10500, 10500, 10500],  # Upper bounds for SDGE
        [10500, 10500, 10500],  # Upper bounds for SMUD
    ],
    y_titles=[
        "Charging Scenarios by Charger Speed and\nDeployment Location (PGE Territory)",
        "Charging Scenarios by Charger Speed and\nDeployment Location (SCE Territory)",
        "Charging Scenarios by Charger Speed and\nDeployment Location (SDGE Territory)",
        "Charging Scenarios by Charger Speed and\nDeployment Location (SMUD Territory)",
    ],
    x_title=" Average Net Benefit and Associated Degradation Cost per Vehicle ($)",  # Shared x-axis title
    figsize=(22, 28),  # Adjust this if you want larger or smaller plots
    title_size=28, axis_text_size=25, row_spacing=0.04, col_spacing=0.04

)

plot_2x2_utility_panels_with_adjustments(
    grouped_data=[VM_PGE, VM_SCE, VM_SDGE, VM_SMUD],
    utilities=["PGE", "SCE", "SDGE", "SMUD"],
    figsize=(14, 12), text_size=14)

# Example datasets
data_list = [
    (CDF_N_grouped_g_pge, CDF_P_grouped_g_pge),
    (CDF_N_grouped_g_sce, CDF_P_grouped_g_sce),
    (CDF_N_grouped_g_sdge, CDF_P_grouped_g_sdge),
    (CDF_N_grouped_g_smud, CDF_P_grouped_g_smud),
]
utility_names = ["PGE", "SCE", "SDGE", "SMUD"]

# Call the function
plot_box_by_tariff_panel(
    data_list=data_list,
    utility_names=utility_names,
    figtitle="Annual CO$_2$ Emissions Reduction per Vehicle from V1G and V2G Participation",
    Rate_type="RT_Rate",
    fz=16,
    figsize=(14, 10)
)

plot_box_by_tariff_panel(
    data_list=data_list,
    utility_names=utility_names,
    figtitle="Annual CO$_2$ Emissions Reduction per Vehicle from V1G and V2G Participation",
    Rate_type="EV_Rate",
    fz=16,
    figsize=(14, 10)
)

plot_box_by_tariff_panel(
    data_list=data_list,
    utility_names=utility_names,
    figtitle="Annual CO$_2$ Emissions Reduction per Vehicle from V1G and V2G Participation",
    Rate_type="TOU",
    fz=16,
    figsize=(14, 10)
)