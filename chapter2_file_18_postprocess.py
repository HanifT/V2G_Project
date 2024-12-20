# %%
import pandas as pd
import warnings
import json
import os
import re
import logging
from parking import (plot_price_chart_EVRATE, plot_price_chart_TOU, read_all_results, read_combined_costs,
                     plot_ghg_distribution_seasons, plot_benefit_vs_degradation, plot_box_by_tariff,
                     process_and_plot_utility_data, all_rates)

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
CDF_N_grouped_g_pge, CDF_N_grouped_c_pge, CDF_P_grouped_g_pge, CDF_P_grouped_c_pge = process_and_plot_utility_data("PGE", actual_hourly, GHG_data, text_size=14)
CDF_N_grouped_g_sce, CDF_N_grouped_c_sce, CDF_P_grouped_g_sce, CDF_P_grouped_c_sce = process_and_plot_utility_data("SCE", actual_hourly, GHG_data, text_size=14)
CDF_N_grouped_g_sdge, CDF_N_grouped_c_sdge, CDF_P_grouped_g_sdge, CDF_P_grouped_c_sdge = process_and_plot_utility_data("SDGE", actual_hourly, GHG_data, text_size=14)
CDF_N_grouped_g_smud, CDF_N_grouped_c_smud, CDF_P_grouped_g_smud, CDF_P_grouped_c_smud = process_and_plot_utility_data("SMUD", actual_hourly, GHG_data, text_size=14)
# %%
plot_benefit_vs_degradation(TOU_rates_total, num_vehicles=50, Utility="PGE", title='TOU_PGE', lb=800, ub=1200, title_size=12, axis_text_size=12)
plot_benefit_vs_degradation(EV_rates_total, num_vehicles=50, Utility="PGE", title='EV_PGE', lb=1500, ub=5800, title_size=12, axis_text_size=12)
plot_benefit_vs_degradation(RT_rates_total_TOU, num_vehicles=50, Utility="PGE", title='RT_PGE_TOU', lb=3000, ub=10500, title_size=12, axis_text_size=12)
plot_benefit_vs_degradation(RT_rates_total_EV, num_vehicles=50, Utility="PGE", title='RT_PGE_EV', lb=3000, ub=10500, title_size=12, axis_text_size=12)


plot_benefit_vs_degradation(TOU_rates_total, num_vehicles=50, Utility="SCE", title='TOU_SCE', lb=1500, ub=4000, title_size=12, axis_text_size=12)
plot_benefit_vs_degradation(EV_rates_total, num_vehicles=50, Utility="SCE", title='EV_SCE', lb=2000, ub=7500, title_size=12, axis_text_size=12)
plot_benefit_vs_degradation(RT_rates_total_TOU, num_vehicles=50, Utility="SCE", title='RT_SCE_TOU', lb=3000, ub=12000, title_size=12, axis_text_size=12)
plot_benefit_vs_degradation(RT_rates_total_EV, num_vehicles=50, Utility="SCE", title='RT_SCE_EV', lb=3000, ub=12000, title_size=12, axis_text_size=12)

plot_benefit_vs_degradation(TOU_rates_total, num_vehicles=50, Utility="SDGE", title='TOU_SDGE', lb=1000, ub=2000, title_size=12, axis_text_size=12)
plot_benefit_vs_degradation(EV_rates_total, num_vehicles=50, Utility="SDGE", title='EV_SDGE', lb=2000, ub=7500, title_size=12, axis_text_size=12)
plot_benefit_vs_degradation(RT_rates_total_TOU, num_vehicles=50, Utility="SDGE", title='RT_SDGE_TOU', lb=3000, ub=13500, title_size=12, axis_text_size=12)
plot_benefit_vs_degradation(RT_rates_total_EV, num_vehicles=50, Utility="SDGE", title='RT_SDGE_EV', lb=3000, ub=13000, title_size=12, axis_text_size=12)

plot_benefit_vs_degradation(TOU_rates_total, num_vehicles=50, Utility="SMUD", title='TOU_SMUD', lb=1000, ub=2000, title_size=12, axis_text_size=12)
plot_benefit_vs_degradation(EV_rates_total, num_vehicles=50, Utility="SMUD", title='EV_SMUD', lb=1000, ub=2500, title_size=12, axis_text_size=12)
plot_benefit_vs_degradation(RT_rates_total_TOU, num_vehicles=50, Utility="SMUD", title='RT_SMUD_TOU', lb=1600, ub=4000, title_size=12, axis_text_size=12)
plot_benefit_vs_degradation(RT_rates_total_EV, num_vehicles=50, Utility="SMUD", title='RT_SMUD_EV', lb=1600, ub=4000, title_size=12, axis_text_size=12)
# %%

plot_box_by_tariff(CDF_N_grouped_g_pge, CDF_P_grouped_g_pge, figtitle="Annual CO$_2$ Emissions Reduction per\nVehicle from V1G and V2G Participation (tonne)",  fz=16, show_dollar=False)

# %%
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches


def plot_benefit_vs_degradation(df, num_vehicles, Utility="PGE", title='title', lb=0, ub=1000, title_size=18, axis_text_size=18, ax=None, last_in_row=False, unified_x_title=False):

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))  # Create a new plot if no axis is provided

    df1 = df.copy()
    df1 = df1[df1["Utility"] == Utility]
    # Convert Charging Speed to string and remove any trailing '.0'
    df1['Charging_Speed'] = df1['Charging_Speed'].astype(str).str.rstrip('.0')
    df1['Scenarios'] = df1['Charge_Type'] + ' - ' + df1['Charging_Speed'] + ' - ' + df1['Scenario'] + ' - ' + df1['V2G_Location']
    baseline_row = df1.iloc[-1]

    # Calculate baseline cost
    baseline_cost = baseline_row['Electricity_Cost'] + baseline_row['Degradation_Cost']

    # Calculate the net benefit (Total Cost Savings) against the baseline and degradation separately, adjusted per vehicle
    df1['Total_Benefit'] = (baseline_cost - (df1['Electricity_Cost'] + df1['Degradation_Cost'])) / num_vehicles
    df1['Degradation_Cost'] = df1['Degradation_Cost'] / num_vehicles

    # Define the speeds and locations for plotting
    speeds = ['6.6', '12', '19']  # Speeds to consider
    locations = ['Home', 'Home_Work']  # Locations for V2G scenarios
    location_labels = {'Home': '', 'Home_Work': ''}

    bar_height = 0.5  # Height for each grouped bar
    smart_colors = ['#fdad1a', '#fdad1a', '#fdad1a']  # Color for Smart Charging bars
    v2g_colors = {
        'Home': '#219f71',  # Color for V2G Home
        'Home_Work': '#004a6d',  # Color for V2G Home_Work
        'Degradation': '#c4383a'  # Red for degradation
    }

    y_positions = []  # Store y positions for each group
    scenario_labels = []  # Store labels for y positions
    group_positions = []  # To store the center of each group for annotation
    current_y = 2  # Current y position

    # Smart Charging Section: Add each speed separately with labels
    smart_bar_offsets = np.linspace(-bar_height / 20, bar_height / 20, len(speeds)) / 2

    for idx, speed in enumerate(speeds):
        smart_data = df1[
            (df1['Charge_Type'] == 'Smart') &
            (df1['Charging_Speed'] == speed) &
            (df1['V2G_Location'] == 'Home_Work')
        ]
        if not smart_data.empty:
            degradation_cost = -smart_data['Degradation_Cost'].values[0]
            total_benefit = smart_data['Total_Benefit'].values[0]

            # Plot Degradation and Benefit bars directly next to each other for each speed
            ax.barh(current_y + smart_bar_offsets[idx] - bar_height / 4 + 0.25, degradation_cost, height=bar_height, color=v2g_colors['Degradation'])
            ax.barh(current_y + smart_bar_offsets[idx] + bar_height / 4, total_benefit, height=bar_height, color=smart_colors[idx])

            # Add text labels for Degradation and Benefit
            ax.text(degradation_cost - 0.25, current_y + smart_bar_offsets[idx], f"${degradation_cost:.0f}", ha='right', va='center', fontsize=axis_text_size)
            ax.text(total_benefit + 0.25, current_y + smart_bar_offsets[idx], f"${total_benefit:.0f}", ha='left', va='center', fontsize=axis_text_size)

            # Add individual labels for each speed
            y_positions.append(current_y + smart_bar_offsets[idx])
            scenario_labels.append(f"{speed} kW")  # Only show speed for Smart Charging
            current_y += 0.6
    # Calculate center of Smart Charging section for annotation
    group_positions.append(np.mean([2.5 + offset for offset in smart_bar_offsets]))
    first_line_position = current_y - 0.15
    second_line_position = current_y + 4.25
    ax.axhline(first_line_position, color='black', linestyle='--', linewidth=1)
    ax.axhline(second_line_position, color='black', linestyle='--', linewidth=1)
    ax.set_xlim(-lb, ub)
    # Helper function to plot side-by-side bars for V2G with locations
    def plot_v2g_section(data, section_label):
        nonlocal current_y
        centers = []

        for speed in speeds:
            bar_offsets = np.linspace(-bar_height * 1.15, bar_height * 1.15, len(locations)) / 2
            for i, loc in enumerate(locations):
                loc_data = data[(data['Charging_Speed'] == speed) & (data['V2G_Location'] == loc)]
                if not loc_data.empty:
                    degradation_cost = -loc_data['Degradation_Cost'].values[0]
                    total_benefit = loc_data['Total_Benefit'].values[0]

                    # Plot bars for degradation and benefit
                    ax.barh(current_y + bar_offsets[i] + 0.5, degradation_cost, height=bar_height, color=v2g_colors['Degradation'])
                    ax.barh(current_y + bar_offsets[i] + bar_height, total_benefit, height=bar_height, color=v2g_colors[loc])

                    # Add text labels
                    ax.text(degradation_cost - 0.2, current_y + bar_offsets[i] + 0.5, f"${degradation_cost:.0f}", ha='right', va='center', fontsize=axis_text_size)
                    ax.text(total_benefit + 0.2, current_y + bar_offsets[i] + 0.5, f"${total_benefit:.0f}", ha='left', va='center', fontsize=axis_text_size)

                    # Add labels
                    y_positions.append(current_y + bar_offsets[i] + 0.5)
                    scenario_labels.append(f"{speed} kW {location_labels[loc]}")
                    centers.append(current_y + bar_offsets[i])

            current_y += 1.5  # Update the y position
        return np.mean(centers)  # Return the center position

    # Plot V2G Actual section by all speeds and store its center
    v2g_actual_data = df1[(df1['Charge_Type'] == 'Bidirectional') & (df1['Scenario'] == 'No_change')]
    if not v2g_actual_data.empty:
        center = plot_v2g_section(v2g_actual_data, "V2G / No Change in Plugging Behavior")
        group_positions.append(center)

    # Plot V2G Potential section by all speeds and store its center
    v2g_potential_data = df1[(df1['Charge_Type'] == 'Bidirectional') & (df1['Scenario'] == 'With_change')]
    if not v2g_potential_data.empty:
        center = plot_v2g_section(v2g_potential_data, "V2G / Plugging-in When Parked")
        group_positions.append(center)

    # Customize y-axis labels with all speeds retained
    ax.set_yticks(y_positions)
    ax.set_yticklabels(scenario_labels, fontsize=axis_text_size - 1)
    ax.axvline(0, color='grey', linewidth=1)
    ax.set_xlim(-lb, ub)
    ax.grid(True, axis='x', linestyle='--', alpha=0.7)

    # Add panel-specific title
    ax.set_title(title, fontsize=title_size)

    # Customize x-axis ticks and labels
    ax.set_xticks([-lb, 0, ub])
    ax.set_xticklabels([f'Loss', f'Baseline Cost\n(${round((baseline_cost / num_vehicles), 2)})', 'Savings'], fontsize=axis_text_size - 1)

    # Add x-axis title only if not unified
    if not unified_x_title:
        ax.set_xlabel("", fontsize=title_size)

    # Add secondary y-axis for the last graph in the row
    if last_in_row:
        x_annotation_position = ub * 1.20
        group_labels = ['V1G', 'V2G\nNo Change in\nPlugging Behavior', 'V2G\nPlugging-in\nWhen Parked']
        group_offsets = [0, 0.5, 1.0]  # Offset for each label, in y-axis units

        for pos, label, offset in zip(group_positions, group_labels, group_offsets):
            ax.text(
                x_annotation_position,
                pos + offset,  # Adjust the y-position with the offset
                label,
                ha='center',
                va='center',
                fontsize=title_size -2,
                weight='bold',
                rotation=90
            )
        # Add legend for the last graph in the row
        smart_patches = [mpatches.Patch(color=smart_colors[i], label=f"Smart Charging") for i in range(len(speeds))]
        v2g_patches = [
            mpatches.Patch(color=v2g_colors['Home'], label="Bidirectional Charger at Home"),
            mpatches.Patch(color=v2g_colors['Home_Work'], label="Bidirectional Charger at Home + Work")
        ]
        degradation_patch = mpatches.Patch(color=v2g_colors['Degradation'], label="Battery Degradation")

        # Place the legend outside the plot at the bottom in one line
        fig = ax.get_figure()
        fig.legend(
            handles=[smart_patches[0]] + v2g_patches + [degradation_patch],
            loc='lower center',
            fontsize=axis_text_size - 1,
            ncol=len(smart_patches) + len(v2g_patches) + 1,  # Number of columns in the legend
            bbox_to_anchor=(0.5, -0.01)  # Center the legend below the plot
        )

    return ax


def plot_benefit_vs_degradation_panel(data_list, num_vehicles, utilities, titles, lbs, ubs, y_title, x_title, figsize=(20, 8), title_size=18, axis_text_size=14):
    num_plots = len(data_list)  # Number of panels
    fig, axes = plt.subplots(1, num_plots, figsize=figsize, sharey=True)  # Create a single row of plots

    for i, ax in enumerate(axes):
        # Call the existing plot function, passing the specific axis
        plot_benefit_vs_degradation(
            data_list[i],
            num_vehicles=num_vehicles,
            Utility=utilities[i],
            title=titles[i],
            lb=lbs[i],
            ub=ubs[i],
            title_size=title_size-1,
            axis_text_size=axis_text_size,
            ax=ax,  # Pass the specific axis
            last_in_row=(i == len(axes) - 1),  # Add legend and secondary y-axis only for the last plot
            unified_x_title=(i != len(axes) - 1)  # Unified x-axis title only for the last plot
        )

    # Add shared y-axis and x-axis labels
    fig.text(0.01, 0.5, y_title, va='center', rotation='vertical', fontsize=title_size+1)  # Y-axis
    fig.text(0.5, 0.04, x_title, ha='center', fontsize=title_size)  # X-axis

    # Adjust layout
    plt.tight_layout(rect=[0.04, 0.09, 1, 1])
    plt.show()

# %%
plot_benefit_vs_degradation_panel(
    data_list=[TOU_rates_total, EV_rates_total, RT_rates_total_TOU, RT_rates_total_EV],
    num_vehicles=50,
    utilities=["PGE", "PGE", "PGE", "PGE"],
    titles=['TOU', 'EV Rate', 'RT Rate compare\nwith TOU', 'RT Rate compare\nwith EV Rate'],
    lbs=[4000, 4000, 4000, 4000],
    ubs=[11000, 11000, 11000, 11000],
    y_title="Charging Scenarios by Charger Speed and\nDeployment Location (PGE Territory)",
    # x_title="Net Benefit and Associated Degradation Cost per Vehicle ($)",
    x_title="",
    figsize=(42, 12),
    title_size=38,
    axis_text_size=35
)

plot_benefit_vs_degradation_panel(
    data_list=[TOU_rates_total, EV_rates_total, RT_rates_total_TOU, RT_rates_total_EV],
    num_vehicles=50,
    utilities=["SCE", "SCE", "SCE", "SCE"],
    titles=['TOU', 'EV Rate', 'RT Rate compare\nwith TOU', 'RT Rate compare\nwith EV Rate'],
    lbs=[4000, 4000, 4000, 4000],
    ubs=[12000, 12000, 12000, 12000],
    y_title="Charging Scenarios by Charger Speed and\nDeployment Location (SCE Territory)",
    # x_title="Net Benefit and Associated Degradation Cost per Vehicle ($)",
    x_title="",
    figsize=(42, 12),
    title_size=38,
    axis_text_size=35
)

plot_benefit_vs_degradation_panel(
    data_list=[TOU_rates_total, EV_rates_total, RT_rates_total_TOU, RT_rates_total_EV],
    num_vehicles=50,
    utilities=["SDGE", "SDGE", "SDGE", "SDGE"],
    titles=['TOU', 'EV Rate', 'RT Rate compare\nwith TOU', 'RT Rate compare\nwith EV Rate'],
    lbs=[4000, 4000, 4000, 4000],
    ubs=[14000, 14000, 14000, 14000],
    y_title="Charging Scenarios by Charger Speed and\nDeployment Location (SDGE Territory)",
    # x_title="Net Benefit and Associated Degradation Cost per Vehicle ($)",
    x_title="",
    figsize=(42, 12),
    title_size=38,
    axis_text_size=35
)

plot_benefit_vs_degradation_panel(
    data_list=[TOU_rates_total, EV_rates_total, RT_rates_total_TOU, RT_rates_total_EV],
    num_vehicles=50,
    utilities=["SMUD", "SMUD", "SMUD", "SMUD"],
    titles=['TOU', 'EV Rate', 'RT Rate compare\nwith TOU', 'RT Rate compare\nwith EV Rate'],
    lbs=[4000, 4000, 4000, 4000],
    ubs=[12000, 12000, 12000, 12000],
    y_title="Charging Scenarios by Charger Speed and\nDeployment Location (SMUD Territory)",
    # x_title="Net Benefit and Associated Degradation Cost per Vehicle ($)",
    x_title="",
    figsize=(42, 12),
    title_size=38,
    axis_text_size=35
)