# %%
import pandas as pd
import warnings
import json
import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.interpolate import griddata
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


# %%
# Load BEV travel data and aggregate distance by vehicle
def load_bev_distance(file_path):
    bev_data = pd.read_csv(file_path)
    bev_distance = bev_data.groupby(["vehicle_name", "bat_cap"])["distance"].sum().reset_index()
    return bev_distance


# Load and prepare cost data from Excel
def load_and_prepare_cost_data(file_path, sheet_name, charging_type, ghg_cost):
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    df = df.drop(df.columns[0], axis=1)  # Drop the unnamed column
    df["Tariff"] = charging_type
    df["GHG Cost"] = ghg_cost
    df["V2G_Location"] = "None"
    df["Plugged-in Sessions"] = "Actual"
    return df


# Adjust GHG cost for the dataset
def adjust_ghg_cost(df, new_ghg_cost, baseline_ghg_cost=0.05):
    df_copy = df.copy()
    df_copy["GHG Cost"] = new_ghg_cost
    df_copy["GHG_Cost"] = (df_copy["GHG_Cost"] / baseline_ghg_cost) * new_ghg_cost
    return df_copy


# Main processing of cost data
def process_actual_cost(ev_cost_file, tou_cost_file):
    # Load and prepare cost data
    costs_ev_rate = load_and_prepare_cost_data(ev_cost_file, 'Individual Costs', "EV Rate", 0.05)
    costs_tou_rate = load_and_prepare_cost_data(tou_cost_file, 'Individual Costs', "TOU Rate", 0.05)
    # Adjust GHG cost for the new value of 0.191
    costs_ev_rate_191 = adjust_ghg_cost(costs_ev_rate, new_ghg_cost=0.191)
    costs_tou_rate_191 = adjust_ghg_cost(costs_tou_rate, new_ghg_cost=0.191)
    # Combine all cost data into a single DataFrame
    combined_costs = pd.concat([costs_ev_rate, costs_tou_rate, costs_ev_rate_191, costs_tou_rate_191]).reset_index(drop=True)
    # Calculate total costs
    combined_costs["Total_cost"] = combined_costs["Electricity_Cost"] + combined_costs["Degradation_Cost"] + combined_costs["GHG_Cost"]
    return combined_costs


def load_and_clean_data(file_path, ghg_dict):
    # Load data from pickle file
    data = pd.read_pickle(file_path)
    data['hour_index'] = data.groupby(['Vehicle', 'Charging Type', 'Charging Speed', 'GHG Cost', 'Tariff', 'Charging_Behavior']).cumcount()
    # Create the 'GHG_Produced' column by multiplying the X_HR column by the corresponding value in the GHG dictionary
    data = data[data['X_CHR'] != 0]
    data['GHG_Produced'] = data.apply(lambda row: (row['X_CHR'] * ghg_dict.get(row['hour_index']) / 1000), axis=1)
    # Group by the specified columns and aggregate the sum of 'GHG_Produced' and 'X_CHR'
    grouped_df = data.groupby(['Charging Type', 'Charging Speed', 'GHG Cost', 'Tariff', 'Charging_Behavior']).agg({
        'GHG_Produced': 'sum',
    }).reset_index()

    return grouped_df


def process_actual_charging(file_path, ghg_dict):
    # Step 1: Load the data from the pickle file
    df = pd.read_pickle(file_path)

    # Step 2: Remove all rows where 'soc_diff' is zero
    df = df[df['soc_diff'] != 0]

    # Step 3: Ensure 'hour' column is of numeric type
    df['hour'] = pd.to_numeric(df['hour'], errors='coerce')

    # Step 4: Calculate the energy difference in kWh (soc_diff / 100 * bat_cap)
    df['energy_kWh'] = (df['soc_diff'] / 100) * df['bat_cap']

    # Step 5: Multiply by the GHG factor based on the 'hour' column
    df['GHG_Produced'] = df.apply(lambda row: row['energy_kWh'] * ghg_dict.get(row['hour'], 0), axis=1)
    df = df['GHG_Produced'].sum()/1000

    return df


# %%  File paths for data

batt_file = "/Users/haniftayarani/V2G_Project/Travel_data/Battery_Price_Per_kWh_Estimations.csv"
bev_file = "/Users/haniftayarani/V2G_Project/data.csv"
ev_cost_file = '/Users/haniftayarani/V2G_Project/Results/Actual/Actual_EV_rate_cost.xlsx'
tou_cost_file = '/Users/haniftayarani/V2G_Project/Results/Actual/Actual_TOU_cost.xlsx'

ev_cost_file = '/Users/haniftayarani/V2G_Project/Results/Actual/Actual_EV_rate_cost.xlsx'
tou_cost_file = '/Users/haniftayarani/V2G_Project/Results/Actual/Actual_TOU_cost.xlsx'
demand_curve = '/Users/haniftayarani/V2G_Project/demand_curve.pkl'

GHG_data = pd.read_csv("CISO.csv")
GHG_dict = dict(enumerate(GHG_data.iloc[:, 0]))

# Load and process BEV distance data
bev_distance = load_bev_distance(bev_file)
batt_price = pd.read_csv(batt_file)
# Process actual costs
actual_cost = process_actual_cost(ev_cost_file, tou_cost_file)

# Load and clean the N and P datasets
all_hourly_charging_N_data = load_and_clean_data('all_hourly_charging_N_data_battery.pkl', GHG_dict)
all_hourly_charging_P_data = load_and_clean_data('all_hourly_charging_P_data_battery.pkl', GHG_dict)

ghg_data = pd.concat([all_hourly_charging_N_data, all_hourly_charging_P_data], axis=0)
ghg_data = ghg_data[ghg_data["Charging Speed"] != 19]
ghg_data_group = (ghg_data.groupby(["Charging Type", "Charging Speed", "GHG Cost", "Tariff",  "Charging_Behavior"])["GHG_Produced"].sum()).reset_index(drop=False) # Convert the MWh to KWh
ghg_data_group = ghg_data_group[~ghg_data_group["Tariff"].str.contains("Home&Work")]

Actual_ghg = process_actual_charging(demand_curve, GHG_dict)
ghg_data_group["Actual_GHG_Produced"] = Actual_ghg
ghg_data_group["GHG_improvement"] = ((ghg_data_group["Actual_GHG_Produced"] - ghg_data_group["GHG_Produced"])/ghg_data_group["Actual_GHG_Produced"]) * 100
