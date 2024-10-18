# %%
import pandas as pd
from parking import (load_bev_distance, process_actual_cost, load_and_clean_data, group_charging_data, process_hourly_charging_data, add_smart_avg, merge_and_calculate_costs, update_savings_columns, plot_saving_ev_vs_distance)
import warnings
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# %%  File paths for data
batt_file = "/Users/haniftayarani/V2G_Project/Travel_data/Battery_Price_Per_kWh_Estimations.csv"
bev_file = "/Users/haniftayarani/V2G_Project/data.csv"
ev_cost_file = '/Users/haniftayarani/V2G_Project/Results/Actual/Actual_EV_rate_cost.xlsx'
tou_cost_file = '/Users/haniftayarani/V2G_Project/Results/Actual/Actual_TOU_cost.xlsx'

# Load and process BEV distance data
bev_distance = load_bev_distance(bev_file)
batt_price = pd.read_csv(batt_file)
# Process actual costs
actual_cost = process_actual_cost(ev_cost_file, tou_cost_file)

# Load and clean the N and P datasets
loaded_N_data_battery = load_and_clean_data('all_hourly_charging_N_data_battery.pkl')
loaded_P_data_battery = load_and_clean_data('all_hourly_charging_P_data_battery.pkl')

# Group the data
all_hourly_charging_N_data_grouped_battery = group_charging_data(loaded_N_data_battery)
all_hourly_charging_P_data_grouped_battery = group_charging_data(loaded_P_data_battery)
# %%

# Process the hourly charging data for N and P datasets
all_hourly_charging_data_grouped_battery = process_hourly_charging_data(all_hourly_charging_N_data_grouped_battery, all_hourly_charging_P_data_grouped_battery)
# Add smart averages and percentage drop
all_hourly_charging_data_grouped_battery = add_smart_avg(all_hourly_charging_data_grouped_battery)

# Merge costs and calculate the total cost
all_hourly_charging_data_grouped_battery = merge_and_calculate_costs(all_hourly_charging_data_grouped_battery, actual_cost, bev_distance)

all_hourly_charging_data_grouped_battery = update_savings_columns(all_hourly_charging_data_grouped_battery, batt_price, current_year=2023, v2g_cost=7300, v1g_cost=500, v1g_cost_19kw=1600)

all_hourly_charging_data_grouped_battery = all_hourly_charging_data_grouped_battery.reset_index(drop=True)

all_hourly_charging_data_grouped_battery_summary = all_hourly_charging_data_grouped_battery.groupby(["Vehicle", "GHG Cost", "Charging_Behavior"]).apply(lambda x: x.loc[x['Saving_EV'].idxmax()])

all_hourly_charging_data_grouped_battery_summary_Actual = all_hourly_charging_data_grouped_battery_summary[all_hourly_charging_data_grouped_battery_summary["Charging_Behavior"] == "Actual"].reset_index(drop=True)
all_hourly_charging_data_grouped_battery_summary_Actual_0 = all_hourly_charging_data_grouped_battery_summary_Actual[all_hourly_charging_data_grouped_battery_summary_Actual["GHG Cost"] == 0]
all_hourly_charging_data_grouped_battery_summary_Actual_50 = all_hourly_charging_data_grouped_battery_summary_Actual[all_hourly_charging_data_grouped_battery_summary_Actual["GHG Cost"] == 0.05]
all_hourly_charging_data_grouped_battery_summary_Actual_191 = all_hourly_charging_data_grouped_battery_summary_Actual[all_hourly_charging_data_grouped_battery_summary_Actual["GHG Cost"] == 0.191]

all_hourly_charging_data_grouped_battery_summary_Potential = all_hourly_charging_data_grouped_battery_summary[all_hourly_charging_data_grouped_battery_summary["Charging_Behavior"] == "Potential"].reset_index(drop=True)
all_hourly_charging_data_grouped_battery_summary_Potential_0 = all_hourly_charging_data_grouped_battery_summary_Potential[all_hourly_charging_data_grouped_battery_summary_Potential["GHG Cost"] == 0]
all_hourly_charging_data_grouped_battery_summary_Potential_50 = all_hourly_charging_data_grouped_battery_summary_Potential[all_hourly_charging_data_grouped_battery_summary_Potential["GHG Cost"] == 0.05]
all_hourly_charging_data_grouped_battery_summary_Potential_191 = all_hourly_charging_data_grouped_battery_summary_Potential[all_hourly_charging_data_grouped_battery_summary_Potential["GHG Cost"] == 0.191]

plot_saving_ev_vs_distance(all_hourly_charging_data_grouped_battery_summary_Actual_191, add_actual_lines=True, add_potential_lines=False, ylim=3500, text_size=18)
plot_saving_ev_vs_distance(all_hourly_charging_data_grouped_battery_summary_Potential_191, add_actual_lines=False, add_potential_lines=False, ylim=5500, text_size=18)
(all_hourly_charging_data_grouped_battery_summary_Potential_191["Saving_TOU"]/all_hourly_charging_data_grouped_battery_summary_Potential_191["average_smart_years"]).mean()
(all_hourly_charging_data_grouped_battery_summary_Actual_191["Saving_TOU"]/all_hourly_charging_data_grouped_battery_summary_Potential_191["average_smart_years"]).mean()
