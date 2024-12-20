# %%
import pandas as pd
from parking import (load_bev_distance, process_actual_cost, update_savings_columns1, group_charging_data,
                     process_hourly_charging_data, add_smart_avg, merge_and_calculate_costs, plot_saving_ev_vs_distance)
import warnings
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
# %%  File paths for data


def process_and_plot_utility_data(utility_name, pkl_path, ylim1, ylim2):
    batt_file = "/Users/haniftayarani/V2G_Project/Travel_data/Battery_Price_Per_kWh_Estimations.csv"
    bev_file = "/Users/haniftayarani/V2G_Project/data.csv"

    # Load and process BEV distance data
    bev_distance_input = load_bev_distance(bev_file)
    batt_price_input = pd.read_csv(batt_file)
    ev_cost_file = f'/Users/haniftayarani/V2G_Project/Results/Actual/Actual_{utility_name}_EV_cost.xlsx'
    tou_cost_file = f'/Users/haniftayarani/V2G_Project/Results/Actual/Actual_{utility_name}_TOU_cost.xlsx'
    actual_cost_input = process_actual_cost(ev_cost_file, tou_cost_file)
    # Load the data for the specified utility
    normal_pkl = f"{pkl_path}/combined_hourly_data_normal_{utility_name}.pkl"
    parking_pkl = f"{pkl_path}/combined_hourly_data_parking_{utility_name}.pkl"

    combined_hourly_data_normal = pd.read_pickle(normal_pkl)
    combined_hourly_data_parking = pd.read_pickle(parking_pkl)

    # Group the data
    grouped_normal = group_charging_data(combined_hourly_data_normal)
    grouped_parking = group_charging_data(combined_hourly_data_parking)

    # Process the data
    all_hourly_charging_data_grouped = process_hourly_charging_data(grouped_normal, grouped_parking)
    all_hourly_charging_data_grouped = add_smart_avg(all_hourly_charging_data_grouped)
    all_hourly_charging_data_grouped = merge_and_calculate_costs(all_hourly_charging_data_grouped, actual_cost_input, bev_distance_input)
    all_hourly_charging_data_grouped = update_savings_columns1(
        all_hourly_charging_data_grouped,
        batt_price_input,
        current_year=2023,
        v2g_cost=7300,
        v1g_cost=0,
        v1g_cost_19kw=1300,
        interest_rate=0.05
    )
    all_hourly_charging_data_grouped = all_hourly_charging_data_grouped.reset_index(drop=True)

    # Generate summaries
    summary = all_hourly_charging_data_grouped.groupby(["Vehicle", "Scenario"]).apply(
        lambda x: x.loc[x['Saving_EV'].idxmax()]
    )
    summary_actual = summary[summary["Scenario"] == "No_change"].reset_index(drop=True)
    summary_potential = summary[summary["Scenario"] == "With_change"].reset_index(drop=True)

    # Plot the figures
    plot_saving_ev_vs_distance(summary_actual, add_actual_lines=False, add_potential_lines=False, ylim=ylim1, text_size=18, title=utility_name)
    plot_saving_ev_vs_distance(summary_potential, add_actual_lines=False, add_potential_lines=False, ylim=ylim2, text_size=18, title=utility_name)
    return all_hourly_charging_data_grouped, summary, summary_actual, summary_potential
# %%


all_hourly_charging_data_grouped_pge, summary_pge, summary_actual_pge, summary_potential_pge = (
    process_and_plot_utility_data("PGE", "/Users/haniftayarani/V2G_Project/Hourly_data", 5500, 7000))

all_hourly_charging_data_grouped_sce, summary_pge_sce, summary_actual_sce, summary_potential_sce = (
    process_and_plot_utility_data("SCE", "/Users/haniftayarani/V2G_Project/Hourly_data", 5500, 7000))

all_hourly_charging_data_grouped_sdge, summary_pge_sdge, summary_actual_sdge, summary_potential_sdge = (
    process_and_plot_utility_data("SDGE", "/Users/haniftayarani/V2G_Project/Hourly_data",5500, 7000))

all_hourly_charging_data_grouped_smud, summary_pge_smud, summary_actual_smud, summary_potential_smud= (
    process_and_plot_utility_data("SMUD", "/Users/haniftayarani/V2G_Project/Hourly_data",1500, 1500))

