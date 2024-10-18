# %%
import os
import json
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from parking import charging_dataframe
from price_factor import tou_price, ev_rate_price
# %%

GHG_data = pd.read_csv("CISO.csv")
GHG_dict = dict(enumerate(GHG_data.iloc[:, 0]))


def normal_load(list):
    trip_data = pd.read_csv("data.csv")
    final_dataframes_charging = charging_dataframe(trip_data, 0)
    charging_data = final_dataframes_charging

    charging_data = charging_data.drop(columns=["ts_start", "ts_end", "charge_type_count"]).sort_values(by=["vehicle_name", "start_time_local"]).reset_index(drop=True)
    # Fill NaN values in 'start_time_charging' with 'end_time_local'
    charging_data['charging_speed'].fillna(6.6, inplace=True)

    def classify_model(vehicle_model):
        if vehicle_model.startswith('Model S'):
            return 'Tesla'
        elif vehicle_model.startswith('Bolt'):
            return 'Chevy'
        else:
            return 'Other'

    # Apply the function to create the new column
    charging_data['Model'] = charging_data['vehicle_model'].apply(classify_model)

    charging_data['start_time_charging'] = pd.to_datetime(charging_data['start_time_charging'])
    charging_data['end_time_charging'] = pd.to_datetime(charging_data["end_time_charging"])
    charging_data["next_departure_time"] = pd.to_datetime(charging_data["next_departure_time"])

    charging_data = charging_data[charging_data["vehicle_name"].isin(list)]

    charging_data["end_time_local"] = pd.to_datetime(charging_data['end_time_local'])
    # charging_data["next_departure_time"] = pd.to_datetime(charging_data["next_departure_time"]).dt.tz_convert('America/Los_Angeles')

    def calculate_hour_of_year_charging(df):
        df['hour_of_year_start'] = df['start_time_charging'].apply(lambda x: (x.year - df['start_time_charging'].min().year) * 8760 + (x.dayofyear - 1) * 24 + x.hour)
        df['hour_of_year_end'] = df['end_time_charging'].apply(lambda x: (x.year - df['end_time_charging'].min().year) * 8760 + (x.dayofyear - 1) * 24 + x.hour)
        return df

    def calculate_hour_of_year_trip(df):
        df['hour_of_year_start'] = df['start_time_local'].apply(lambda x: (x.year - df['start_time_local'].min().year) * 8760 + (x.dayofyear - 1) * 24 + x.hour)
        df['hour_of_year_end'] = df['end_time_local'].apply(lambda x: (x.year - df['end_time_local'].min().year) * 8760 + (x.dayofyear - 1) * 24 + x.hour)
        return df

    # Group charging data by vehicle_name and apply the function
    charging_data = charging_data.groupby('vehicle_name').apply(calculate_hour_of_year_charging)

    charging_data['SOC_Diff'] = charging_data["battery[soc][end][charging]"] - charging_data["battery[soc][start][charging]"]

    def create_charging_dictionary(df):
        ch_dict = {}  # Initialize an empty dictionary

        # Iterate through each row of the dataframe
        for index, row in df.iterrows():
            vehicle_name = row['vehicle_name']
            start_time = row['hour_of_year_start']
            end_time = row['hour_of_year_end']
            bat_cap = row['bat_cap']
            charge_type = row['charge_type']
            location = row['destination_label']

            # Check if the vehicle name already exists in the dictionary
            if vehicle_name not in ch_dict:
                ch_dict[vehicle_name] = {}  # Initialize an empty dictionary for the vehicle

            # Iterate through the range of hours from start_time to end_time for the current vehicle
            for hour in range(start_time, end_time + 1):
                # If the hour already exists in the dictionary, update the values
                if hour in ch_dict[vehicle_name]:
                    ch_dict[vehicle_name][hour]['charging_indicator'] = 1
                    ch_dict[vehicle_name][hour]['end_time'] = end_time
                    ch_dict[vehicle_name][hour]['bat_cap'] = bat_cap
                    ch_dict[vehicle_name][hour]['charge_type'] = charge_type
                    ch_dict[vehicle_name][hour]['location'] = location

                # Otherwise, add a new entry for the hour
                else:
                    ch_dict[vehicle_name][hour] = {
                        'charging_indicator': 1,
                        'end_time': end_time,
                        'bat_cap': bat_cap,
                        'charge_type': charge_type,
                        'location': location,

                    }

        return ch_dict

    def adj_charging_dictionary(df):
        for vehicle_name, hours_data in df.items():
            max_hour = max(hours_data.keys(), default=0)
            for hour in range(max_hour + 1):
                if hour not in hours_data:
                    df[vehicle_name][hour] = {'charging_indicator': 0, 'end_time': 0, 'charge_type': "None", 'location': "None", 'bat_cap': 0}

        return df

    def adj_charging_dictionary_trip(df):
        # df = charging_dict_soc.copy()
        model_dict = {vehicle_name: next(iter(hours_data.values()))['model'] for vehicle_name, hours_data in df.items()}

        for vehicle_name, hours_data in df.items():
            model = model_dict[vehicle_name]
            max_hour = max(hours_data.keys(), default=0)
            for hour in range(max_hour + 1):
                if hour not in hours_data:
                    df[vehicle_name][hour] = {'model': model}

        return df

    def sort_nested_dictionary(df):
        sorted_dict = {}
        for vehicle_name, hours_data in df.items():
            # Sort the nested dictionary based on the hour key
            sorted_hours_data = dict(sorted(hours_data.items()))
            sorted_dict[vehicle_name] = sorted_hours_data
        return sorted_dict

    def calculate_distance_per_hour(data):
        vehicle_dict = {}
        for _, row in data.iterrows():
            vehicle_name = row["vehicle_name"]
            soc_diff = row['SOC_Diff']
            model = row["Model"]
            hour_of_year_start = int(row['hour_of_year_start'])
            hour_of_year_end = int(row['hour_of_year_end'])

            num_hours = hour_of_year_end - hour_of_year_start

            if num_hours > 0:
                num_hours = num_hours + 1
            elif num_hours == 0:
                num_hours = 1

            SOC_diff_per_hour = soc_diff / num_hours

            hour_values = {}
            for hour in range(hour_of_year_start, hour_of_year_end + 1):
                if vehicle_name not in vehicle_dict:
                    vehicle_dict[vehicle_name] = {}
                if hour not in vehicle_dict[vehicle_name]:
                    vehicle_dict[vehicle_name][hour] = {'soc_diff': 0, 'model': model}

                vehicle_dict[vehicle_name][hour]['soc_diff'] += SOC_diff_per_hour

        return vehicle_dict

    def merge_nested_dicts(dict1, dict2):
        merged_dict = {}
        all_keys = set(dict1.keys()).union(dict2.keys())
        for key in all_keys:
            if key in dict1 and key in dict2:
                if isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
                    # Both values are dictionaries - recursively merge them
                    merged_dict[key] = merge_nested_dicts(dict1[key], dict2[key])
                else:
                    # Values are not both dictionaries. For simplicity,
                    # we'll prioritize the value from dict1.
                    merged_dict[key] = dict1[key]
            elif key in dict1:
                merged_dict[key] = dict1[key]
            else:
                merged_dict[key] = dict2[key]

        return merged_dict

    charging_dict_soc = calculate_distance_per_hour(charging_data)
    charging_dict_soc = adj_charging_dictionary_trip(charging_dict_soc)
    charging_dict_soc = sort_nested_dictionary(charging_dict_soc)

    # Create the charging dictionary
    charging_dict = create_charging_dictionary(charging_data)
    charging_dict = adj_charging_dictionary(charging_dict)
    charging_dict = sort_nested_dictionary(charging_dict)

    # Example usage:
    merged_dict = merge_nested_dicts(charging_dict_soc, charging_dict)

    for vehicle_name, hours_data in merged_dict.items():
        for hour, hour_data in hours_data.items():
            if pd.isna(hour_data.get('soc_diff')):
                hour_data['soc_diff'] = 0

    normal_load_BEV = merged_dict
    #
    # total_soc_diff = 0
    # missing_soc_diff_keys = []
    #
    # # Calculate the sum of soc_diff across all nested dictionaries
    # for inner_dict in normal_load_BEV.values():
    #     for v in inner_dict.values():
    #         if 'soc_diff' in v:
    #             total_soc_diff += v['soc_diff']
    #         else:
    #             missing_soc_diff_keys.append(v)  # Track dictionaries missing 'soc_diff ' key
    #
    # print("Total sum of soc_diff:", total_soc_diff)
    # charging_data["SOC_Diff"].sum()

    return normal_load_BEV


vehicle_list = ["P_1087", "P_1091", "P_1092", "P_1093", "P_1094", "P_1098", "P_1100", 'P_1109', 'P_1111', "P_1112", "P_1123", "P_1125",
                "P_1125a", "P_1127", 'P_1131', 'P_1132', 'P_1135', 'P_1137', "P_1141", "P_1143", 'P_1217', 'P_1253', 'P_1257', 'P_1260',
                'P_1271', 'P_1272', 'P_1279', 'P_1280', 'P_1281', 'P_1285', 'P_1288', 'P_1294', 'P_1295', 'P_1296', 'P_1304', 'P_1307',
                "P_1357", "P_1367", 'P_1375', 'P_1353', 'P_1368', 'P_1371', "P_1376", 'P_1393', "P_1414", 'P_1419', 'P_1421', 'P_1422', 'P_1424', 'P_1427']

BEV_load_curve = normal_load(vehicle_list)

with open("normal_load_BEV.json", "w") as json_file:
    json.dump(BEV_load_curve, json_file)

with open("normal_load_BEV.json", "r") as json_file:
    BEV_load_curve = json.load(json_file)


# # Convert nested dictionary to DataFrame
rows = []
for outer_key, inner_dict in BEV_load_curve.items():
    for inner_key, values_dict in inner_dict.items():
        row = {**{'vehicle': outer_key, 'hour': inner_key}, **values_dict}
        rows.append(row)

df = pd.DataFrame(rows)



# %%

with open("combined_price_PGE_average.json", "r") as json_file:
    combined_price_PGE_average = json.load(json_file)

tou_prices = tou_price(550, 450, 430, 400)
ev_rate_prices = ev_rate_price(310, 510, 620, 310, 480, 490)

# Convert keys to integers
RT_PGE = {int(key): value for key, value in combined_price_PGE_average.items()}
tou_prices = {int(key): value for key, value in tou_prices.items()}
ev_rate_prices = {int(key): value for key, value in ev_rate_prices.items()}



# %%

def price_curve(vehicle_data, electricity_price, ghg, dc_fast_price, degradation_slope):
    vehicle_costs = {}
    vehicle_total_charge = {}
    vehicle_total_ghg = {}
    hourly_data = []

    # Iterate through vehicle data and calculate costs
    for vehicle_id, hours in vehicle_data.items():
        total_charge = 0  # Initialize cumulative charge for the vehicle
        vehicle_costs[vehicle_id] = []
        ghg_cost_accumulated = 0  # Initialize cumulative GHG cost

        for hour, data in hours.items():
            charge_type = data.get('charge_type', 'None')
            soc_diff = data.get('soc_diff', 0)
            batt = data.get('bat_cap', 80)
            hour_index = int(hour)

            if charge_type != 'DC_FAST' and hour_index in electricity_price:
                cost = (soc_diff/100 * batt) * 1.05 * (electricity_price[hour_index]) / 1000
                ghg_cost = ((soc_diff * batt)/100) * 1.05 * (ghg[hour_index]) / 1000
                total_charge += soc_diff/100 * batt
                vehicle_costs[vehicle_id].append(cost)
                ghg_cost_accumulated += ghg_cost

            elif charge_type == 'DC_FAST':
                cost = (soc_diff/100 * batt) * 1.05 * dc_fast_price / 1000
                ghg_cost = ((soc_diff * batt)/100) * 1.05 * (ghg[hour_index]) / 1000
                total_charge += soc_diff/100 * batt
                vehicle_costs[vehicle_id].append(cost)
                ghg_cost_accumulated += ghg_cost

            # Append hourly data
            hourly_data.append({
                'Vehicle': vehicle_id,
                'Hour': hour_index,
                'Charge_Type': charge_type,
                'SOC_Diff': soc_diff,
                'Battery_Capacity': batt,
                'Electricity_Cost': cost,
                'GHG_Cost': ghg_cost,
                'Total_Charge': total_charge
            })

        vehicle_total_charge[vehicle_id] = total_charge
        vehicle_total_ghg[vehicle_id] = ghg_cost_accumulated

    # Calculate degradation cost and create a list of dictionaries for DataFrame creation
    df_data = []
    for vehicle_id, costs in vehicle_costs.items():
        vehicle_cost = sum(costs)
        total_charge = vehicle_total_charge[vehicle_id]
        degradation_cost = degradation_slope * total_charge * 1.05  # Degradation cost calculation
        ghg_cost = vehicle_total_ghg[vehicle_id] * 0.05  # Use accumulated GHG cost for this vehicle
        df_data.append({'Vehicle': vehicle_id, 'Electricity_Cost': vehicle_cost, 'Degradation_Cost': degradation_cost, 'GHG_Cost': ghg_cost, 'Total Charge': total_charge})

    # Convert list of dictionaries to DataFrame for aggregated data
    df_aggregated = pd.DataFrame(df_data)

    # Convert list of dictionaries to DataFrame for hourly data
    df_hourly = pd.DataFrame(hourly_data)

    # # Filter out zero values before calculating the mean
    df_hourly = df_hourly[df_hourly["Charge_Type"] != "None"]
    df_hourly = df_hourly[df_hourly["SOC_Diff"] > 0]
    # # Calculate the mean of non-zero Electricity_Cost for each daily hour
    df_hourly["X_CHR"] = (df_hourly["SOC_Diff"] * df_hourly["Battery_Capacity"]) / 100
    df_hourly["daily_hour"] = df_hourly["Hour"] % 24
    df_hourly_mean = df_hourly.groupby(["Vehicle", "daily_hour"])["X_CHR"].mean().reset_index(drop=False)
    #
    df_hourly_mean = df_hourly_mean.groupby("daily_hour").agg({
        "X_CHR": "sum",
    }).reset_index(drop=False)


    return df_aggregated, df_hourly_mean, df_hourly


RT_cost, RT_cost_hourly, RT_cost_hourly_in = price_curve(BEV_load_curve, RT_PGE, GHG_dict, 560, 0.021)
TOU_cost, TOU_cost_hourly, TOU_cost_hourly_in = price_curve(BEV_load_curve, tou_prices, GHG_dict, 560, 0.021)
EV_rate_cost, EV_cost_hourly, EV_cost_hourly_in = price_curve(BEV_load_curve, ev_rate_prices, GHG_dict, 560, 0.021)

# %%
# Define the file paths
rt_cost_file = 'Actual_RT_cost.xlsx'
tou_cost_file = 'Actual_TOU_cost.xlsx'
ev_rate_cost_file = 'Actual_EV_rate_cost.xlsx'

# Save RT_cost to an Excel file
with pd.ExcelWriter(rt_cost_file, engine='xlsxwriter') as writer:
    RT_cost.to_excel(writer, sheet_name='Individual Costs')
print(f"RT_cost saved: {rt_cost_file}")

# Save TOU_cost to an Excel file
with pd.ExcelWriter(tou_cost_file, engine='xlsxwriter') as writer:
    TOU_cost.to_excel(writer, sheet_name='Individual Costs')
print(f"TOU_cost saved: {tou_cost_file}")

# Save EV_rate_cost to an Excel file
with pd.ExcelWriter(ev_rate_cost_file, engine='xlsxwriter') as writer:
    EV_rate_cost.to_excel(writer, sheet_name='Individual Costs')
print(f"EV_rate_cost saved: {ev_rate_cost_file}")

rt_cost_file_hourly = 'Actual_RT_cost_hourly.xlsx'
tou_cost_file_hourly = 'Actual_TOU_cost_hourly.xlsx'
ev_rate_cost_file_hourly = 'Actual_EV_rate_cost_hourly.xlsx'

# Save RT_cost to an Excel file
with pd.ExcelWriter(rt_cost_file_hourly, engine='xlsxwriter') as writer:
    RT_cost_hourly.to_excel(writer, sheet_name='Individual Costs')
print(f"RT_cost saved: {rt_cost_file_hourly}")

# Save TOU_cost to an Excel file
with pd.ExcelWriter(tou_cost_file_hourly, engine='xlsxwriter') as writer:
    TOU_cost_hourly.to_excel(writer, sheet_name='Individual Costs')
print(f"TOU_cost saved: {tou_cost_file_hourly}")

# Save EV_rate_cost to an Excel file
with pd.ExcelWriter(ev_rate_cost_file_hourly, engine='xlsxwriter') as writer:
    EV_cost_hourly.to_excel(writer, sheet_name='Individual Costs')
print(f"EV_rate_cost saved: {ev_rate_cost_file_hourly}")


rt_cost_file_hourly_in = 'Actual_RT_cost_hourly_in.xlsx'
tou_cost_file_hourly_in = 'Actual_TOU_cost_hourly_in.xlsx'
ev_rate_cost_file_hourly_in = 'Actual_EV_rate_cost_hourly_in.xlsx'

# Save RT_cost to an Excel file
with pd.ExcelWriter(rt_cost_file_hourly_in, engine='xlsxwriter') as writer:
    RT_cost_hourly_in.to_excel(writer, sheet_name='Individual Costs')
print(f"RT_cost saved: {rt_cost_file_hourly_in}")

# Save TOU_cost to an Excel file
with pd.ExcelWriter(tou_cost_file_hourly_in, engine='xlsxwriter') as writer:
    TOU_cost_hourly_in.to_excel(writer, sheet_name='Individual Costs')
print(f"TOU_cost saved: {tou_cost_file_hourly_in}")

# Save EV_rate_cost to an Excel file
with pd.ExcelWriter(ev_rate_cost_file_hourly_in, engine='xlsxwriter') as writer:
    EV_cost_hourly_in.to_excel(writer, sheet_name='Individual Costs')
print(f"EV_rate_cost saved: {ev_rate_cost_file_hourly_in}")

df.to_pickle('demand_curve.pkl')
