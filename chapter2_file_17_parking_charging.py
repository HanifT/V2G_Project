import pandas as pd
import json
import os
import logging
from parking import charging_dataframe_parking
logging.getLogger('pyomo.core').setLevel(logging.ERROR)
# %%


def real_time_data_parking(list):
    trip_data = pd.read_csv("data.csv")
    final_dataframes_charging = charging_dataframe_parking(trip_data, 0)
    charging_data = final_dataframes_charging

    charging_data = charging_data.drop(columns=["ts_start", "ts_end", "charge_type_count"]).sort_values(by=["vehicle_name", "start_time_local"]).reset_index(drop=True)
    trip_data = trip_data.drop(columns=["ts_start", "ts_end", "charge_type_count"]).sort_values(by=["vehicle_name", "start_time_local"]).reset_index(drop=True)
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
    trip_data['Model'] = trip_data['vehicle_model'].apply(classify_model)

    charging_data['start_time_charging'] = pd.to_datetime(charging_data['start_time_charging'])
    charging_data['end_time_charging'] = pd.to_datetime(charging_data["end_time_charging"])
    charging_data["next_departure_time"] = pd.to_datetime(charging_data["next_departure_time"])

    trip_data['start_time_local'] = pd.to_datetime(trip_data['start_time_local'])
    trip_data['end_time_local'] = pd.to_datetime(trip_data["end_time_local"])
    trip_data["next_departure_time"] = pd.to_datetime(trip_data["next_departure_time"])

    charging_data = charging_data[charging_data["vehicle_name"].isin(list)]

    charging_data["end_time_local"] = pd.to_datetime(charging_data['end_time_local'])
    trip_data = trip_data[trip_data["vehicle_name"].isin(list)]

    def calculate_hour_of_year_charging(df):
        df['hour_of_year_start'] = df['end_time_local'].apply(lambda x: (x.year - df['end_time_local'].min().year) * 8760 + (x.dayofyear - 1) * 24 + x.hour)
        df['hour_of_year_end'] = df['next_departure_time'].apply(lambda x: (x.year - df['next_departure_time'].min().year) * 8760 + (x.dayofyear - 1) * 24 + x.hour)
        return df

    def calculate_hour_of_year_trip(df):
        df['hour_of_year_start'] = df['start_time_local'].apply(lambda x: (x.year - df['start_time_local'].min().year) * 8760 + (x.dayofyear - 1) * 24 + x.hour)
        df['hour_of_year_end'] = df['end_time_local'].apply(lambda x: (x.year - df['end_time_local'].min().year) * 8760 + (x.dayofyear - 1) * 24 + x.hour)
        return df

    # Group charging data by vehicle_name and apply the function
    charging_data = charging_data.groupby('vehicle_name').apply(calculate_hour_of_year_charging)

    # Group trip data by vehicle_name and apply the function
    trip_data = trip_data.groupby('vehicle_name').apply(calculate_hour_of_year_trip)

    def create_charging_dictionary(df):
        ch_dict = {}  # Initialize an empty dictionary

        # Iterate through each row of the dataframe
        for index, row in df.iterrows():
            vehicle_name = row['vehicle_name']
            start_time = row['hour_of_year_start']
            end_time = row['hour_of_year_end']
            soc_init = row['battery[soc][start][charging]']
            soc_end = row['battery[soc][end][charging]']
            soc_need = row['SOC_need_next_charge']
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
                    ch_dict[vehicle_name][hour]['soc_init'] = soc_init
                    ch_dict[vehicle_name][hour]['soc_end'] = soc_end
                    ch_dict[vehicle_name][hour]['soc_need'] = soc_need
                    ch_dict[vehicle_name][hour]['bat_cap'] = bat_cap
                    ch_dict[vehicle_name][hour]['charge_type'] = charge_type
                    ch_dict[vehicle_name][hour]['location'] = location

                # Otherwise, add a new entry for the hour
                else:
                    ch_dict[vehicle_name][hour] = {
                        'charging_indicator': 1,
                        'soc_init': soc_init,
                        'end_time': end_time,
                        'soc_need': soc_need,
                        'soc_end': soc_end,
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
                    df[vehicle_name][hour] = {'charging_indicator': 0, 'soc_init': 0, 'soc_need': 0, 'end_time': 0, 'soc_end': 0, 'charge_type': "None", 'location': "None", 'bat_cap': 0}

        return df

    def adj_charging_dictionary_trip(df):
        model_dict = {vehicle_name: next(iter(hours_data.values()))['model'] for vehicle_name, hours_data in df.items()}

        for vehicle_name, hours_data in df.items():
            model = model_dict[vehicle_name]
            max_hour = max(hours_data.keys(), default=0)
            for hour in range(max_hour + 1):
                if hour not in hours_data:
                    df[vehicle_name][hour] = {'soc_diff ': 0, 'model': model}

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

    trip_dict = calculate_distance_per_hour(trip_data)
    trip_dict = adj_charging_dictionary_trip(trip_dict)
    trip_dict = sort_nested_dictionary(trip_dict)

    # Create the charging dictionary
    charging_dict = create_charging_dictionary(charging_data)
    charging_dict = adj_charging_dictionary(charging_dict)
    charging_dict = sort_nested_dictionary(charging_dict)

    # Example usage:
    merged_dict = merge_nested_dicts(trip_dict, charging_dict)

    # Function to replace zero battery capacity with the first non-zero value for each vehicle
    def replace_zero_bat_cap(merged_dict):
        for vehicle, hours in merged_dict.items():
            # Find the first non-zero bat_cap value
            non_zero_bat_cap = next((data.get('bat_cap') for hour, data in hours.items() if data.get('bat_cap', 0) != 0), None)

            if non_zero_bat_cap is not None:
                for hour, data in hours.items():
                    if data.get('bat_cap', 0) == 0:
                        data['bat_cap'] = non_zero_bat_cap
        return merged_dict

    # Replace zero battery capacity with the first non-zero value found in each vehicle's data
    replace_zero_bat_cap(merged_dict)

    with open("trip_dict_parking.json", "w") as json_file:
        json.dump(trip_dict, json_file)

    with open("charging_dict_parking.json", "w") as json_file:
        json.dump(charging_dict, json_file)

    with open("merged_dict_parking.json", "w") as json_file:
        json.dump(merged_dict, json_file)


# # %%
# vehicle_list = ["P_1087", "P_1091", "P_1092", "P_1093", "P_1094", "P_1098", "P_1100", 'P_1109', 'P_1111', "P_1112", "P_1123", "P_1125",
#                 "P_1125a", "P_1127", 'P_1131', 'P_1132', 'P_1135', 'P_1137', "P_1141", "P_1143", 'P_1217', 'P_1253', 'P_1257', 'P_1260',
#                 'P_1271', 'P_1272', 'P_1279', 'P_1280', 'P_1281', 'P_1285', 'P_1288', 'P_1294', 'P_1295', 'P_1296', 'P_1304', 'P_1307',
#                 "P_1357", "P_1367", 'P_1375', 'P_1353', 'P_1368', 'P_1371', "P_1376", 'P_1393', "P_1414", 'P_1419', 'P_1421', 'P_1422', 'P_1424', 'P_1427']
#
# real_time_data_parking(vehicle_list)
#
# with open("merged_dict_parking.json", "r") as json_file:
#     merged_dict_parking = json.load(json_file)
#
# with open("merged_dict.json", "r") as json_file:
#     merged_dict = json.load(json_file)
#
#
# def dic_data(df):
#
#     # # Convert nested dictionary to DataFrame
#     rows = []
#     for outer_key, inner_dict in df.items():
#         for inner_key, values_dict in inner_dict.items():
#             row = {**{'vehicle': outer_key, 'hour': inner_key}, **values_dict}
#             rows.append(row)
#
#     df = pd.DataFrame(rows)
#     return df
#
# normal_test = dic_data(merged_dict)
# parking_test = dic_data(merged_dict_parking)
