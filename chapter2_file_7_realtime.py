import pandas as pd
import json
from parking import charging_dataframe
# %%


def real_time_data(list):

    trip_data = pd.read_csv("data.csv")
    final_dataframes_charging = charging_dataframe(trip_data, 0)
    charging_data = final_dataframes_charging

    charging_data = charging_data.drop(columns=["Unnamed: 0.1", "Unnamed: 0", "ts_start", "ts_end", "charge_type_count"]).sort_values(by=["vehicle_name", "start_time_local"]).reset_index(drop=True)
    trip_data = trip_data.drop(columns=["Unnamed: 0.1", "Unnamed: 0", "ts_start", "ts_end", "charge_type_count"]).sort_values(by=["vehicle_name", "start_time_local"]).reset_index(drop=True)

    charging_data['start_time_charging'] = pd.to_datetime(charging_data['start_time_charging'])
    charging_data['end_time_charging'] = pd.to_datetime(charging_data["end_time_charging"])
    charging_data["next_departure_time"] = pd.to_datetime(charging_data["next_departure_time"])

    trip_data['start_time_local'] = pd.to_datetime(trip_data['start_time_local'])
    trip_data['end_time_local'] = pd.to_datetime(trip_data["end_time_local"])
    trip_data["next_departure_time"] = pd.to_datetime(trip_data["next_departure_time"])

    charging_data = charging_data[charging_data["vehicle_name"].isin(list)]
    # charging_data = charging_data[charging_data["vehicle_name"].isin(["P_1111"])]
    trip_data = trip_data[trip_data["vehicle_name"].isin(list)]
    # trip_data = trip_data[trip_data["vehicle_name"].isin(["P_1111"])]

    def calculate_hour_of_year_charging(df):
        df['hour_of_year_start'] = df['start_time_charging'].apply(lambda x: (x.year - df['start_time_charging'].min().year) * 8760 + (x.dayofyear - 1) * 24 + x.hour)
        df['hour_of_year_end'] = df['next_departure_time'].apply(lambda x: (x.year - df['next_departure_time'].min().year) * 8760 + (x.dayofyear - 1) * 24 + x.hour)
        return df

    def calculate_hour_of_year_trip(df):
        df['hour_of_year_start'] = df['start_time_local'].apply(lambda x: (x.year - df['start_time_local'].min().year) * 8760 + (x.dayofyear - 1) * 24 + x.hour)
        df['hour_of_year_end'] = df['end_time_local'].apply(lambda x: (x.year - df['end_time_local'].min().year) * 8760 + (x.dayofyear - 1) * 24 + x.hour)
        return df

    # Group charging data by vehicle_name and calculate hour of the year for each group
    charging_data = charging_data.groupby('vehicle_name').apply(calculate_hour_of_year_charging)
    trip_data = trip_data.groupby("vehicle_name").apply(calculate_hour_of_year_trip)

    trip_data.loc[~trip_data["energy[charge_type][type]"].isna(), "hour_of_year_end"] -= 1

    # %%

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
                # Otherwise, add a new entry for the hour
                else:
                    ch_dict[vehicle_name][hour] = {
                        'charging_indicator': 1,
                        'soc_init': soc_init,
                        'end_time': end_time,
                        'soc_need': soc_need,
                        'soc_end': soc_end,
                        'bat_cap': bat_cap,
                        'charge_type': charge_type
                    }

        return ch_dict

    def adj_charging_dictionary(df):
        for vehicle_name, hours_data in df.items():
            max_hour = max(hours_data.keys(), default=0)
            for hour in range(max_hour + 1):
                if hour not in hours_data:
                    df[vehicle_name][hour] = {'charging_indicator': 0, 'soc_init': 0, 'soc_need': 0, 'end_time': 0, 'soc_end': 0, 'charge_type': "None", 'bat_cap': 100}

        return df

    def adj_charging_dictionary_trip(df):
        for vehicle_name, hours_data in df.items():
            max_hour = max(hours_data.keys(), default=0)
            for hour in range(max_hour + 1):
                if hour not in hours_data:
                    df[vehicle_name][hour] = {'distance': 0}

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
            distance = row['distance']
            hour_of_year_start = int(row['hour_of_year_start'])
            hour_of_year_end = int(row['hour_of_year_end'])

            num_hours = hour_of_year_end - hour_of_year_start + 1
            distance_per_hour = distance / (num_hours+0.00001)

            hour_values = {}
            for hour in range(hour_of_year_start, hour_of_year_end + 1):
                hour_values[hour] = {'distance': distance_per_hour}

            if vehicle_name in vehicle_dict:
                vehicle_dict[vehicle_name].update(hour_values)
            else:
                vehicle_dict[vehicle_name] = hour_values

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

    # %% saving json file

    with open("trip_dict.json", "w") as json_file:
        json.dump(trip_dict, json_file)

    with open("charging_dict.json", "w") as json_file:
        json.dump(charging_dict, json_file)

    with open("merged_dict.json", "w") as json_file:
        json.dump(merged_dict, json_file)
