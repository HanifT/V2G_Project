import pandas as pd
import pytz
import numpy as np
from haversine import haversine, Unit
from datetime import datetime
from collections import Counter
import matplotlib.pyplot as plt
import statsmodels.api as sm


# %%
def read_time_series(all_files):
    df = pd.concat((pd.read_csv(f) for f in all_files))
    df = df.sort_values("Time")
    df1 = df
    df1.index = list(range(0, len(df1)))
    df1['Time1'] = pd.to_datetime(df1["Time"], format="%Y-%m-%dT%H:%M:%S.%f")
    df1['Time1'] = pd.to_datetime(df1['Time1']).apply(lambda x: x.replace(microsecond=0))
    df1['Time1'] = pd.to_datetime(df1['Time1']).apply(lambda x: x.replace(tzinfo=None))
    df1['Time2'] = df1['Time1'].apply(lambda x: x.tz_localize('UTC').tz_convert('US/Pacific'), convert_dtype=False)
    date_time = pd.to_datetime(df1['Time2'].tolist(), format='%Y-%m-%d %H:%M:%S')
    date_time1 = date_time.strftime("%M")
    df1['ts'] = df1.Time2.values.astype(np.int64) // 10 ** 9
    df1['year'] = df1['Time2'].dt.year
    df1['month'] = df1['Time2'].dt.month
    df1['day'] = df1['Time2'].dt.day
    df1['hour'] = df1['Time2'].dt.hour
    df1['minute'] = date_time1
    df1['second'] = df1['Time2'].dt.second
    df2 = df1[["ts", "year", "month", "day", "hour", "minute", "second", "Speed", "GpsLat", "GpsLon", "GpsAlt", "HVBattSOC", "OutsideAirTemp"]]
    df2 = df2.reset_index(drop=True)
    df2["hour_new"] = df2["hour"].astype(str) + df2["minute"]
    df2["hour_new"] = df2["hour_new"].astype(int)
    # df2 = df2.dropna(axis=0, subset=None, inplace=False)
    df2.drop(df.tail(1).index, inplace=True)
    df2["ymd"] = df2["year"].astype(str) + df2["month"].astype(str) + df2["day"].astype(str)
    df2["ymd"] = df2["ymd"].astype(int)
    df2["Destination_label"] = 0
    df2["charge_level"] = 0
    df2["id"] = 0
    return df1, df2


def read_clean(df_timeseries2, vehicle_name):
    data_trips_full = pd.read_csv("G:\\My Drive\\PycharmProjects\\LSTM\\bev_trips_full.csv", low_memory=False)
    data_trips = pd.read_csv("G:\\My Drive\\PycharmProjects\\LSTM\\bev_trips.csv")
    data_trips = data_trips.drop("destination_label", axis=1)
    data_trips = pd.merge(data_trips, data_trips_full[["id", "Lat", "Long", "cluster", "destination_label"]], how="left", on="id")
    data_charge = pd.read_csv("G:\\My Drive\\PycharmProjects\\LSTM\\bev_zcharges.csv", low_memory=False)
    # Define aggregation functions for each column
    agg_funcs = {
        'battery[soc][end]': 'last',
        'energy[charge_type][type]': 'last',
        'battery[soc][start]': 'first',
        'duration': 'sum',
        "total_energy": 'sum',
        'start_time': 'first',
        'end_time': 'last',
        'location[latitude]': "last",
        'location[longitude]': "last"
    }
    # Group by the specified columns and apply the aggregation functions
    data_charge_grouped = data_charge.groupby(['last_trip_id']).agg(agg_funcs)
    # Reset the index to get a flat DataFrame
    data_charge_grouped.reset_index(inplace=True)
    data_charge_grouped["id"] = data_charge_grouped["last_trip_id"]
    data = pd.merge(data_trips, data_charge_grouped, on="id", how="left")
    # Rename columns for clarity
    data.rename(columns={'duration_y': 'duration_charging',
                         'start_time_y': 'start_time_charging',
                         'end_time_y': 'end_time_charging',
                         'duration_x': 'duration_trip',
                         'start_time_x': 'start_time_trip',
                         'end_time_x': 'end_time_trip',
                         'battery[soc][end]_x': 'battery[soc][end][trip]',
                         'battery[soc][start]_x': 'battery[soc][start][trip]',
                         'battery[soc][end]_y': 'battery[soc][end][charging]',
                         'battery[soc][start]_y': 'battery[soc][start][charging]'}, inplace=True)

    data["energy[charge_type][type]"] = data["energy[charge_type][type]"].fillna("NA")
    data["charge_level"] = data["charge_level"].fillna("NA")
    data1 = data.groupby("id").tail(n=1)
    data1 = data1.reset_index(drop=True)
    data1.loc[(data1["charge_level"] == "NA") & (data1["energy[charge_type][type]"] != "NA"), "charge_level"] = data1["energy[charge_type][type]"]
    data1.loc[(data1["charge_level"] == "NA") & (data1["energy[charge_type][type]"] != "NA"), "charge_after"] = 1
    data1["start_time_local"] = pd.to_datetime(data1["start_time_ (local)"])
    data1["end_time_local"] = pd.to_datetime(data1["end_time_ (local)"])
    # Set the timezone to PST
    timezone = pytz.timezone('US/Pacific')
    # Convert the datetime to PST timezone
    data1["start_time_local"] = pd.to_datetime(data1["start_time_local"]).dt.tz_localize(timezone, ambiguous='NaT')
    data1["end_time_local"] = pd.to_datetime(data1["end_time_local"]).dt.tz_localize(timezone, ambiguous='NaT')
    # Convert datetime to timestamp
    data1["ts_start"] = data1.start_time_local.astype(np.int64) // 10 ** 9
    data1["ts_end"] = data1.end_time_local.astype(np.int64) // 10 ** 9

    data1.loc[(data1["destination_label"] == "Home") & (data1["energy[charge_type][type]"] == "DC_FAST"), "destination_label"] = "Other"
    data1.loc[(data1["destination_label"] == "Work") & (data1["energy[charge_type][type]"] == "DC_FAST"), "destination_label"] = "Other"

    data_input = data1[data1["vehicle_name"] == vehicle_name].copy()

    data_input = data_input.reset_index(drop=True)
    data_input["origin_label"] = data_input["destination_label"].shift(1)

    # Select relevant columns from df_timeseries2
    df3 = df_timeseries2[["ts", "id", 'charge_level', "Destination_label", "Speed", "HVBattSOC", "OutsideAirTemp", 'hour', "minute"]]

    # Reset index and set a placeholder value for the new column
    df3 = df3.reset_index(drop=True)
    df3["Origin_label"] = 0

    # Group by 'ts' and keep the last entry for each group
    df3 = df3.groupby('ts', as_index=False).last()
    # Reset index and sort by 'ts'
    df3 = df3.reset_index(drop=True)
    df3.sort_values("ts", inplace=True)
    # Create a new DataFrame 'data_input_new' from 'data_input'
    data_input_new = data_input.copy()
    # Convert 'ts_start' to int and sort the DataFrame by 'ts_start'
    data_input_new["ts_start"] = data_input_new["ts_start"].astype(int)
    data_input_new.sort_values(by='ts_start', inplace=True, ignore_index=True)
    # Set 'ts_start' as the index for 'data_input_new'
    data_input_new = data_input_new.set_index('ts_start', drop=False)
    # Reset index for 'df3' and set 'ts' as the index
    df3 = df3.reset_index(drop=True)
    df3 = df3.sort_values(by='ts', ignore_index=True).set_index('ts', drop=False)
    # Sort 'data_input_new' by index
    data_input_new = data_input_new.sort_index()
    # Convert 'ts_end' and 'ts_start' to int
    data_input_new['ts_end'] = data_input_new['ts_end'].astype(int)
    data_input_new['ts_start'] = data_input_new['ts_start'].astype(int)
    # Convert 'ts' column in 'df3' to int
    df3["ts"] = df3["ts"].astype(int)
    # Reset index for both DataFrames
    df3 = df3.reset_index(drop=True)
    data_input_new = data_input_new.reset_index(drop=True)
    # Sort DataFrames by 'ts' and 'ts_start' respectively
    df3.sort_values("ts", inplace=True)
    data_input_new.sort_values("ts_start", inplace=True)
    data_input_new.rename(columns={"hour": "hour_start"}, inplace=True)
    # Assuming ts_array, ts_start_array, and ts_end_array are Pandas Series
    ts_array = df3["ts"]
    ts_start_array = data_input_new["ts_start"]
    ts_end_array = data_input_new["ts_end"]

    # Initialize ts_start_result and ts_end_result columns with "NA"
    df3["ts_start"] = 0
    df3["ts_end"] = 0

    # Iterate over ts_start_array and ts_end_array
    for j in range(len(ts_start_array)):
        mask = (ts_array > ts_start_array[j]) & (ts_array < ts_end_array[j])
        df3.loc[mask, "ts_start"] = ts_start_array[j].astype(int)
        df3.loc[mask, "ts_end"] = ts_end_array[j].astype(int)

    result = pd.merge(df3[["ts", "ts_start", "ts_end", "Speed", "HVBattSOC", "OutsideAirTemp", "hour", "minute"]], data_input_new, left_on=["ts_start", "ts_end"], right_on=["ts_start", "ts_end"], how="left")

    result = result.loc[result["ts_start"] != 0]
    # select the columns that you need from the result dataframe

    # Renaming columns
    result.rename(columns={"destination_label": "Destination_label", "origin_label": "Origin_label"}, inplace=True)

    # Dropping rows with missing values in specific columns
    result.dropna(subset=["id", "Origin_label", "Destination_label"], inplace=True)

    # Resetting the index
    result.reset_index(drop=True, inplace=True)

    # Filter the DataFrame to exclude rows where "id" is equal to 0
    df5 = result[result["id"] != 0].reset_index(drop=True)

    # Assign charge levels to specific rows using loc and boolean indexing
    df5.loc[df5.index == 0, 'charge_level'] = "LEVEL_1"
    df5.loc[df5.index == 1, 'charge_level'] = "LEVEL_2"
    df5.loc[df5.index == 2, 'charge_level'] = "DC_FAST"

    data_1100_home = data_input_new[data_input_new["destination_label"] == "Home"]
    data_1100_home = data_1100_home[["id", "location[latitude]", "location[longitude]", "Lat", "Long"]]
    data_1100_home = data_1100_home.dropna(axis=0, subset=None, inplace=False)
    data_1100_home["location[latitude]"] = round(data_1100_home["location[latitude]"], 4)
    data_1100_home["location[longitude]"] = round(data_1100_home["location[longitude]"], 4)
    data_1100_home["Lat"] = round(data_1100_home["Lat"], 4)
    data_1100_home["Long"] = round(data_1100_home["Long"], 4)
    data_1100_work = data_input_new[data_input_new["destination_label"] == "Work"]
    data_1100_work = data_1100_work[["id", "location[latitude]", "location[longitude]", "Lat", "Long"]]
    data_1100_work = data_1100_work.fillna(0)
    data_1100_work["location[latitude]"] = round(data_1100_work["location[latitude]"], 4)
    data_1100_work["location[longitude]"] = round(data_1100_work["location[longitude]"], 4)
    data_1100_work["Lat"] = round(data_1100_work["Lat"], 4)
    data_1100_work["Long"] = round(data_1100_work["Long"], 4)
    data_1100_home = data_1100_home.reset_index(drop=True)
    data_1100_work = data_1100_work.reset_index(drop=True)
    home_coordinate = (data_1100_home.loc[0, "Lat"], data_1100_home.loc[0, "Long"])
    work_coordinate = (data_1100_work.loc[0, "Lat"], data_1100_work.loc[0, "Long"])
    df5["distance_home"] = df5.apply(lambda row: haversine(home_coordinate, (pd.to_numeric(row["Lat"]), pd.to_numeric(row['Long'])), unit=Unit.MILES), axis=1)
    df5["distance_work"] = df5.apply(lambda row: haversine(work_coordinate, (pd.to_numeric(row["Lat"]), pd.to_numeric(row['Long'])), unit=Unit.MILES), axis=1)
    # convert the column to a datetime type
    df5['hour_end1'] = df5["end_time_ (local)"].astype(str)
    df5['hour_end1'] = df5['hour_end1'].str.split(' ').str[0]
    df5['hour_end1'] = df5['hour_end1'].apply(lambda x: datetime.strptime(x, '%m/%d/%Y').date())
    df5['hour_end1'] = pd.to_datetime(df5['hour_end1'])

    df5['month'] = df5['hour_end1'].dt.month

    df5 = df5.reset_index(drop=True)
    df5 = df5.sort_values(by="id")
    df5 = df5.sort_values(by='ts')

    df5["ts_1"] = df5["ts"].shift(-1)
    df5["speed_1"] = df5["Speed"].shift(-1)
    df5["distance"] = ((df5["speed_1"] * 0.0002777778) * (df5["ts_1"] - df5["ts"]))
    df5.loc[df5["distance"] > 1, "distance"] = 0
    distance = data_input[["id", "distance", "year", "month", "day", "hour", "energy[charge_type][type]"]].copy()

    # Create a new column to count occurrences of non-'NA' values in "energy[charge_type][type]"
    distance["charge_type_count"] = (distance["energy[charge_type][type]"] != 'NA').cumsum()

    # Replace 'NA' values with the count
    distance["energy[charge_type][type]"] = np.where(distance["energy[charge_type][type]"] == 'NA', distance["charge_type_count"], distance["energy[charge_type][type]"])
    distance["distance_to_end_day"] = distance.groupby(["year", "month", "day"])["distance"].transform(lambda x: x.iloc[::-1].cumsum())
    distance["next_departure"] = distance["hour"].shift(-1)
    distance.rename(columns={"hour": "hour_start"}, inplace=True)
    distance["distance_to_next_charge"] = distance.groupby("charge_type_count")["distance"].transform(lambda x: x.iloc[::-1].cumsum())
    distance.rename(columns={"distance": "trip_distance"}, inplace=True)
    df5 = pd.merge(df5,  distance[["id", "energy[charge_type][type]", "distance_to_end_day", "next_departure", "distance_to_next_charge", 'trip_distance']], on="id", how="left")
    df5['cumulative_distance'] = df5.groupby('id')['trip_distance'].cumsum()

    df5 = df5.sort_values(by=["ts"])

    df6 = df5.copy()
    df6 = df6.loc[~df6["cumulative_distance"].isna()]
    df6["ts_extra"] = df6["ts"]
    nearest = 5
    df6["ts_extra"] = round(df6["ts_extra"] / nearest) * nearest
    df6["cumulative_distance"] = df6["cumulative_distance"].astype(int)

    df8 = df6.groupby(['id', 'day']).last().reset_index()
    df8 = df8.sort_values(by=["ts"])
    df8["count"] = range(0, len(df8))

    test = df8.groupby(['year', 'month', 'day', 'id', 'count', "cluster", "trip_distance", "duration_trip", "charge_level"])['Destination_label'].apply(lambda x: ' -> '.join(x.unique())).reset_index()
    test = test.sort_values(by=["count"])
    test = test.reset_index(drop=True)
    test["Origin_label"] = test["Destination_label"].shift(1)
    test.loc[test["cluster"] == -1, "cluster"] = 99
    test['new_Destination'] = test.apply(lambda row: f"{row['Destination_label']} {row['cluster']}" if row['Destination_label'] == 'Other' else row['Destination_label'], axis=1)
    data2 = data1.copy()
    data2 = data2[data2["vehicle_name"] == vehicle_name]
    data2 = pd.merge(data2, test[["id", "Destination_label", "Origin_label", "new_Destination"]], how="left", on="id")
    return df5, test, data2


def split_func(input_df):
    input_df["next_trip_start"] = input_df["start_time_ (local)"].shift(-1)
    input_df['start_time (local)'] = pd.to_datetime(input_df['start_time_ (local)'])
    input_df['next_trip_start'] = pd.to_datetime(input_df['next_trip_start'])
    input_df["parking_duration"] = (input_df["next_trip_start"] - input_df["start_time (local)"]).dt.total_seconds() / 60

    charging_sessions = input_df.loc[(input_df["energy[charge_type][type]"] != "NA") & ((input_df["Make"] == "Tesla") | (input_df["Make"] == "Chevrolet"))].copy()
    charging_sessions.loc[:, "duration_charging"] = charging_sessions["duration_charging"] / 60
    charging_sessions = charging_sessions.loc[charging_sessions["total_energy"] > 2]
    charging_sessions = charging_sessions.loc[~((charging_sessions["total_energy"] < 10) & (charging_sessions["duration_charging"] > 10 * 3600))]
    charging_sessions_tesla = charging_sessions.loc[charging_sessions["Make"] == "Tesla"]
    charging_sessions_Bolt = charging_sessions.loc[charging_sessions["Make"] == "Chevrolet"]

    return charging_sessions, charging_sessions_tesla, charging_sessions_Bolt


def data_departure(input_df):
    input_df = input_df[["month", "day_name", "hour", "battery[soc][start][trip]", "HVBattSOC", "OutsideAirTemp", "Lat", "Long", "Origin_label", "Destination_label", "energy[charge_type][type]_x", "distance", "next_departure"]]

    return input_df


def drawing_data(input_df):
    test = input_df.copy()

    group_test = test.groupby(['year', 'month', 'day'])

    # Group by ['year', 'month', 'day'] and aggregate statistics
    grouped_data1 = group_test.agg({
        'trip_distance': ['mean', 'std'],
        'duration_trip': ['mean', 'std']
    }).reset_index()

    # Rename the columns for clarity
    grouped_data1.columns = ['year', 'month', 'day', 'avg_trip_distance', 'std_trip_distance', 'avg_duration_trip', 'std_duration_trip']

    # Create a list of lists to store the grouped data
    grouped_data = []

    # Iterate through the groups and store them in a list
    for name, group in group_test:
        grouped_data.append(group['new_Destination'].tolist())

    # Flatten the list of lists and create a list of tuples with the sequences
    sequences = [tuple(sublist) for sublist in grouped_data]

    df_sequences = pd.DataFrame({'sequences': sequences})

    # Concatenate the DataFrames
    merged_data = pd.concat([grouped_data1, df_sequences], axis=1)

    # Get the unique sequences
    unique_sequences = merged_data['sequences'].unique()

    # Create a mapping of unique sequences to unique numbers
    sequence_to_number = {seq: i for i, seq in enumerate(unique_sequences)}
    # Get the unique sequences and their mapping
    merged_data['sequence_number'], unique_sequences = pd.factorize(merged_data['sequences'])

    # Count the occurrences of each sequence
    sequence_counts = Counter(sequences)

    # Define a custom sorting key that considers both the count and frequency of sequences
    def custom_sorting_key(item):
        sequence, count = item
        frequency = count / len(grouped_data)
        return (count, frequency)

    # Calculate the total number of sequences
    total_sequences = len(sequences)

    # Sort the sequence_counts based on the custom sorting key
    sorted_counts = sorted(sequence_counts.items(), key=custom_sorting_key, reverse=True)

    # Create a list to store the data
    data = []

    # Iterate through the sorted sequence counts and calculate frequency
    for sequence_destination, count in sorted_counts:
        frequency = (count / total_sequences) * 100
        data.append([sequence_destination, count, frequency])

    # Create a DataFrame
    final_sequence = pd.DataFrame(data, columns=["Sequence_destination", "Count", "Frequency"])

    final_sequence.Count.sum()
    final_sequence.Frequency.sum()

    # Filter sequences with a count larger than 5
    filtered_data = final_sequence[final_sequence["Count"] > 5]

    # Sum of counts for the sequences with count <= 5
    rest_count = final_sequence[final_sequence["Count"] <= 5]["Count"].sum()

    # Create a DataFrame for "The Rest" category
    rest_data = pd.DataFrame({"Sequence_destination": ["The Rest"], "Count": [rest_count]})

    # Add "The Rest" category to filtered_data
    filtered_data = pd.concat([filtered_data, rest_data], ignore_index=True)

    return filtered_data

def plot_py(input_df):
    filtered_data = input_df.copy()
    # Plot the pie chart
    fig, ax = plt.subplots(figsize=(21, 8))
    wedges, texts, autotexts = plt.pie(filtered_data["Count"], autopct='%1.1f%%', startangle=90, textprops={'color': 'black', 'fontsize': 14})

    # Calculate the percentages manually and add them as text labels outside the pie chart
    percentages = (filtered_data["Count"] / filtered_data["Count"].sum()) * 100
    labels = [f'{filtered_data.iloc[i]["Sequence_destination"]} ({percentages.iloc[i]:.1f}%)' for i in range(len(filtered_data))]

    # Set aspect ratio to be equal so the pie is drawn as a circle.
    ax.axis('equal')

    # Add a legend inside the pie chart with reduced bbox_to_anchor and increased font size
    plt.legend(labels, title="Sequences", loc="upper left", bbox_to_anchor=(0.7, 0.9), prop={'size': 12}, fontsize=16, )

    # Add a label to the graph with increased font size
    plt.figtext(0.5, 0.93, " Frequency of Daily Trip Sequence", fontsize=16, ha='center')

    plt.show()
    return plt.show()


def plot_autocorrelation1(df, lag_range, vehicle_name):

    df.year = df.year.astype(int)
    df.month = df.month.astype(int)
    df.day = df.day.astype(int)

    df = df.groupby(["year", 'month', 'day'])['trip_distance'].sum()
    # Initialize an empty list to store the autocorrelation results
    autocorr_results = []

    # Calculate autocorrelation for lags in the specified range
    for lag in range(2, lag_range):
        autocorr_value = df.autocorr(lag)
        autocorr_results.append({"Lag": lag, "Autocorrelation": autocorr_value})

    # Convert the list to a DataFrame
    autocorr_results_df = pd.DataFrame(autocorr_results)

    # Plot the results as a bar plot
    plt.figure(figsize=(10, 6))
    plt.bar(autocorr_results_df["Lag"], autocorr_results_df["Autocorrelation"])
    plt.xlabel("Lag")
    plt.ylabel("Autocorrelation")
    plt.title(f"Autocorrelation for Lags from 2 to {lag_range} - {vehicle_name}")
    plt.grid(True, axis="y")
    plt.xticks(range(lag_range))

    plt.show()

    # Find and print the largest autocorrelation value and its associated lag
    max_index = autocorr_results_df["Autocorrelation"].idxmax()
    max_autocorr = autocorr_results_df.loc[max_index, "Autocorrelation"]
    lag_with_max_autocorr = autocorr_results_df.loc[max_index, "Lag"]

    print(f"Largest Autocorrelation Value: {max_autocorr}")
    print(f"Associated Lag: {lag_with_max_autocorr}")
    return lag_with_max_autocorr


def plot_autocorrelation2(df, lag_range, vehicle_name):
    df.year = df.year.astype(int)
    df.month = df.month.astype(int)
    df.day = df.day.astype(int)
    # Convert the year, month, and day columns to datetime
    df['date'] = pd.to_datetime(df[['year', 'month', 'day']].astype(str).agg('-'.join, axis=1))

    # Create a date range covering the entire period
    start_date = df['date'].min()
    end_date = df['date'].max()
    date_range = pd.date_range(start=start_date, end=end_date)

    # Create a DataFrame from the date range
    date_range_df = pd.DataFrame({'date': date_range})

    # Merge your existing data with the date range
    merged_df = pd.merge(date_range_df, df, left_on='date', right_on='date', how='left')

    # Fill missing values with zeros
    merged_df.fillna(0, inplace=True)

    # Replace missing year, month, and day based on the dates we found
    merged_df['year'] = merged_df['date'].dt.year
    merged_df['month'] = merged_df['date'].dt.month
    merged_df['day'] = merged_df['date'].dt.day

    df = merged_df.groupby(["year", 'month', 'day'])['trip_distance'].sum()
    # Initialize an empty list to store the autocorrelation results
    autocorr_results = []

    # Calculate autocorrelation for lags in the specified range
    for lag in range(2, lag_range):
        autocorr_value = df.autocorr(lag)
        autocorr_results.append({"Lag": lag, "Autocorrelation": autocorr_value})

    # Convert the list to a DataFrame
    autocorr_results_df = pd.DataFrame(autocorr_results)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Assuming you have autocorr_results_df, lag_range, and vehicle_name defined
    ax.bar(autocorr_results_df["Lag"], autocorr_results_df["Autocorrelation"])
    ax.set_xlabel("Lag")
    ax.set_ylabel("Autocorrelation")
    ax.set_title(f"Autocorrelation for Lags from 2 to {lag_range} - {vehicle_name}")
    ax.grid(True, axis="y")
    ax.set_xticks(range(2, lag_range))

    # Set y-axis limits between -1 and 1
    ax.set_ylim(-1, 1)

    plt.show()


def plot_autocorrelation3(df, lag_with_max_autocorr):
    # Perform seasonal decomposition
    df_full_trips_daily = df.groupby(["year", 'month', 'day'])['trip_distance'].sum()
    result = sm.tsa.seasonal_decompose(df_full_trips_daily, model='additive', period=lag_with_max_autocorr.astype(int))

    # Access the trend, seasonal, and residual components
    trend = result.trend
    seasonal = result.seasonal
    residual = result.resid

    # Ensure the components are in Pandas Series
    trend = pd.Series(trend)
    seasonal = pd.Series(seasonal)
    residual = pd.Series(residual)

    # Create a DataFrame to store the components
    components_df = pd.DataFrame({
        "actual": df_full_trips_daily,
        "Trend": trend,
        "Seasonal": seasonal,
        "Residual": residual
    })
    components_df = components_df.reset_index(drop=True)
    # Plot the decomposed components with different colors
    fig, axs = plt.subplots(4, 1, figsize=(10, 6))
    colors = ['blue', 'black', 'red', 'darkorange']

    for i, component in enumerate(components_df.columns):
        ax = axs[i]
        ax.plot(components_df.index, components_df[component], label=component, color=colors[i])
        ax.set_title(component)
        ax.set_ylabel("mile")  # Add the y-axis title
        ax.legend()

    plt.tight_layout()
    plt.show()


def plot_autocorrelation_season(df, lag_range, vehicle_name):
    df.year = df.year.astype(int)
    df.month = df.month.astype(int)
    df.day = df.day.astype(int)

    # Convert the year, month, and day columns to datetime
    df['date'] = pd.to_datetime(df[['year', 'month', 'day']].astype(str).agg('-'.join, axis=1))

    # Create a date range covering the entire period
    start_date = df['date'].min()
    end_date = df['date'].max()
    date_range = pd.date_range(start=start_date, end=end_date)

    # Create a DataFrame from the date range
    date_range_df = pd.DataFrame({'date': date_range})

    # Merge your existing data with the date range
    merged_df = pd.merge(date_range_df, df, left_on='date', right_on='date', how='left')

    # Fill missing values with zeros
    merged_df.fillna(0, inplace=True)

    # Replace missing year, month, and day based on the dates we found
    merged_df['year'] = merged_df['date'].dt.year
    merged_df['month'] = merged_df['date'].dt.month
    merged_df['day'] = merged_df['date'].dt.day

    df = merged_df.groupby(["year", 'month', 'day'], as_index=False)['trip_distance'].sum()

    # Define seasons based on months
    spring = [3, 4, 5]  # March, April, May
    summer = [6, 7, 8]  # June, July, August
    fall = [9, 10, 11]  # September, October, November
    winter = [12, 1, 2]  # December, January, February
    # Calculate autocorrelations for each season
    seasons = {
        "Spring": spring,
        "Summer": summer,
        "Fall": fall,
        "Winter": winter,
    }
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 10))
    fig.suptitle(f"Autocorrelation for Lags from 2 to {lag_range} - {vehicle_name}")
    df_season = pd.DataFrame()
    for i, (season_name, season_months) in enumerate(seasons.items()):
        row, col = i // 2, i % 2
        ax = axes[row, col]

        # Filter data for the current season
        season_data = df[df['month'].isin(season_months)]

        # Group by year, month, and day, and calculate the sum of trip_distance
        season_data = season_data.groupby(["year", 'month', 'day'])['trip_distance'].sum()

        # Calculate autocorrelations for lags from 2 to 21
        autocorr_season = []
        for lag in range(2, lag_range):
            autocorr_value = season_data.autocorr(lag)
            autocorr_season.append({"Lag": lag, "Autocorrelation": autocorr_value})

        # Convert the list to a DataFrame
        autocorr_season_df = pd.DataFrame(autocorr_season)

        ax.bar(autocorr_season_df["Lag"], autocorr_season_df["Autocorrelation"])
        ax.set_title(f"{season_name} Season")
        ax.set_xlabel("Lag")
        ax.set_ylabel("Autocorrelation")
        ax.grid(True, axis="y")
        ax.set_xticks(range(2, lag_range))
        # Set y-axis limits between 1 and -1
        ax.set_ylim(-1, 1)

        autocorr_season_df["vehicle"] = vehicle_name
        autocorr_season_df["season_name"] = f"{season_name} Season"
        df_season = df_season.append(pd.DataFrame(autocorr_season_df))

    df_season.to_csv(f'{vehicle_name} season.csv', index=False)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()

    return autocorr_season_df["Autocorrelation"]


def plot_autocorrelation_month(df, lag_range, vehicle_name):

    df.year = df.year.astype(int)
    df.month = df.month.astype(int)
    df.day = df.day.astype(int)

    # Convert the year, month, and day columns to datetime
    df['date'] = pd.to_datetime(df[['year', 'month', 'day']].astype(str).agg('-'.join, axis=1))

    # Create a date range covering the entire period
    start_date = df['date'].min()
    end_date = df['date'].max()
    date_range = pd.date_range(start=start_date, end=end_date)

    # Create a DataFrame from the date range
    date_range_df = pd.DataFrame({'date': date_range})

    # Merge your existing data with the date range
    merged_df = pd.merge(date_range_df, df, left_on='date', right_on='date', how='left')

    # Fill missing values with zeros
    merged_df.fillna(0, inplace=True)

    # Replace missing year, month, and day based on the dates we found
    merged_df['year'] = merged_df['date'].dt.year
    merged_df['month'] = merged_df['date'].dt.month
    merged_df['day'] = merged_df['date'].dt.day

    df = merged_df.groupby(["year", 'month', 'day'], as_index=False)['trip_distance'].sum()

    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(18, 12))
    fig.suptitle("Autocorrelation by Month for Lags from 2 to 21")

    # Create a list of months to iterate through
    months = range(1, 13)
    df_month = pd.DataFrame()
    for i, month in enumerate(months):
        row, col = i // 4, i % 4
        ax = axes[row, col]

        # Filter data for the current month
        month_data = df[df['month'] == month]

        # Group by year, month, and day, and calculate the sum of trip_distance
        month_data = month_data.groupby(["year", 'month', 'day'])['trip_distance'].sum()

        # Calculate autocorrelations for lags from 2 to 21
        autocorr_month = []
        for lag in range(2, lag_range):
            autocorr_value = month_data.autocorr(lag)
            autocorr_month.append({"Lag": lag, "Autocorrelation": autocorr_value})

        # Convert the list to a DataFrame
        autocorr_month_df = pd.DataFrame(autocorr_month)

        ax.bar(autocorr_month_df["Lag"], autocorr_month_df["Autocorrelation"])
        ax.set_title(f"Month {month}")
        ax.set_xlabel("Lag")
        ax.set_ylabel("Autocorrelation")
        ax.grid(True, axis="y")
        ax.set_xticks(range(2, lag_range))
        # Set y-axis limits between 1 and -1
        ax.set_ylim(-1, 1)

        autocorr_month_df["vehicle"] = vehicle_name
        autocorr_month_df["month"] = f"Month {month}"
        df_month = df_month.append(pd.DataFrame(autocorr_month_df))

    df_month.to_csv(f'{vehicle_name} month.csv', index=False)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()

    return autocorr_month_df["Autocorrelation"]


def r_sq(df1, df2):
    # Calculate mean of actual values
    mean_actual = np.mean(df1)

    # Calculate the sum of squared differences between actual and predicted
    ss_residual = np.sum((df1 - df2) ** 2)

    # Calculate the total sum of squares
    ss_total = np.sum((df1 - mean_actual) ** 2)

    # Calculate R-squared
    r_squared = 1 - (ss_residual / ss_total)

    print(f'R-squared: {r_squared:.4f}')

    return r_squared

