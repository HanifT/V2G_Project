import pytz
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import math


def read_clean_kai():
    data = pd.read_csv("C:\\Users\\ht9\\Box\\Chapter2\\Kai\\us_unified_labeled_2401.csv", low_memory=False)
    data_vin = pd.read_csv("C:\\Users\\ht9\\Box\\Chapter2\\Kai\\vehicles_vin.csv")
    data = pd.merge(data, data_vin, how="left", on="vehicle_id")
    data = data.groupby(["vehicle_id", "created_at"]).tail(n=1)
    data = data[(data["activity"] != "driving") | ((data["activity"] == "driving") & (data["distance"] > 0))]
    # Assuming your column name is 'timestamp'
    data['created_at'] = pd.to_datetime(data['created_at'])  # Convert to datetime if not already

    # Filter dates after November 2021
    filtered_data = data[data['created_at'] >= '2021-11-01']

    # data = data[~data["battery_level_start"].isna()]
    data = filtered_data.sort_values(by=["vehicle_id", "created_at"])
    data.rename(columns={'created_at': 'start_time_trip_local'}, inplace=True)
    data["start_time_trip_local"] = data["local_time"]
    data = data.drop("local_time", axis=1)
    # Assuming "duration" is in minutes
    data["start_time_trip_local"] = pd.to_datetime(data["start_time_trip_local"])
    data["duration_timedate"] = pd.to_timedelta(data["duration"], unit='s')

    # Assuming "start_time_trip_local" is already in datetime format
    data["end_time_trip_local"] = data["start_time_trip_local"] + data["duration_timedate"]
    data = data.sort_values(["vehicle_id", "start_time_trip_local"]).reset_index(drop=True).reset_index(drop=False)

    # Identify consecutive driving events and assign a group identifier
    data['driving_group'] = (data['activity'] == 'driving').astype(int).groupby(data['vehicle_id']).cumsum()

    # Create an identifier for each event based on the driving group
    data['identifier'] = data['driving_group'].astype(str)

    # If 'charging' is followed by another 'charging', keep the same identifier
    mask_charging = (data['activity'] == 'charging') & (data['activity'].shift(-1) == 'charging')
    data['identifier'] = data['identifier'].where(~mask_charging, data['identifier'].shift())

    # Create a common index for consecutive charging events
    mask_common_index = (data['activity'] == 'charging') & (data['activity'].shift(-1) == 'charging')
    data['index'] = data['index'].where(~mask_common_index, data['index'].shift())

    # Drop the temporary 'driving_group' column
    data.drop(['driving_group'], axis=1, inplace=True)
    data["identifier"] = data["identifier"].astype(int)

    # Aggregate charging events within the same identifier
    aggregated_data = data.groupby(['identifier', 'activity', "vehicle_id"], as_index=False).agg({
        "start_time_trip_local": "first",
        "end_time_trip_local": "last",
        "battery_level_start": "first",
        "battery_level_end": "last",
        "distance": "first",
        "duration": 'sum',
        "Destination Label": 'first',
        "participant_id": 'first',
        "World Manufacturing Identifier": 'first',
        "Make/Line/Series": 'first',
        "Charger/Battery/Fuel Type": 'first',
        "tz": "first"

    })
    aggregated_data = aggregated_data.sort_values(["vehicle_id", "start_time_trip_local"])

    # Create a new DataFrame for charging events
    charging_data = aggregated_data[aggregated_data['activity'] == 'charging'].copy()
    driving_data = aggregated_data[aggregated_data['activity'] != 'charging'].copy()
    # Create a column for the previous driving event
    charging_data['previous_driving_event'] = charging_data['start_time_trip_local'].shift()

    # Filter out the first charging event for each vehicle (where previous_driving_event is NaN)
    charging_data = charging_data[charging_data['previous_driving_event'].notna()]

    # Select relevant columns for the final DataFrame
    final_columns = ['previous_driving_event', 'start_time_trip_local', 'end_time_trip_local', 'battery_level_start', 'battery_level_end']

    # Rename columns for clarity
    charging_data.rename(columns={'start_time_trip_local': 'charging_start_time', 'end_time_trip_local': 'charging_end_time',
                                  'battery_level_start': 'charging_soc_start', 'battery_level_end': 'charging_soc_end', 'duration': 'charging_duration'}, inplace=True)

    # Merge the charging_data DataFrame with the original data to get the details of the previous driving event
    result_data = pd.merge(driving_data, charging_data[["identifier", "vehicle_id", "charging_start_time", "charging_end_time", "charging_soc_start", "charging_soc_end"]], how='left', on=["identifier", "vehicle_id"])
    result_data["bat_cap"] = 90

    # Reset index for the final DataFrame
    result_data.reset_index(drop=True, inplace=True)

    return result_data


def extract_datetime_components(df):
    # Convert the timestamp_column to datetime format
    df['year'] = df['start_time_trip_local'].apply(lambda x: pd.to_datetime(x).year)
    df['month'] = df['start_time_trip_local'].apply(lambda x: pd.to_datetime(x).month)
    df['day'] = df['start_time_trip_local'].apply(lambda x: pd.to_datetime(x).day)
    df['hour'] = df['start_time_trip_local'].apply(lambda x: pd.to_datetime(x).hour)

    return df


def rename_col(df):
    df.rename(columns={'start_time_trip_local': 'start_time_local', 'end_time_trip_local': 'end_time_local', "vehicle_id": "vehicle_name",
                       "Make/Line/Series": "vehicle_model", 'battery_level_start': 'battery[soc][start][trip]', 'battery_level_end': 'battery[soc][end][trip]',
                       'charging_soc_start': 'battery[soc][start][charging]', 'charging_soc_end': 'battery[soc][end][charging]',
                       'duration': 'duration_trip'}, inplace=True)
    return df


def trip_summary_k(df):
    df = data.copy()
    data3 = df[["start_time_local", 'end_time_local', 'vehicle_name', "year", "month", "day", "hour", "duration_trip", "distance", "battery[soc][start][trip]", "battery[soc][end][trip]",]]
    data3 = df[["start_time_local", "end_time_local", "vehicle_name", "vehicle_model", "year", "month", "day", "hour", "duration_trip", "distance", "battery[soc][start][trip]",
               "battery[soc][end][trip]", "Lat", "Long", "destination_label", "origin_label", "energy[charge_type][type]", "battery[soc][start][charging]", "battery[soc][end][charging]", "start_time", "end_time", "duration_charging"]].copy()

    data3.loc[:, "next_departure_time"] = data3["start_time_local"].shift(-1)
    data3.loc[data3["next_departure_time"] < data3["end_time_charging"], "end_time_charging"] = data3["next_departure_time"]
    data3.loc[data3["end_time_charging"] < data3["start_time_charging"], "end_time_charging"] = data3["next_departure_time"]
    data3.loc[:, "parking_time"] = data3["next_departure_time"] - data3["end_time_local"]
    data3.loc[:, "parking_time_minute"] = data3["parking_time"].dt.total_seconds() / 60
    data3.loc[:, "duration_charging_min"] = data3.loc[:, "duration_charging"] / 60
    data3.loc[data3["duration_charging_min"] > data3["parking_time_minute"], "parking_time_minute"] = data3["duration_charging_min"]
    data3["bat_cap"] = data3['vehicle_model'].apply(extract_last_chars)

    return data3


def draw_parking(df):
    # Calculate average time spent at each location
    average_duration = df.groupby(['origin_label', 'destination_label'])['parking_time_minute'].mean().reset_index(name='Average Parking Time')
    frequency = df.groupby(['origin_label', 'destination_label']).size().reset_index(name='Frequency')
    bubble_data = pd.merge(average_duration, frequency, on=['origin_label', 'destination_label'])
    bubble_data['origin_label'] = pd.Categorical(bubble_data['origin_label'], categories=bubble_data['origin_label'].unique(), ordered=True)
    bubble_data['destination_label'] = pd.Categorical(bubble_data['destination_label'], categories=bubble_data['destination_label'].unique(), ordered=True)

    # Set the size limits based on the frequency values
    size_min, size_max = bubble_data['Frequency'].min() * 0.7, bubble_data['Frequency'].max() * 0.7

    # Create a bubble chart using seaborn and matplotlib
    plt.figure(figsize=(10, 6))
    scatterplot = sns.scatterplot(
        data=bubble_data,
        x='origin_label',
        y='destination_label',
        size='Frequency',
        hue='Average Parking Time',
        sizes=(size_min, size_max),
        palette='viridis'  # Set the color palette to viridis, you can choose any other color map
    )

    # Reverse the order of the y-axis
    plt.gca().invert_yaxis()

    # Adjust the space between the first ticks and the origin on both axes
    plt.margins(x=0.3, y=0.3)

    # Automatically adjust the scale considering the margins
    plt.autoscale()

    # Customize the layout
    plt.title('Average Parking Time and Frequency', fontsize=22)
    plt.xlabel('Origin', fontsize=18)
    plt.ylabel('Destination', fontsize=18)

    # Increase the font size of the x-axis ticks
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    scatterplot.get_legend().remove()
    # Add color bar to the figure
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=bubble_data['Average Parking Time'].min(), vmax=bubble_data['Average Parking Time'].max()))
    sm.set_array([])  # You need to set an empty array for the ScalarMappable
    cbar = plt.colorbar(sm, ax=scatterplot.axes)
    cbar.set_label('Average Parking Time (min)', fontsize=16)
    plt.savefig('bubble_chart_with_parking.png', bbox_inches='tight')
    plt.show()


def draw_parking(df):
    # Calculate average time spent at each location
    average_duration = df.groupby(['origin_label', 'destination_label'])['minrange'].mean().reset_index(name='Average Minimum Range')
    frequency = df.groupby(['origin_label', 'destination_label']).size().reset_index(name='Frequency')
    bubble_data = pd.merge(average_duration, frequency, on=['origin_label', 'destination_label'])
    bubble_data['origin_label'] = pd.Categorical(bubble_data['origin_label'], categories=bubble_data['origin_label'].unique(), ordered=True)
    bubble_data['destination_label'] = pd.Categorical(bubble_data['destination_label'], categories=bubble_data['destination_label'].unique(), ordered=True)

    # Set the size limits based on the frequency values
    size_min, size_max = bubble_data['Frequency'].min() * 0.7, bubble_data['Frequency'].max() * 0.7

    # Create a bubble chart using seaborn and matplotlib
    plt.figure(figsize=(10, 6))
    scatterplot = sns.scatterplot(
        data=bubble_data,
        x='origin_label',
        y='destination_label',
        size='Frequency',
        hue='Average Minimum Range',
        sizes=(size_min, size_max),
        palette='viridis'  # Set the color palette to viridis, you can choose any other color map
    )

    # Reverse the order of the y-axis
    plt.gca().invert_yaxis()

    # Adjust the space between the first ticks and the origin on both axes
    plt.margins(x=0.3, y=0.3)

    # Automatically adjust the scale considering the margins
    plt.autoscale()

    # Customize the layout
    plt.title('Average Minimum Range and Frequency', fontsize=22)
    plt.xlabel('Origin', fontsize=18)
    plt.ylabel('Destination', fontsize=18)

    # Increase the font size of the x-axis ticks
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    scatterplot.get_legend().remove()
    # Add color bar to the figure
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=bubble_data['Average Minimum Range'].min(), vmax=bubble_data['Average Minimum Range'].max()))
    sm.set_array([])  # You need to set an empty array for the ScalarMappable
    cbar = plt.colorbar(sm, ax=scatterplot.axes)
    cbar.set_label('Average Minimum Range (mi)', fontsize=16)
    plt.savefig('bubble_chart_with_range.png', bbox_inches='tight')
    plt.show()


def draw_charging(df):
    # Calculate average time spent at each location
    average_duration = df.groupby(['origin_label', 'destination_label'])['duration_charging'].mean().reset_index(name='Average Charging Time')
    frequency = df.groupby(['origin_label', 'destination_label']).size().reset_index(name='Frequency')
    bubble_data = pd.merge(average_duration, frequency, on=['origin_label', 'destination_label'])
    bubble_data['origin_label'] = pd.Categorical(bubble_data['origin_label'], categories=bubble_data['origin_label'].unique(), ordered=True)
    bubble_data['destination_label'] = pd.Categorical(bubble_data['destination_label'], categories=bubble_data['destination_label'].unique(), ordered=True)

    # Set the size limits based on the frequency values
    size_min, size_max = bubble_data['Frequency'].min() * 1, bubble_data['Frequency'].max() * 1

    # Create a bubble chart using seaborn and matplotlib
    plt.figure(figsize=(10, 6))
    scatterplot = sns.scatterplot(
        data=bubble_data,
        x='origin_label',
        y='destination_label',
        size='Frequency',
        hue='Average Charging Time',
        sizes=(size_min, size_max),
        legend='brief',
        palette='viridis'  # Set the color palette to viridis, you can choose any other color map
    )

    # Reverse the order of the y-axis
    plt.gca().invert_yaxis()

    # Adjust the space between the first ticks and the origin on both axes
    plt.margins(x=0.3, y=0.3)

    # Automatically adjust the scale considering the margins
    plt.autoscale()

    # Customize the layout
    plt.title('Average Charging Time and Frequency', fontsize=22)
    plt.xlabel('Origin', fontsize=18)
    plt.ylabel('Destination', fontsize=18)

    # Increase the font size of the x-axis ticks
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    # Create a second legend for hue with reduced size
    hue_legend = plt.legend(bbox_to_anchor=(1, 1.05), loc='upper left', prop={'size': 35}, ncol=2)

    # Decrease the font size of the legend items
    for text in hue_legend.get_texts():
        text.set_fontsize(20)

    # Find the legend handles associated with 'Average Parking Time' and 'Frequency'
    handles = scatterplot.legend_.legendHandles

    # Increase the size of the color bubble for the 'Average Parking Time' legend
    for lh in list(range(1, 7)):
        avg_parking_time_handle = handles[lh]  # Change index if needed
        # Increase the size of the color bubble for the 'Average Parking Time' legend
        avg_parking_time_handle.set_sizes([500])  # Adjust the size as needed

    # Set a different background color for the legend box
    hue_legend.get_frame().set_facecolor('#f0f0f0')  # Set the desired color

    # Save the plot as an image file (e.g., PNG)
    plt.savefig('bubble_chart_with_charging.png', bbox_inches='tight')
    plt.show()


def draw_parking_boxplots(df):
    # Assuming 'final_dataframes' is your DataFrame
    df = df[df["destination_label"] != "Other"]
    df = df[~((df["destination_label"] == "Work") & (df["origin_label"] == "Work"))]
    # Calculate average time spent at each location
    df["box"] = df["origin_label"] + "-" + df["destination_label"]

    df = df[df["parking_time_minute"] < 5000]

    # Calculate average time spent at each location
    average_duration = df.groupby(['box'])['parking_time_minute'].mean().reset_index(name='Average Parking Time')

    # Calculate average SOC before parking
    average_soc = df.groupby(['box'])['battery[soc][start][trip]'].mean().reset_index(name='Average SOC before parking')

    # Merge the two dataframes
    average = pd.merge(average_soc, average_duration, how="left", on="box")

    # Set up the custom color dictionary
    custom_colors = {'Home-Work': "#04e762", 'Work-Home': '#f77f00', 'Other-Home': '#00a1e4', 'Other-Work': '#dc0073', 'Home-Home': '#d62828'}

    # Set up the box plot
    fig, ax1 = plt.subplots(figsize=(12, 8))

    # Ensure consistent order for the box plot
    box_order = average.sort_values('Average SOC before parking')['box']

    # Use hue to assign colors based on 'box' category
    sns.boxplot(data=df, x='box', y='battery[soc][start][trip]', order=box_order, ax=ax1, palette=custom_colors)

    # Set labels and title for the first y-axis with larger font size
    ax1.set_ylabel('SOC %', fontsize=20)
    ax1.set_xlabel('Origin-Destination', fontsize=20)
    ax1.set_title('Parking Time and SOC for Different Origin-Destination Pairs', fontsize=22)

    # Create a second y-axis
    ax2 = ax1.twinx()

    # Set up the box plot for SOC

    # Adding average lines for parking time
    legend_handles1 = []  # Collect handles for the first legend
    for box in box_order:
        avg = average_duration.loc[average_duration['box'] == box, 'Average Parking Time'].values[0]
        line = ax2.axhline(avg, color=custom_colors[box], linestyle='dashed', linewidth=2, label=f'Avg {box} Parking Time: {avg:.2f} mins')
        legend_handles1.append(line)

    # Set labels and title for the second y-axis with larger font size
    ax2.set_ylabel('Parking Time (minutes)', fontsize=20)
    # Increase tick font size
    ax1.tick_params(axis='both', labelsize=18)
    ax2.tick_params(axis='both', labelsize=18)

    # Show the plot
    plt.savefig('soc.png', bbox_inches='tight')
    plt.show()


def soc_next(df):
    # df = df_full_trips_short.copy()
    df = df.reset_index(drop=True)

    mask = (df["battery[soc][start][trip]"] - df["battery[soc][end][trip]"] < 0)
    # Update the values where the condition is true
    df.loc[mask, 'battery[soc][end][trip]'] = df.loc[mask, 'battery[soc][start][trip]'] - ((0.28 * df.loc[mask, 'distance']) * (100 / 66))

    df.loc[((~df["battery[soc][start][charging]"].isna()) & (df["parking_time_minute"] >= 60.1) & (df["battery[soc][end][trip]"] + 10 < df["battery[soc][start][trip]"].shift(-1))), "energy[charge_type][type]"] = "LEVEL_2/1"
    df.loc[((~df["battery[soc][start][charging]"].isna()) & (df["parking_time_minute"] < 60.1) & (df["battery[soc][end][trip]"] + 10 < df["battery[soc][start][trip]"].shift(-1))), "energy[charge_type][type]"] = "DC_FAST"

    df.loc[(df["battery[soc][end][trip]"] + 10 < df["battery[soc][start][trip]"].shift(-1)) &
           (df["parking_time_minute"] >= 60.1), "energy[charge_type][type]"] = "LEVEL_2/1"

    df.loc[(df["battery[soc][end][trip]"] + 10 < df["battery[soc][start][trip]"].shift(-1)) &
           (df["parking_time_minute"] < 60.1), "energy[charge_type][type]"] = "DC_FAST"

    mask = (df["energy[charge_type][type]"] == "DC_FAST") & (df["destination_label"] == "Home")
    # Update the values where the condition is true
    df.loc[mask, 'destination_label'] = "Other"

    df["origin_label"] = df["destination_label"].shift(1)

    # Check if battery[soc][end][trip] is nan and battery[soc][end][charging] is not nan
    mask = (df['battery[soc][end][charging]'].isna()) & (df['energy[charge_type][type]'] != "NA")
    # Update the values where the condition is true
    df.loc[mask, 'battery[soc][start][charging]'] = df['battery[soc][end][trip]']
    df.loc[mask, 'battery[soc][end][charging]'] = df['battery[soc][start][trip]'].shift(-1)[mask]

    # Check if battery[soc][end][trip] is nan and battery[soc][end][charging] is not nan
    mask = (df['battery[soc][end][trip]'].isna()) & (~df['battery[soc][start][trip]'].isna())
    # Update the values where the condition is true
    df.loc[mask, 'battery[soc][end][trip]'] = df['battery[soc][start][trip]'].shift(-1)[mask]

    # Check if battery[soc][start][trip] is nan and battery[soc][end][charging] is not nan for the previous row
    mask = (df['battery[soc][start][trip]'].isna()) & (~df['battery[soc][end][charging]'].shift(1).isna())
    # Update the values where the condition is true
    df.loc[mask, 'battery[soc][start][trip]'] = df['battery[soc][end][charging]'].shift(1)[mask]

    # Check if battery[soc][end][trip] is nan and battery[soc][end][charging] is not nan
    mask = (df['battery[soc][start][trip]'].isna()) & (df['battery[soc][end][charging]'].shift(1).isna())
    # Update the values where the condition is true
    df.loc[mask, 'battery[soc][start][trip]'] = df.loc[mask, 'battery[soc][end][trip]'].shift(1)

    # Check if battery[soc][end][trip] is nan and battery[soc][end][charging] is not nan
    mask = (df['battery[soc][end][trip]'].isna()) & (~df['battery[soc][start][charging]'].isna())
    # Update the values where the condition is true
    df.loc[mask, 'battery[soc][end][trip]'] = df.loc[mask, 'battery[soc][start][charging]']

    # Check if battery[soc][start][trip] is nan and battery[soc][end][trip] is nan and before and after is not nan
    mask = (df['battery[soc][start][trip]'].isna()) & (df['battery[soc][end][trip]'].isna()) & (~df['battery[soc][start][trip]'].shift(1).isna())
    # Update the values where the condition is true
    df.loc[mask, 'battery[soc][start][trip]'] = df['battery[soc][end][trip]'].shift(1)[mask]

    # Check if battery[soc][start][trip] is nan and battery[soc][end][trip] is nan and before and after is not nan
    mask = (df['battery[soc][start][trip]'].isna()) & (df['battery[soc][end][trip]'].isna()) & (~df['battery[soc][end][trip]'].shift(-1).isna())
    # Update the values where the condition is true
    df.loc[mask, 'battery[soc][end][trip]'] = df['battery[soc][start][trip]'].shift(-1)[mask]

    # Check if battery[soc][start][trip] is nan and battery[soc][end][trip] is not nan
    mask = (df['battery[soc][start][trip]'].isna()) & (~df['battery[soc][end][trip]'].isna())
    # Update the values where the condition is true
    df.loc[mask, 'battery[soc][start][trip]'] = df.loc[mask, 'battery[soc][end][trip]'] + ((0.28 * df.loc[mask, 'distance']) * (100 / df["bat_cap"]))

    # Check if battery[soc][start][trip] is not nan and battery[soc][end][trip] is nan and start charging is nan
    mask = (~df['battery[soc][start][trip]'].isna()) & (df['battery[soc][end][trip]'].isna()) & (df['battery[soc][start][charging]'].isna())
    # Update the values where the condition is true
    df.loc[mask, 'battery[soc][end][trip]'] = df.loc[mask, 'battery[soc][start][trip]'] - ((0.28 * df.loc[mask, 'distance']) * (100 / df["bat_cap"]))

    # Check if battery[soc][start][trip] is nan and battery[soc][end][trip] is nan and before and after is not nan
    mask = (df['battery[soc][start][trip]'].isna()) & (df['battery[soc][end][trip]'].isna()) & (~df['battery[soc][start][trip]'].shift(1).isna()) & (~df['battery[soc][end][trip]'].shift(-1).isna())
    # Update the values where the condition is true
    df.loc[mask, 'battery[soc][start][trip]'] = df['battery[soc][end][trip]'].shift(1)[mask]
    df.loc[mask, 'battery[soc][end][trip]'] = df['battery[soc][start][trip]'].shift(-1)[mask]

    df = df.dropna(subset=["battery[soc][start][trip]", "battery[soc][end][trip]"])
    df = df.copy()
    df["SOC_Diff"] = df["battery[soc][start][trip]"] - df["battery[soc][end][trip]"]

    # Check if battery[soc][start][trip] is not nan and battery[soc][end][trip] is nan and start charging is nan
    mask = (df['SOC_Diff'] < 0)
    # Update the values where the condition is true
    df.loc[mask, 'battery[soc][end][trip]'] = df.loc[mask, 'battery[soc][start][trip]'] - ((0.28 * df.loc[mask, 'distance']) * (100 / df["bat_cap"]))

    df["SOC_Diff"] = df["battery[soc][start][trip]"] - df["battery[soc][end][trip]"]
    df["SOC_next_trip"] = df["SOC_Diff"].shift(-1)
    df['charge_type'] = df.groupby(["battery[soc][start][trip]", 'energy[charge_type][type]'])['energy[charge_type][type]'].head(1)
    df.loc[df['charge_type'].isna(), 'charge_type'] = 'NA'
    df.loc[(df['energy[charge_type][type]'] != "NA") & (df['charge_type'] == "NA"), 'charge_type'] = df.loc[(df['energy[charge_type][type]'] != "NA") & (df['charge_type'] == "NA"), 'energy[charge_type][type]']

    df["charge_type_count"] = (df["charge_type"] != 'NA').cumsum().shift(1).fillna(0)
    df["SOC_need_next_charge"] = df.groupby("charge_type_count")["SOC_Diff"].transform(lambda x: x[::-1].cumsum()[::-1]).shift(-1)
    df = df.iloc[:-1]

    return df


def charging_selection(df):
    # Filter rows where charging duration is not NaN
    final_df_charging = df.loc[~df["duration_charging"].isna()].copy()

    # Calculate minimum range for different scenarios
    final_df_charging["minrange"] = (final_df_charging["bat_cap"] * (final_df_charging["battery[soc][end][charging]"] / 100)) / 0.28
    final_df_charging["minrange_need"] = (final_df_charging["bat_cap"] * (final_df_charging["SOC_next_trip"] / 100)) / 0.28
    final_df_charging["minrange_need_nextc"] = (final_df_charging["bat_cap"] * (final_df_charging["SOC_need_next_charge"] / 100)) / 0.28

    return final_df_charging


def charging_speed(df):
    df["charging_speed"] = ((((df["battery[soc][end][charging]"] - df["battery[soc][start][charging]"]) / 100) * df["bat_cap"]) /
                            (df["duration_charging_min"] / 60))

    df.loc[df["charging_speed"] <= 1.6, "charge_type"] = "LEVEL_1"
    df.loc[(df["charging_speed"] > 1.6) & (df["charging_speed"] < 21), "charge_type"] = "LEVEL_2"
    df.loc[df["charging_speed"] >= 21, "charge_type"] = "DC_FAST"
    return df


def range_indicator(df):
    # next trip fail indicator
    df.loc[:, "next_trip_fail"] = df.loc[:, "minrange"] < df.loc[:, "minrange_need"]

    # next charging sessions fail indicator
    df.loc[:, "next_c_fail"] = df.loc[:, "minrange"] < df.loc[:, "minrange_need_nextc"]
    return df


def v2g_draw(df):
    # calculating the V2G time which is the difference between departure and end of charging
    # Convert 'end_time_charging' to datetime format
    df["end_time_charging"] = pd.to_datetime(df["end_time_charging"], errors='coerce', format='%Y-%m-%d %H:%M:%S%z')
    df["next_departure_time"] = pd.to_datetime(df["next_departure_time"], errors='coerce', format='%Y-%m-%d %H:%M:%S%z')

    # Set the timezone to PST
    timezone = pytz.timezone('US/Pacific')

    # Localize the datetime to PST timezone
    df["end_time_charging"] = df["end_time_charging"].apply(lambda x: x.astimezone(timezone) if pd.notnull(x) else x)
    df["next_departure_time"] = df["next_departure_time"].apply(lambda x: x.astimezone(timezone) if pd.notnull(x) else x)

    # Calculate the V2G time
    df["V2G_time_min"] = df["next_departure_time"] - df["end_time_charging"]
    df["V2G_time_min"] = df["V2G_time_min"].dt.total_seconds() / 60
    # Combine origin and destination into a new column "trip"
    df["trip"] = df["origin_label"] + " to " + df["destination_label"]

    # Group by the combined "trip" column and create a histogram for the "V2G_time" column
    # Filter the DataFrame for V2G_time less than 10000
    filtered_df = df[df["V2G_time_min"] < 10000]

    # Combine origin and destination into a new column "trip" using .loc
    filtered_df.loc[:, "trip"] = filtered_df["origin_label"] + " to " + filtered_df["destination_label"]

    # Set up a 3x3 grid
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(10, 10))

    # Set a color palette for better distinction
    colors = sns.color_palette("husl", n_colors=len(filtered_df["trip"].unique()))

    # Define a common y-limit for all subplots
    common_x_limit = filtered_df.groupby("trip")["V2G_time_min"].max().max()  # Adjust if needed

    # Group by the combined "trip" column and create a histogram for each group in the grid
    for i, (trip, group) in enumerate(filtered_df.groupby("trip")["V2G_time_min"]):
        row, col = divmod(i, 3)
        ax = axes[row, col]
        ax.hist(group, bins=20, alpha=0.5, label=trip, color=colors[i])
        ax.set_title(trip)
        ax.set_xlabel("V2G Time (min) \n Parking duration - Charging duration")
        ax.set_ylabel("Frequency")
        ax.set_xlim(0, common_x_limit)  # Set a common y-limit for all subplots

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()

    # Assuming final_dataframes_charging is your DataFrame
    filtered_df = df.loc[(df["V2G_time_min"] < 10000) & (df["V2G_time_min"] > 30)]

    # Combine origin and destination into a new column "trip" using .loc
    filtered_df.loc[:, "trip"] = filtered_df["origin_label"] + " to " + filtered_df["destination_label"]

    # Set up a 3x3 grid
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(10, 10))

    # Set a color palette for better distinction
    colors = sns.color_palette("husl", n_colors=len(filtered_df["trip"].unique()))

    # Define a common y-limit for all subplots
    common_x_limit = filtered_df.groupby("trip")["V2G_time_min"].max().max()  # Adjust if needed

    # Group by the combined "trip" column and create a histogram for each group in the grid
    for i, (trip, group) in enumerate(filtered_df.groupby("trip")["V2G_time_min"]):
        row, col = divmod(i, 3)
        ax = axes[row, col]
        ax.hist(group, bins=20, alpha=0.5, label=trip, color=colors[i])
        ax.set_title(trip)
        ax.set_xlabel("V2G Time (min) \n Parking duration - Charging duration")
        ax.set_ylabel("Frequency")
        ax.set_xlim(0, common_x_limit)  # Set a common y-limit for all subplots

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()

    return df


def V2G_cap_ch_r(df):
    # current speed
    df["V2G_SOC_half"] = (((df["battery[soc][end][charging]"] / 100) * df["bat_cap"]) - ((df["V2G_time_min"] / 60) / 2) * df["charging_speed"]) / (df["bat_cap"]) * 100
    df.loc[df["V2G_SOC_half"] < 0, "V2G_SOC_half"] = 0
    df["V2G_cap"] = (abs(df["V2G_SOC_half"]-df["battery[soc][end][charging]"])/100)*df["bat_cap"]
    # with Level 2

    # Assuming df is your DataFrame
    df["V2G_SOC_half_6k"] = (((df["battery[soc][end][charging]"] / 100) * df["bat_cap"]) - ((df["V2G_time_min"]/60)/2) * np.maximum(df["charging_speed"], 6.6)) / (df["bat_cap"]) * 100
    df.loc[df["V2G_SOC_half_6k"] < 0, "V2G_SOC_half_6k"] = 0
    df["V2G_cap_6k"] = (abs(df["V2G_SOC_half_6k"]-df["battery[soc][end][charging]"]) / 100) * df["bat_cap"]

    # Assuming df is your DataFrame
    df["V2G_SOC_half_19k"] = (((df["battery[soc][end][charging]"] / 100) * df["bat_cap"]) - ((df["V2G_time_min"]/60)/2) * np.maximum(df["charging_speed"], 19)) / (df["bat_cap"]) * 100
    df.loc[df["V2G_SOC_half_19k"] < 0, "V2G_SOC_half_19k"] = 0
    df["V2G_cap_19k"] = (abs(df["V2G_SOC_half_19k"] - (df["battery[soc][end][charging]"])) / 100) * df["bat_cap"]

    return df


def V2G_cap_soc_r5(df):
    df1 = df.copy()
    df1["battery[soc][end][charging]"] = df1["battery[soc][end][charging]"] - 5
    df1 = charging_selection(df1)
    df1 = range_indicator(df1)
    df1["battery[soc][end][charging]"] = df1["battery[soc][end][charging]"] + 5

    # current speed
    df1["V2G_SOC_half"] = (((df1["battery[soc][end][charging]"] / 100) * df1["bat_cap"]) - ((df1["V2G_time_min"] / 60) / 2) * df1["charging_speed"]) / (df1["bat_cap"]) * 100
    df1.loc[df1["V2G_SOC_half"] < 0, "V2G_SOC_half"] = 0
    df1["V2G_cap"] = (abs(df1["V2G_SOC_half"] - (df1["battery[soc][end][charging]"] + 5)) / 100) * df1["bat_cap"]

    # with Level 2

    # Assuming df1 is your DataFrame
    df1["V2G_SOC_half_6k"] = (((df1["battery[soc][end][charging]"] / 100) * df1["bat_cap"]) - ((df1["V2G_time_min"] / 60) / 2) * np.maximum(df1["charging_speed"], 6.6)) / (df1["bat_cap"])*100
    df1.loc[df1["V2G_SOC_half_6k"] < 0, "V2G_SOC_half_6k"] = 0
    df1["V2G_cap_6k"] = (abs(df1["V2G_SOC_half_6k"] - (df1["battery[soc][end][charging]"] + 5)) / 100) * df1["bat_cap"]

    # Assuming df1 is your DataFrame
    df1["V2G_SOC_half_19k"] = (((df1["battery[soc][end][charging]"] / 100) * df1["bat_cap"]) - ((df1["V2G_time_min"] / 60) / 2) * np.maximum(df1["charging_speed"], 19)) / (df1["bat_cap"]) * 100
    df1.loc[df1["V2G_SOC_half_19k"] < 0, "V2G_SOC_half_19k"] = 0
    df1["V2G_cap_19k"] = (abs(df1["V2G_SOC_half_19k"] - (df1["battery[soc][end][charging]"] + 5)) / 100) * df1["bat_cap"]

    return df1


def V2G_cap_soc_r10(df):
    df1 = df.copy()
    df1["battery[soc][end][charging]"] = df1["battery[soc][end][charging]"] - 10
    df1 = charging_selection(df1)
    df1 = range_indicator(df1)
    df1["battery[soc][end][charging]"] = df1["battery[soc][end][charging]"] + 10

    # current speed
    df1["V2G_SOC_half"] = (((df1["battery[soc][end][charging]"] / 100) * df1["bat_cap"]) - ((df1["V2G_time_min"] / 60) / 2) * df1["charging_speed"]) / (df1["bat_cap"]) * 100
    df1.loc[df1["V2G_SOC_half"] < 0, "V2G_SOC_half"] = 0
    df1["V2G_cap"] = (abs(df1["V2G_SOC_half"] - (df1["battery[soc][end][charging]"] + 10)) / 100) * df1["bat_cap"]

    # with Level 2

    # Assuming df1 is your DataFrame
    df1["V2G_SOC_half_6k"] = (((df1["battery[soc][end][charging]"] / 100) * df1["bat_cap"]) - ((df1["V2G_time_min"] / 60) / 2) * np.maximum(df1["charging_speed"], 6.6)) / (df1["bat_cap"]) * 100
    df1.loc[df1["V2G_SOC_half_6k"] < 0, "V2G_SOC_half_6k"] = 0
    df1["V2G_cap_6k"] = (abs(df1["V2G_SOC_half_6k"] - (df1["battery[soc][end][charging]"] + 10)) / 100) * df1["bat_cap"]

    # Assuming df1 is your DataFrame
    df1["V2G_SOC_half_19k"] = (((df1["battery[soc][end][charging]"] / 100) * df1["bat_cap"]) - ((df1["V2G_time_min"] / 60) / 2) * np.maximum(df1["charging_speed"], 19)) / (df1["bat_cap"]) * 100
    df1.loc[df1["V2G_SOC_half_19k"] < 0, "V2G_SOC_half_19k"] = 0
    df1["V2G_cap_19k"] = (abs(df1["V2G_SOC_half_19k"] - (df1["battery[soc][end][charging]"] + 10)) / 100) * df1["bat_cap"]

    return df1


def storage_cap(df):

    V2G_hourly = pd.DataFrame(index=df.index, columns=range(24))
    V2G_hourly_existing_speed = V2G_hourly.fillna(0)
    V2G_hourly_6 = V2G_hourly_existing_speed.copy()
    V2G_hourly_19 = V2G_hourly_existing_speed.copy()

    for i in df.index:
        start_hour = df.loc[i, "end_time_charging"].hour
        discharging_speed = df.loc[i, "charging_speed"]
        total_capacity = df.loc[i, "V2G_cap"]

        current_hour = start_hour
        remaining_capacity = total_capacity
        while remaining_capacity > 0:
            discharge_amount = min(discharging_speed, remaining_capacity)
            V2G_hourly_existing_speed.loc[i, current_hour] += discharge_amount
            remaining_capacity -= discharge_amount

            if current_hour < 23:
                current_hour += 1
            else:
                current_hour = 0

    for i in df.index:
        start_hour = df.loc[i, "end_time_charging"].hour
        discharging_speed = 6.6
        total_capacity = df.loc[i, "V2G_cap_6k"]

        current_hour = start_hour
        remaining_capacity = total_capacity
        while remaining_capacity > 0:
            discharge_amount = min(discharging_speed, remaining_capacity)
            V2G_hourly_6.loc[i, current_hour] += discharge_amount
            remaining_capacity -= discharge_amount

            if current_hour < 23:
                current_hour += 1
            else:
                current_hour = 0

    for i in df.index:
        start_hour = df.loc[i, "end_time_charging"].hour
        discharging_speed = 19
        total_capacity = df.loc[i, "V2G_cap_19k"]

        current_hour = start_hour
        remaining_capacity = total_capacity
        while remaining_capacity > 0:
            discharge_amount = min(discharging_speed, remaining_capacity)
            V2G_hourly_19.loc[i, current_hour] += discharge_amount
            remaining_capacity -= discharge_amount

            if current_hour < 23:
                current_hour += 1
            else:
                current_hour = 0

    V2G_hourly_existing_speed = pd.merge(df[["month", "day"]], V2G_hourly_existing_speed, how="left", left_index=True, right_index=True)
    V2G_hourly_existing_speed_sum = V2G_hourly_existing_speed.groupby(["month", "day"]).sum()

    V2G_hourly_6 = pd.merge(df[["month", "day"]], V2G_hourly_6, how="left", left_index=True, right_index=True)
    V2G_hourly_6_sum = V2G_hourly_6.groupby(["month", "day"]).sum()

    V2G_hourly_19 = pd.merge(df[["month", "day"]], V2G_hourly_19, how="left", left_index=True, right_index=True)
    V2G_hourly_19_sum = V2G_hourly_19.groupby(["month", "day"]).sum()

    # V2G_hourly_existing_speed_sum = V2G_hourly_existing_speed.sum()
    # V2G_hourly_6_sum = V2G_hourly_6.sum()
    # V2G_hourly_19_sum = V2G_hourly_19.sum()

    return V2G_hourly_existing_speed, V2G_hourly_6, V2G_hourly_19, V2G_hourly_existing_speed_sum, V2G_hourly_6_sum, V2G_hourly_19_sum


def v2g_cap_plot(df1, df2, df3):
    # Plot the lines for each dataframe
    plt.plot(df1.index.to_numpy(), df1.values, label='Existing Charging Speed')
    plt.plot(df2.index.to_numpy(), df2.values, label='6.6 kW')
    plt.plot(df3.index.to_numpy(), df3.values, label='19 kW')

    # Add labels and legend
    plt.xlabel('Hour')  # You might want to replace 'Index' with a relevant label
    plt.ylabel('Total Discharge Amount kWh')
    plt.legend(loc='upper right', title='V2G/Discharging Speed')
    plt.ylim(0, 65000)
    plt.grid(True)

    # Show the plot
    plt.show()


def heat_plot(df):
    # Create a larger figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot a heatmap with specified vmin and vmax, and add legend label
    heatmap = sns.heatmap(df, cmap='viridis', ax=ax, vmin=0, vmax=250, cbar_kws={'label': 'Available Storage (kW)'})

    # Adjust font size for labels and ticks
    heatmap.set_xlabel('Hour of Day', fontsize=18)
    heatmap.set_ylabel('Aggregated Charging Events', fontsize=18)

    # Set Y-axis ticks to show only 1 to 12
    # y_ticks_subset = range(1, 13)
    # y_tick_positions = [i - 0.5 for i in y_ticks_subset]  # Position ticks at the center of each cell
    # plt.yticks(y_tick_positions, [str(i) for i in y_ticks_subset], rotation=0, fontsize=10)

    plt.xticks(fontsize=12)

    # Add a title with increased font size
    plt.title('Available V2G Capacity', fontsize=18)

    # Increase font size for colorbar label
    cbar = heatmap.collections[0].colorbar
    cbar.set_label('Available Storage (kW)', fontsize=18)
    plt.gca().invert_yaxis()
    plt.tight_layout()

    # Show the plot
    plt.show()


def box_plot_with_stats_for_three(df_box, labels):
    # Set the y-axis limit
    y_min, y_max = 0, 700

    # Plot box plots for each dataframe separately
    for df, label in zip(df_box, labels):
        fig, ax = plt.subplots(figsize=(10, 8))
        # Adjust layout to remove margins
        plt.subplots_adjust(left=0.14, right=0.979, top=0.94, bottom=0.1)
        boxplot = ax.boxplot(df, labels=[f" {i}" for i in range(1, 25)], patch_artist=True)
        for box in boxplot['boxes']:
            box.set_facecolor('lightblue')  # Adjust the color as needed
        # Plot average line for each hour
        averages = df.values.mean(axis=0)
        ax.plot(range(1, 25), averages, marker='o', color='red', label='Average', linewidth=2)
        ax.set_title(f'V2G Availability - {label}', fontsize=24)
        ax.set_ylim(y_min, y_max)  # Set the y-axis limit
        ax.set_xlabel('Hour of Day', fontsize=22)
        ax.set_ylabel('Available Storage kW', fontsize=22)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_xticklabels([f" {i}" for i in range(1, 25)], rotation=0, ha='right', fontsize=14)
        ax.set_yticks(ax.get_yticks())
        ax.set_yticklabels(ax.get_yticks(), fontsize=18)
        # Add legend
        ax.legend(loc="upper right", fontsize=20)
        # Show the plot
        plt.show()


def V2G_cap_ch_r_mc(df):
    df = df[df["charging_speed"] != 0].fillna(0)
    df['end_time_charging'] = pd.to_datetime(df['end_time_charging'])
    # current speed
    df["V2G_SOC_half"] = (((df["battery[soc][end][charging]"] / 100) * df["bat_cap"]) - ((df["V2G_time_min"] / 60) / 2) * df["charging_speed"]) / (df["bat_cap"]) * 100
    df.loc[df["V2G_SOC_half"] < 0, "V2G_SOC_half"] = 0
    df["V2G_cap"] = (abs(df["V2G_SOC_half"]-df["battery[soc][end][charging]"])/100) * df["bat_cap"]
    df["V2G_cycle_time"] = (((df["battery[soc][end][charging]"] / 100) * df["bat_cap"]) / df["charging_speed"]) * 2
    df.replace({float('inf'): 0, float('-inf'): 0, pd.NaT: 0}, inplace=True)
    df.loc[df["V2G_cycle_time"] < 0, "V2G_cycle_time"] = 0
    df["V2G_max_cycle"] = df.apply(lambda row: math.floor((row["V2G_time_min"] / 60) / row["V2G_cycle_time"]) if row["V2G_cycle_time"] != 0 else 0, axis=1)
    df.loc[df["V2G_max_cycle"] < 0, "V2G_max_cycle"] = 0
    df.loc[df["V2G_max_cycle"] != 0, "V2G_cap"] *= df["V2G_max_cycle"]
    # with Level 2

    # Assuming df is your DataFrame
    df["V2G_SOC_half_6k"] = (((df["battery[soc][end][charging]"] / 100) * df["bat_cap"]) - ((df["V2G_time_min"]/60)/2) * np.maximum(df["charging_speed"], 6.6)) / (df["bat_cap"]) * 100
    df.loc[df["V2G_SOC_half_6k"] < 0, "V2G_SOC_half_6k"] = 0
    df["V2G_cap_6k"] = (abs(df["V2G_SOC_half_6k"]-df["battery[soc][end][charging]"])/100) * df["bat_cap"]
    df["V2G_cycle_6k_time"] = (((df["battery[soc][end][charging]"] / 100) * df["bat_cap"]) / (np.maximum(df["charging_speed"], 6.6))) * 2
    df.replace({float('inf'): 0, float('-inf'): 0, pd.NaT: 0}, inplace=True)
    df["V2G_max_cycle_6k"] = df.apply(lambda row: math.floor((row["V2G_time_min"] / 60) / row["V2G_cycle_6k_time"]) if row["V2G_cycle_6k_time"] != 0 else 0, axis=1)
    df.loc[df["V2G_max_cycle_6k"] < 0, "V2G_max_cycle_6k"] = 0
    df.loc[df["V2G_max_cycle_6k"] != 0, "V2G_cap_6k"] *= df["V2G_max_cycle_6k"]

    # Assuming df is your DataFrame
    df["V2G_SOC_half_19k"] = (((df["battery[soc][end][charging]"] / 100) * df["bat_cap"]) - ((df["V2G_time_min"]/60)/2) * np.maximum(df["charging_speed"], 19)) / (df["bat_cap"]) * 100
    df.loc[df["V2G_SOC_half_19k"] < 0, "V2G_SOC_half_19k"] = 0
    df["V2G_cap_19k"] = (abs(df["V2G_SOC_half_19k"]-df["battery[soc][end][charging]"])/100) * df["bat_cap"]
    df["V2G_cycle_19k_time"] = (((df["battery[soc][end][charging]"] / 100) * df["bat_cap"]) / np.maximum(df["charging_speed"], 19)) * 2
    df.replace({float('inf'): 0, float('-inf'): 0, pd.NaT: 0}, inplace=True)
    df["V2G_max_cycle_19k"] = df.apply(lambda row: math.floor((row["V2G_time_min"] / 60) / row["V2G_cycle_19k_time"]) if row["V2G_cycle_19k_time"] != 0 else 0, axis=1)
    df.loc[df["V2G_max_cycle_19k"] < 0, "V2G_max_cycle_19k"] = 0
    df.loc[df["V2G_max_cycle_19k"] != 0, "V2G_cap_19k"] *= df["V2G_max_cycle_19k"]

    return df


def V2G_cap_soc_r5_mc(df):
    df1 = df.copy()
    df1 = df1[df1["charging_speed"] != 0].fillna(0)
    df1['end_time_charging'] = pd.to_datetime(df1['end_time_charging'])

    df1["battery[soc][end][charging]"] = df1["battery[soc][end][charging]"] - 5
    df1 = charging_selection(df1)
    df1 = range_indicator(df1)
    df1["battery[soc][end][charging]"] = df1["battery[soc][end][charging]"] + 5
    # current speed
    df1["V2G_SOC_half"] = (((df1["battery[soc][end][charging]"] / 100) * df1["bat_cap"]) - ((df1["V2G_time_min"] / 60) / 2) * df1["charging_speed"]) / (df1["bat_cap"]) * 100
    df1.loc[df1["V2G_SOC_half"] < 0, "V2G_SOC_half"] = 0
    df1["V2G_cap"] = (abs(df1["V2G_SOC_half"] - (df1["battery[soc][end][charging]"] + 5)) / 100) * df1["bat_cap"]
    df1["V2G_cycle_time"] = (((df1["battery[soc][end][charging]"] / 100) * df1["bat_cap"]) / df1["charging_speed"]) + ((((df1["battery[soc][end][charging]"] - 5) / 100) * df1["bat_cap"]) / df1["charging_speed"])
    df1.replace({float('inf'): 0, float('-inf'): 0, pd.NaT: 0}, inplace=True)
    df1.loc[df1["V2G_cycle_time"] < 0, "V2G_cycle_time"] = 0
    df1["V2G_max_cycle"] = df1.apply(lambda row: math.floor((row["V2G_time_min"] / 60) / row["V2G_cycle_time"]) if row["V2G_cycle_time"] != 0 else 0, axis=1)
    df1.loc[df1["V2G_max_cycle"] < 0, "V2G_max_cycle"] = 0
    df1.loc[df1["V2G_max_cycle"] != 0, "V2G_cap"] *= df1["V2G_max_cycle"]
    # with Level 2

    # Assuming df1 is your DataFrame
    df1["V2G_SOC_half_6k"] = (((df1["battery[soc][end][charging]"] / 100) * df1["bat_cap"]) - ((df1["V2G_time_min"] / 60) / 2) * np.maximum(df1["charging_speed"], 6.6)) / (df1["bat_cap"])*100
    df1.loc[df1["V2G_SOC_half_6k"] < 0, "V2G_SOC_half_6k"] = 0
    df1["V2G_cap_6k"] = (abs(df1["V2G_SOC_half_6k"] - (df1["battery[soc][end][charging]"] + 5)) / 100) * df1["bat_cap"]
    df1["V2G_cycle_6k_time"] = (((df1["battery[soc][end][charging]"] / 100) * df1["bat_cap"]) / np.maximum(df1["charging_speed"], 6.6)) + ((((df1["battery[soc][end][charging]"] - 5) / 100) * df1["bat_cap"]) / np.maximum(df1["charging_speed"], 6.6))
    df1.replace({float('inf'): 0, float('-inf'): 0, pd.NaT: 0}, inplace=True)
    df1["V2G_max_cycle_6k"] = df1.apply(lambda row: math.floor((row["V2G_time_min"] / 60) / row["V2G_cycle_6k_time"]) if row["V2G_cycle_6k_time"] != 0 else 0, axis=1)
    df1.loc[df1["V2G_max_cycle_6k"] < 0, "V2G_max_cycle_6k"] = 0
    df1.loc[df1["V2G_max_cycle_6k"] != 0, "V2G_cap_6k"] *= df1["V2G_max_cycle_6k"]

    # Assuming df1 is your DataFrame
    df1["V2G_SOC_half_19k"] = (((df1["battery[soc][end][charging]"] / 100) * df1["bat_cap"]) - ((df1["V2G_time_min"] / 60) / 2) * np.maximum(df1["charging_speed"], 19)) / (df1["bat_cap"]) * 100
    df1.loc[df1["V2G_SOC_half_19k"] < 0, "V2G_SOC_half_19k"] = 0
    df1["V2G_cap_19k"] = (abs(df1["V2G_SOC_half_19k"] - (df1["battery[soc][end][charging]"] + 5)) / 100) * df1["bat_cap"]
    df1["V2G_cycle_19k_time"] = (((df1["battery[soc][end][charging]"] / 100) * df1["bat_cap"]) / np.maximum(df1["charging_speed"], 19)) + ((((df1["battery[soc][end][charging]"] - 5) / 100) * df1["bat_cap"]) / np.maximum(df1["charging_speed"], 19))
    df1.replace({float('inf'): 0, float('-inf'): 0, pd.NaT: 0}, inplace=True)
    df1["V2G_max_cycle_19k"] = df1.apply(lambda row: math.floor((row["V2G_time_min"] / 60) / row["V2G_cycle_19k_time"]) if row["V2G_cycle_19k_time"] != 0 else 0, axis=1)
    df1.loc[df1["V2G_max_cycle_19k"] < 0, "V2G_max_cycle_19k"] = 0
    df1.loc[df1["V2G_max_cycle_19k"] != 0, "V2G_cap_19k"] *= df1["V2G_max_cycle_19k"]
    return df1


def V2G_cap_soc_r10_mc(df):
    df1 = df.copy()
    df1 = df1[df1["charging_speed"] != 0].fillna(0)
    df1['end_time_charging'] = pd.to_datetime(df1['end_time_charging'])
    df1["battery[soc][end][charging]"] = df1["battery[soc][end][charging]"] - 10
    df1 = charging_selection(df1)
    df1 = range_indicator(df1)
    df1["battery[soc][end][charging]"] = df1["battery[soc][end][charging]"] + 10
    # current speed
    df1["V2G_SOC_half"] = (((df1["battery[soc][end][charging]"] / 100) * df1["bat_cap"]) - ((df1["V2G_time_min"] / 60) / 2) * df1["charging_speed"]) / (df1["bat_cap"]) * 100
    df1.loc[df1["V2G_SOC_half"] < 0, "V2G_SOC_half"] = 0
    df1["V2G_cap"] = (abs(df1["V2G_SOC_half"] - (df1["battery[soc][end][charging]"] + 10)) / 100) * df1["bat_cap"]
    df1["V2G_cycle_time"] = (((df1["battery[soc][end][charging]"] / 100) * df1["bat_cap"]) / df1["charging_speed"]) + ((((df1["battery[soc][end][charging]"] - 10) / 100) * df1["bat_cap"]) / df1["charging_speed"])
    df1.replace({float('inf'): 0, float('-inf'): 0, pd.NaT: 0}, inplace=True)
    df1.loc[df1["V2G_cycle_time"] < 0, "V2G_cycle_time"] = 0
    df1["V2G_max_cycle"] = df1.apply(lambda row: math.floor((row["V2G_time_min"] / 60) / row["V2G_cycle_time"]) if row["V2G_cycle_time"] != 0 else 0, axis=1)
    df1.loc[df1["V2G_max_cycle"] < 0, "V2G_max_cycle"] = 0
    df1.loc[df1["V2G_max_cycle"] != 0, "V2G_cap"] *= df1["V2G_max_cycle"]
    # with Level 2

    # Assuming df1 is your DataFrame
    df1["V2G_SOC_half_6k"] = (((df1["battery[soc][end][charging]"] / 100) * df1["bat_cap"]) - ((df1["V2G_time_min"] / 60) / 2) * np.maximum(df1["charging_speed"], 6.6)) / (df1["bat_cap"]) * 100
    df1.loc[df1["V2G_SOC_half_6k"] < 0, "V2G_SOC_half_6k"] = 0
    df1["V2G_cap_6k"] = (abs(df1["V2G_SOC_half_6k"] - (df1["battery[soc][end][charging]"] + 10)) / 100) * df1["bat_cap"]
    df1["V2G_cycle_6k_time"] = (((df1["battery[soc][end][charging]"] / 100) * df1["bat_cap"]) / np.maximum(df1["charging_speed"], 6.6)) + ((((df1["battery[soc][end][charging]"] - 10) / 100) * df1["bat_cap"]) / np.maximum(df1["charging_speed"], 6.6))
    df1.replace({float('inf'): 0, float('-inf'): 0, pd.NaT: 0}, inplace=True)
    df1["V2G_max_cycle_6k"] = df1.apply(lambda row: math.floor((row["V2G_time_min"] / 60) / row["V2G_cycle_6k_time"]) if row["V2G_cycle_6k_time"] != 0 else 0, axis=1)
    df1.loc[df1["V2G_max_cycle_6k"] < 0, "V2G_max_cycle_6k"] = 0
    df1.loc[df1["V2G_max_cycle_6k"] != 0, "V2G_cap_6k"] *= df1["V2G_max_cycle_6k"]

    # Assuming df1 is your DataFrame
    df1["V2G_SOC_half_19k"] = (((df1["battery[soc][end][charging]"] / 100) * df1["bat_cap"]) - ((df1["V2G_time_min"] / 60) / 2) * np.maximum(df1["charging_speed"], 19)) / (df1["bat_cap"]) * 100
    df1.loc[df1["V2G_SOC_half_19k"] < 0, "V2G_SOC_half_19k"] = 0
    df1["V2G_cap_19k"] = (abs(df1["V2G_SOC_half_19k"] - (df1["battery[soc][end][charging]"] + 10)) / 100) * df1["bat_cap"]
    df1["V2G_cycle_19k_time"] = (((df1["battery[soc][end][charging]"] / 100) * df1["bat_cap"]) / np.maximum(df1["charging_speed"], 19)) + ((((df1["battery[soc][end][charging]"] - 10) / 100) * df1["bat_cap"]) / np.maximum(df1["charging_speed"], 19))
    df1.replace({float('inf'): 0, float('-inf'): 0, pd.NaT: 0}, inplace=True)
    df1["V2G_max_cycle_19k"] = df1.apply(lambda row: math.floor((row["V2G_time_min"] / 60) / row["V2G_cycle_19k_time"]) if row["V2G_cycle_19k_time"] != 0 else 0, axis=1)
    df1.loc[df1["V2G_max_cycle_19k"] < 0, "V2G_max_cycle_19k"] = 0
    df1.loc[df1["V2G_max_cycle_19k"] != 0, "V2G_cap_19k"] *= df1["V2G_max_cycle_19k"]
    return df1
