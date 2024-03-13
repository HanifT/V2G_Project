import pytz
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import math
from datetime import timedelta
##################################################################################################################
##################################################################################################################


def read_clean(value):
    data_trips_full = pd.read_csv("G:\\My Drive\\PycharmProjects\\LSTM\\bev_trips_full.csv", low_memory=False)
    data_trips = pd.read_csv("G:\\My Drive\\PycharmProjects\\LSTM\\bev_trips.csv").drop("destination_label", axis=1)
    data_trips = pd.merge(data_trips, data_trips_full[["id", "Lat", "Long", "cluster_rank", "destination_label"]], how="left", on="id")
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
    data_charge_grouped = data_charge.groupby(['last_trip_id']).agg(agg_funcs).reset_index(inplace=True)
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
    data1 = data.groupby("id").tail(n=1).reset_index(drop=True)
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
    data1["origin_label"] = data1["destination_label"].shift(1)
    data2 = data1.copy()
    data2 = data2[data2["vehicle_name"] == value]
    data2 = data2.sort_values(by="ts_start")
    return data2
##################################################################################################################
##################################################################################################################


def clean_data():
    vehicle_names = ["P_1352", "P_1353", "P_1357", "P_1367", "P_1368", "P_1370", "P_1371", "P_1376",
                     "P_1381", "P_1384", "P_1388", "P_1393", "P_1403", "P_1409", "P_1412", "P_1414",
                     "P_1419", "P_1421", "P_1422", "P_1423", "P_1424", "P_1427", "P_1429", "P_1435",
                     "P_1087", "P_1091", "P_1092", "P_1093", "P_1094", "P_1098", "P_1100", "P_1109",
                     "P_1111", "P_1112", "P_1122", "P_1123", "P_1125", "P_1125a", "P_1127", "P_1131",
                     "P_1132", "P_1135", "P_1137", "P_1140", "P_1141", "P_1143", "P_1144", "P_1217",
                     "P_1253", "P_1257", "P_1260", "P_1267", "P_1271", "P_1272", "P_1279", "P_1280",
                     "P_1281", "P_1285", "P_1288", "P_1294", "P_1295", "P_1296", "P_1304", "P_1307", "P_1375"]
    df = pd.DataFrame()
    V2G_dataframes = pd.DataFrame()
    for vehicle_name in vehicle_names:
        df_full_trips = read_clean(vehicle_name)
        df_full_trips_short = trip_summary(df_full_trips)
        df_soc_req = soc_next(df_full_trips_short)
        # Failed Next Trip
        df_soc_req["f_next_trip"] = df_soc_req["battery[soc][start][trip]"].shift(-1) - df_soc_req["SOC_next_trip"]
        # Failed Next Charging
        df_soc_req["f_next_charge"] = df_soc_req["battery[soc][start][trip]"].shift(-1) - df_soc_req["SOC_need_next_charge"]
        df = pd.concat([df, df_soc_req], axis=0, ignore_index=True)
    return df
##################################################################################################################
##################################################################################################################


# Function to extract characters after the last "_" or " "
def extract_last_chars(input_string):
    if not isinstance(input_string, (str, bytes)):
        return None
    matches = re.findall(r'(?:_| )(\w{2,3})$', input_string)
    if matches:
        return int(matches[-1])
    return None
##################################################################################################################
##################################################################################################################


def trip_summary(df):
    data3 = df[["ts_start", "ts_end", "start_time_local", "end_time_local", "vehicle_name", "vehicle_model", "year", "month", "day", "hour", "duration_trip", "distance", "battery[soc][start][trip]",
               "battery[soc][end][trip]", "Lat", "Long", "destination_label", "origin_label", "energy[charge_type][type]", "battery[soc][start][charging]", "battery[soc][end][charging]", "start_time", "end_time", "duration_charging"]].copy()
    data3 = data3.rename(columns={'start_time': 'start_time_charging'})
    data3 = data3.rename(columns={'end_time': 'end_time_charging'})
    data3["start_time_charging"] = pd.to_datetime(data3["start_time_charging"])
    data3["end_time_charging"] = pd.to_datetime(data3["end_time_charging"])
    # Set the timezone to PST
    timezone = pytz.timezone('US/Pacific')
    # Convert the datetime to PST timezone
    data3["start_time_charging"] = pd.to_datetime(data3["start_time_charging"]).dt.tz_localize(timezone, ambiguous='NaT')
    data3["end_time_charging"] = pd.to_datetime(data3["end_time_charging"]).dt.tz_localize(timezone, ambiguous='NaT')
    data3.loc[:, "next_departure_time"] = data3["start_time_local"].shift(-1)
    data3.loc[data3["next_departure_time"] < data3["end_time_charging"], "end_time_charging"] = data3["next_departure_time"]
    data3.loc[data3["end_time_charging"] < data3["start_time_charging"], "end_time_charging"] = data3["next_departure_time"]
    data3.loc[:, "parking_time"] = data3["next_departure_time"] - data3["end_time_local"]
    data3.loc[:, "parking_time_minute"] = data3["parking_time"].dt.total_seconds() / 60
    data3.loc[:, "duration_charging_min"] = data3.loc[:, "duration_charging"] / 60
    data3.loc[data3["duration_charging_min"] > data3["parking_time_minute"], "parking_time_minute"] = data3["duration_charging_min"]
    data3["bat_cap"] = data3['vehicle_model'].apply(extract_last_chars)
    return data3
##################################################################################################################
##################################################################################################################


# selecting only the trips that have charging session at the end
def charging_dataframe(df, time):
    final_dataframes_charging = charging_selection(df)
    # determine teh charging speed based on the parking time, charging time and SOC before and after charging
    final_dataframes_charging = charging_speed(final_dataframes_charging)
    # range indicator is indicating if the trip will fail or not
    final_dataframes_charging = range_indicator(final_dataframes_charging)
    final_dataframes_charging = v2g_draw(final_dataframes_charging)
    final_dataframes_charging = final_dataframes_charging.loc[final_dataframes_charging["V2G_time_min"] >= time]
    return final_dataframes_charging
##################################################################################################################
##################################################################################################################


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


##################################################################################################################
##################################################################################################################


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

# draw_parking(final_dataframes)
# draw_charging(final_dataframes)
##################################################################################################################
##################################################################################################################


def draw_parking_boxplots(df):
    # Assuming 'final_dataframes' is your DataFrame
    df = df[df["destination_label"] != "Other"]
    # df = df[~((df["destination_label"] == "Work") & (df["origin_label"] == "Work"))]
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
    custom_colors = {'Home-Work': "#FFFF00", 'Work-Home': '#0019ff', 'Other-Home': '#0092ff', 'Other-Work': '#FFB100', 'Home-Home': '#00f3ff', 'Work-Work': '#FF7300'}
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
    ax2.set_ylabel('Average Parking Time (minutes)', fontsize=20)
    # Increase tick font size
    ax1.tick_params(axis='both', labelsize=18)
    ax2.tick_params(axis='both', labelsize=18)
    # Show the plot
    plt.savefig('soc.png', bbox_inches='tight')
    plt.show()

# draw_parking_boxplots(final_dataframes)
##################################################################################################################
##################################################################################################################


def soc_next(df):
    # df = df_full_trips_short.copy()
    df = df.reset_index(drop=True)
    mask = (df["battery[soc][start][trip]"] - df["battery[soc][end][trip]"] < 0)
    # Update the values where the condition is true
    df.loc[mask, 'battery[soc][end][trip]'] = df.loc[mask, 'battery[soc][start][trip]'] - ((0.28 * df.loc[mask, 'distance']) * (100 / df["bat_cap"]))
    df.loc[((~df["battery[soc][start][charging]"].isna()) & (df["parking_time_minute"] >= 60.1) & (df["battery[soc][end][trip]"] + 10 < df["battery[soc][start][trip]"].shift(-1))), "energy[charge_type][type]"] = "LEVEL_2/1"
    df.loc[((~df["battery[soc][start][charging]"].isna()) & (df["parking_time_minute"] < 60.1) & (df["battery[soc][end][trip]"] + 10 < df["battery[soc][start][trip]"].shift(-1))), "energy[charge_type][type]"] = "DC_FAST"
    df.loc[(df["battery[soc][end][trip]"] + 10 < df["battery[soc][start][trip]"].shift(-1)) & (df["parking_time_minute"] >= 60.1), "energy[charge_type][type]"] = "LEVEL_2/1"
    df.loc[(df["battery[soc][end][trip]"] + 10 < df["battery[soc][start][trip]"].shift(-1)) & (df["parking_time_minute"] < 60.1), "energy[charge_type][type]"] = "DC_FAST"
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
##################################################################################################################
##################################################################################################################


def charging_selection(df):
    # Filter rows where charging duration is not NaN
    final_df_charging = df.loc[~df["duration_charging"].isna()].copy()
    # Calculate minimum range for different scenarios
    final_df_charging["minrange"] = (final_df_charging["bat_cap"] * (final_df_charging["battery[soc][end][charging]"] / 100)) / 0.28
    final_df_charging["minrange_need"] = (final_df_charging["bat_cap"] * (final_df_charging["SOC_next_trip"] / 100)) / 0.28
    final_df_charging["minrange_need_nextc"] = (final_df_charging["bat_cap"] * (final_df_charging["SOC_need_next_charge"] / 100)) / 0.28
    return final_df_charging

##################################################################################################################
##################################################################################################################


def charging_speed(df):
    df["charging_speed"] = ((((df["battery[soc][end][charging]"] - df["battery[soc][start][charging]"]) / 100) * df["bat_cap"]) / (df["duration_charging_min"] / 60))
    df.loc[df["charging_speed"] <= 1.6, "charge_type"] = "LEVEL_1"
    df.loc[(df["charging_speed"] > 1.6) & (df["charging_speed"] < 21), "charge_type"] = "LEVEL_2"
    df.loc[df["charging_speed"] >= 21, "charge_type"] = "DC_FAST"
    return df
##################################################################################################################
##################################################################################################################


def range_indicator(df):
    # next trip fail indicator
    df.loc[:, "next_trip_fail"] = df.loc[:, "minrange"] < df.loc[:, "minrange_need"]
    # next charging sessions fail indicator
    df.loc[:, "next_c_fail"] = df.loc[:, "minrange"] < df.loc[:, "minrange_need_nextc"]
    return df
##################################################################################################################
##################################################################################################################


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
##################################################################################################################
##################################################################################################################


def V2G_cap_ch_r(df):
    # level 2 12
    df["V2G_SOC_half_12k"] = (((df["battery[soc][end][charging]"] / 100) * df["bat_cap"]) - ((df["V2G_time_min"] / 60) / 2) * 12) / (df["bat_cap"]) * 100
    df.loc[df["V2G_SOC_half_12k"] < 0, "V2G_SOC_half_12k"] = 0
    df["V2G_cap_12k"] = (abs(df["V2G_SOC_half_12k"]-df["battery[soc][end][charging]"])/100)*df["bat_cap"]
    # with Level 2
    # Assuming df is your DataFrame
    df["V2G_SOC_half_6k"] = (((df["battery[soc][end][charging]"] / 100) * df["bat_cap"]) - ((df["V2G_time_min"]/60)/2) * 6.6) / (df["bat_cap"]) * 100
    df.loc[df["V2G_SOC_half_6k"] < 0, "V2G_SOC_half_6k"] = 0
    df["V2G_cap_6k"] = (abs(df["V2G_SOC_half_6k"]-df["battery[soc][end][charging]"]) / 100) * df["bat_cap"]
    # Assuming df is your DataFrame
    df["V2G_SOC_half_19k"] = (((df["battery[soc][end][charging]"] / 100) * df["bat_cap"]) - ((df["V2G_time_min"]/60)/2) * 19) / (df["bat_cap"]) * 100
    df.loc[df["V2G_SOC_half_19k"] < 0, "V2G_SOC_half_19k"] = 0
    df["V2G_cap_19k"] = (abs(df["V2G_SOC_half_19k"] - (df["battery[soc][end][charging]"])) / 100) * df["bat_cap"]
    return df
##################################################################################################################
##################################################################################################################


def V2G_cap_soc_r5(df):
    df1 = df.copy()
    df1["battery[soc][end][charging]"] = df1["battery[soc][end][charging]"] - 5
    df1 = charging_selection(df1)
    df1 = range_indicator(df1)
    df1["battery[soc][end][charging]"] = df1["battery[soc][end][charging]"] + 5
    # current speed
    df1["V2G_SOC_half_12k"] = (((df1["battery[soc][end][charging]"] / 100) * df1["bat_cap"]) - ((df1["V2G_time_min"] / 60) / 2) * 12) / (df1["bat_cap"]) * 100
    df1.loc[df1["V2G_SOC_half_12k"] < 0, "V2G_SOC_half_12k"] = 0
    df1["V2G_cap_12k"] = (abs(df1["V2G_SOC_half_12k"] - (df1["battery[soc][end][charging]"] + 5)) / 100) * df1["bat_cap"]
    # Assuming df1 is your DataFrame
    df1["V2G_SOC_half_6k"] = (((df1["battery[soc][end][charging]"] / 100) * df1["bat_cap"]) - ((df1["V2G_time_min"] / 60) / 2) * 6.6) / (df1["bat_cap"])*100
    df1.loc[df1["V2G_SOC_half_6k"] < 0, "V2G_SOC_half_6k"] = 0
    df1["V2G_cap_6k"] = (abs(df1["V2G_SOC_half_6k"] - (df1["battery[soc][end][charging]"] + 5)) / 100) * df1["bat_cap"]
    # Assuming df1 is your DataFrame
    df1["V2G_SOC_half_19k"] = (((df1["battery[soc][end][charging]"] / 100) * df1["bat_cap"]) - ((df1["V2G_time_min"] / 60) / 2) * 19) / (df1["bat_cap"]) * 100
    df1.loc[df1["V2G_SOC_half_19k"] < 0, "V2G_SOC_half_19k"] = 0
    df1["V2G_cap_19k"] = (abs(df1["V2G_SOC_half_19k"] - (df1["battery[soc][end][charging]"] + 5)) / 100) * df1["bat_cap"]
    return df1
##################################################################################################################
##################################################################################################################


def V2G_cap_soc_r10(df):
    df1 = df.copy()
    df1["battery[soc][end][charging]"] = df1["battery[soc][end][charging]"] - 10
    df1 = charging_selection(df1)
    df1 = range_indicator(df1)
    df1["battery[soc][end][charging]"] = df1["battery[soc][end][charging]"] + 10
    # current speed
    df1["V2G_SOC_half_12k"] = (((df1["battery[soc][end][charging]"] / 100) * df1["bat_cap"]) - ((df1["V2G_time_min"] / 60) / 2) * 12) / (df1["bat_cap"]) * 100
    df1.loc[df1["V2G_SOC_half_12k"] < 0, "V2G_SOC_half_12k"] = 0
    df1["V2G_cap_12k"] = (abs(df1["V2G_SOC_half_12k"] - (df1["battery[soc][end][charging]"] + 10)) / 100) * df1["bat_cap"]
    # Assuming df1 is your DataFrame
    df1["V2G_SOC_half_6k"] = (((df1["battery[soc][end][charging]"] / 100) * df1["bat_cap"]) - ((df1["V2G_time_min"] / 60) / 2) * 6.6) / (df1["bat_cap"]) * 100
    df1.loc[df1["V2G_SOC_half_6k"] < 0, "V2G_SOC_half_6k"] = 0
    df1["V2G_cap_6k"] = (abs(df1["V2G_SOC_half_6k"] - (df1["battery[soc][end][charging]"] + 10)) / 100) * df1["bat_cap"]
    # Assuming df1 is your DataFrame
    df1["V2G_SOC_half_19k"] = (((df1["battery[soc][end][charging]"] / 100) * df1["bat_cap"]) - ((df1["V2G_time_min"] / 60) / 2) * 19) / (df1["bat_cap"]) * 100
    df1.loc[df1["V2G_SOC_half_19k"] < 0, "V2G_SOC_half_19k"] = 0
    df1["V2G_cap_19k"] = (abs(df1["V2G_SOC_half_19k"] - (df1["battery[soc][end][charging]"] + 10)) / 100) * df1["bat_cap"]
    return df1
##################################################################################################################
##################################################################################################################


def storage_cap(df):
    V2G_hourly = pd.DataFrame(index=df.index, columns=range(24))
    V2G_hourly_12 = V2G_hourly.fillna(0)
    V2G_hourly_6 = V2G_hourly_12.copy()
    V2G_hourly_19 = V2G_hourly_12.copy()
    for i in df.index:
        start_hour = df.loc[i, "end_time_charging"].hour
        discharging_speed = 12
        total_capacity = df.loc[i, "V2G_cap_12k"]
        current_hour = start_hour
        remaining_capacity = total_capacity
        while remaining_capacity > 0:
            discharge_amount = min(discharging_speed, remaining_capacity)
            V2G_hourly_12.loc[i, current_hour] += discharge_amount
            remaining_capacity -= discharge_amount
            if current_hour < 21:
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
            if current_hour < 21:
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
            if current_hour < 21:
                current_hour += 1
            else:
                current_hour = 0
    V2G_hourly_12 = pd.merge(df[["month", "day"]], V2G_hourly_12, how="left", left_index=True, right_index=True)
    V2G_hourly_12_sum = V2G_hourly_12.groupby(["month", "day"]).sum()
    V2G_hourly_6 = pd.merge(df[["month", "day"]], V2G_hourly_6, how="left", left_index=True, right_index=True)
    V2G_hourly_6_sum = V2G_hourly_6.groupby(["month", "day"]).sum()
    V2G_hourly_19 = pd.merge(df[["month", "day"]], V2G_hourly_19, how="left", left_index=True, right_index=True)
    V2G_hourly_19_sum = V2G_hourly_19.groupby(["month", "day"]).sum()
    return V2G_hourly_12, V2G_hourly_6, V2G_hourly_19, V2G_hourly_12_sum, V2G_hourly_6_sum, V2G_hourly_19_sum
##################################################################################################################
##################################################################################################################


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
##################################################################################################################
##################################################################################################################


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
##################################################################################################################
##################################################################################################################


def box_plot_with_stats_for_three(df_box, labels, ymin1, ymax1):
    # Set the y-axis limit
    y_min, y_max = ymin1, ymax1
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
        ax.set_ylabel('Available Storage kW per Day', fontsize=22)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_xticklabels([f" {i}" for i in range(0, 24)], rotation=0, ha='right', fontsize=14)
        ax.set_yticks(ax.get_yticks())
        ax.set_yticklabels(ax.get_yticks(), fontsize=18)
        # Add legend
        ax.legend(loc="upper right", fontsize=20)
        # Show the plot
        plt.show()
##################################################################################################################
##################################################################################################################


def V2G_cap_ch_r_mc(df):
    df = df[df["charging_speed"] != 0].fillna(0)
    df['end_time_charging'] = pd.to_datetime(df['end_time_charging'])

    # current speed
    df["V2G_SOC_half_12k"] = (((df["battery[soc][end][charging]"] / 100) * df["bat_cap"]) - ((df["V2G_time_min"] / 60) / 2) * 12) / (df["bat_cap"]) * 100
    df.loc[df["V2G_SOC_half_12k"] < 0, "V2G_SOC_half_12k"] = 0
    df["V2G_cap_12k"] = (abs(df["V2G_SOC_half_12k"]-df["battery[soc][end][charging]"])/100) * df["bat_cap"]
    df["V2G_cycle_12k_time"] = (((df["battery[soc][end][charging]"] / 100) * df["bat_cap"]) / 12) * 2
    df.replace({float('inf'): 0, float('-inf'): 0, pd.NaT: 0}, inplace=True)
    df.loc[df["V2G_cycle_12k_time"] < 0, "V2G_cycle_12k_time"] = 0
    df["V2G_max_cycle_12k"] = df.apply(lambda row: math.floor((row["V2G_time_min"] / 60) / row["V2G_cycle_12k_time"]) if row["V2G_cycle_12k_time"] != 0 else 0, axis=1)
    df.loc[df["V2G_max_cycle_12k"] < 0, "V2G_max_cycle_12k"] = 0
    df.loc[df["V2G_max_cycle_12k"] != 0, "V2G_cap_12k"] *= df["V2G_max_cycle_12k"]

    # Assuming df is your DataFrame
    df["V2G_SOC_half_6k"] = (((df["battery[soc][end][charging]"] / 100) * df["bat_cap"]) - ((df["V2G_time_min"]/60)/2) * 6.6) / (df["bat_cap"]) * 100
    df.loc[df["V2G_SOC_half_6k"] < 0, "V2G_SOC_half_6k"] = 0
    df["V2G_cap_6k"] = (abs(df["V2G_SOC_half_6k"]-df["battery[soc][end][charging]"])/100) * df["bat_cap"]
    df["V2G_cycle_6k_time"] = (((df["battery[soc][end][charging]"] / 100) * df["bat_cap"]) / (6.6)) * 2
    df.replace({float('inf'): 0, float('-inf'): 0, pd.NaT: 0}, inplace=True)
    df["V2G_max_cycle_6k"] = df.apply(lambda row: math.floor((row["V2G_time_min"] / 60) / row["V2G_cycle_6k_time"]) if row["V2G_cycle_6k_time"] != 0 else 0, axis=1)
    df.loc[df["V2G_max_cycle_6k"] < 0, "V2G_max_cycle_6k"] = 0
    df.loc[df["V2G_max_cycle_6k"] != 0, "V2G_cap_6k"] *= df["V2G_max_cycle_6k"]

    # Assuming df is your DataFrame
    df["V2G_SOC_half_19k"] = (((df["battery[soc][end][charging]"] / 100) * df["bat_cap"]) - ((df["V2G_time_min"]/60)/2) * 19) / (df["bat_cap"]) * 100
    df.loc[df["V2G_SOC_half_19k"] < 0, "V2G_SOC_half_19k"] = 0
    df["V2G_cap_19k"] = (abs(df["V2G_SOC_half_19k"]-df["battery[soc][end][charging]"])/100) * df["bat_cap"]
    df["V2G_cycle_19k_time"] = (((df["battery[soc][end][charging]"] / 100) * df["bat_cap"]) / 19) * 2
    df.replace({float('inf'): 0, float('-inf'): 0, pd.NaT: 0}, inplace=True)
    df["V2G_max_cycle_19k"] = df.apply(lambda row: math.floor((row["V2G_time_min"] / 60) / row["V2G_cycle_19k_time"]) if row["V2G_cycle_19k_time"] != 0 else 0, axis=1)
    df.loc[df["V2G_max_cycle_19k"] < 0, "V2G_max_cycle_19k"] = 0
    df.loc[df["V2G_max_cycle_19k"] != 0, "V2G_cap_19k"] *= df["V2G_max_cycle_19k"]
    return df
##################################################################################################################
##################################################################################################################


def V2G_cap_soc_r5_mc(df):
    df1 = df.copy()
    df1 = df1[df1["charging_speed"] != 0].fillna(0)
    df1['end_time_charging'] = pd.to_datetime(df1['end_time_charging'])
    df1["battery[soc][end][charging]"] = df1["battery[soc][end][charging]"] - 5
    df1 = charging_selection(df1)
    df1 = range_indicator(df1)
    df1["battery[soc][end][charging]"] = df1["battery[soc][end][charging]"] + 5

    # current speed
    df1["V2G_SOC_half_12k"] = (((df1["battery[soc][end][charging]"] / 100) * df1["bat_cap"]) - ((df1["V2G_time_min"] / 60) / 2) * 12) / (df1["bat_cap"]) * 100
    df1.loc[df1["V2G_SOC_half_12k"] < 0, "V2G_SOC_half_12k"] = 0
    df1["V2G_cap_12k"] = (abs(df1["V2G_SOC_half_12k"] - (df1["battery[soc][end][charging]"] + 5)) / 100) * df1["bat_cap"]
    df1["V2G_cycle_12k_time"] = (((df1["battery[soc][end][charging]"] / 100) * df1["bat_cap"]) /12) + ((((df1["battery[soc][end][charging]"] - 5) / 100) * df1["bat_cap"]) / 12)
    df1.replace({float('inf'): 0, float('-inf'): 0, pd.NaT: 0}, inplace=True)
    df1.loc[df1["V2G_cycle_12k_time"] < 0, "V2G_cycle_12k_time"] = 0
    df1["V2G_max_cycle_12k"] = df1.apply(lambda row: math.floor((row["V2G_time_min"] / 60) / row["V2G_cycle_12k_time"]) if row["V2G_cycle_12k_time"] != 0 else 0, axis=1)
    df1.loc[df1["V2G_max_cycle_12k"] < 0, "V2G_max_cycle_12k"] = 0
    df1.loc[df1["V2G_max_cycle_12k"] != 0, "V2G_cap_12k"] *= df1["V2G_max_cycle_12k"]

    # Assuming df1 is your DataFrame
    df1["V2G_SOC_half_6k"] = (((df1["battery[soc][end][charging]"] / 100) * df1["bat_cap"]) - ((df1["V2G_time_min"] / 60) / 2) * 6.6) / (df1["bat_cap"])*100
    df1.loc[df1["V2G_SOC_half_6k"] < 0, "V2G_SOC_half_6k"] = 0
    df1["V2G_cap_6k"] = (abs(df1["V2G_SOC_half_6k"] - (df1["battery[soc][end][charging]"] + 5)) / 100) * df1["bat_cap"]
    df1["V2G_cycle_6k_time"] = (((df1["battery[soc][end][charging]"] / 100) * df1["bat_cap"]) / 6.6) + ((((df1["battery[soc][end][charging]"] - 5) / 100) * df1["bat_cap"]) / 6.6)
    df1.replace({float('inf'): 0, float('-inf'): 0, pd.NaT: 0}, inplace=True)
    df1["V2G_max_cycle_6k"] = df1.apply(lambda row: math.floor((row["V2G_time_min"] / 60) / row["V2G_cycle_6k_time"]) if row["V2G_cycle_6k_time"] != 0 else 0, axis=1)
    df1.loc[df1["V2G_max_cycle_6k"] < 0, "V2G_max_cycle_6k"] = 0
    df1.loc[df1["V2G_max_cycle_6k"] != 0, "V2G_cap_6k"] *= df1["V2G_max_cycle_6k"]

    # Assuming df1 is your DataFrame
    df1["V2G_SOC_half_19k"] = (((df1["battery[soc][end][charging]"] / 100) * df1["bat_cap"]) - ((df1["V2G_time_min"] / 60) / 2) * 19) / (df1["bat_cap"]) * 100
    df1.loc[df1["V2G_SOC_half_19k"] < 0, "V2G_SOC_half_19k"] = 0
    df1["V2G_cap_19k"] = (abs(df1["V2G_SOC_half_19k"] - (df1["battery[soc][end][charging]"] + 5)) / 100) * df1["bat_cap"]
    df1["V2G_cycle_19k_time"] = (((df1["battery[soc][end][charging]"] / 100) * df1["bat_cap"]) / 19) + ((((df1["battery[soc][end][charging]"] - 5) / 100) * df1["bat_cap"]) / 19)
    df1.replace({float('inf'): 0, float('-inf'): 0, pd.NaT: 0}, inplace=True)
    df1["V2G_max_cycle_19k"] = df1.apply(lambda row: math.floor((row["V2G_time_min"] / 60) / row["V2G_cycle_19k_time"]) if row["V2G_cycle_19k_time"] != 0 else 0, axis=1)
    df1.loc[df1["V2G_max_cycle_19k"] < 0, "V2G_max_cycle_19k"] = 0
    df1.loc[df1["V2G_max_cycle_19k"] != 0, "V2G_cap_19k"] *= df1["V2G_max_cycle_19k"]
    return df1
##################################################################################################################
##################################################################################################################


def V2G_cap_soc_r10_mc(df):
    df1 = df.copy()
    df1 = df1[df1["charging_speed"] != 0].fillna(0)
    df1['end_time_charging'] = pd.to_datetime(df1['end_time_charging'])
    df1["battery[soc][end][charging]"] = df1["battery[soc][end][charging]"] - 10
    df1 = charging_selection(df1)
    df1 = range_indicator(df1)
    df1["battery[soc][end][charging]"] = df1["battery[soc][end][charging]"] + 10

    # current speed
    df1["V2G_SOC_half_12k"] = (((df1["battery[soc][end][charging]"] / 100) * df1["bat_cap"]) - ((df1["V2G_time_min"] / 60) / 2) * 12) / (df1["bat_cap"]) * 100
    df1.loc[df1["V2G_SOC_half_12k"] < 0, "V2G_SOC_half_12k"] = 0
    df1["V2G_cap_12k"] = (abs(df1["V2G_SOC_half_12k"] - (df1["battery[soc][end][charging]"] + 10)) / 100) * df1["bat_cap"]
    df1["V2G_cycle_12k_time"] = (((df1["battery[soc][end][charging]"] / 100) * df1["bat_cap"]) / np.maximum(df1["charging_speed"], 12)) + ((((df1["battery[soc][end][charging]"] - 10) / 100) * df1["bat_cap"]) / 12)
    df1.replace({float('inf'): 0, float('-inf'): 0, pd.NaT: 0}, inplace=True)
    df1.loc[df1["V2G_cycle_12k_time"] < 0, "V2G_cycle_12k_time"] = 0
    df1["V2G_max_cycle_12k"] = df1.apply(lambda row: math.floor((row["V2G_time_min"] / 60) / row["V2G_cycle_12k_time"]) if row["V2G_cycle_12k_time"] != 0 else 0, axis=1)
    df1.loc[df1["V2G_max_cycle_12k"] < 0, "V2G_max_cycle_12k"] = 0
    df1.loc[df1["V2G_max_cycle_12k"] != 0, "V2G_cap_12k"] *= df1["V2G_max_cycle_12k"]

    # Assuming df1 is your DataFrame
    df1["V2G_SOC_half_6k"] = (((df1["battery[soc][end][charging]"] / 100) * df1["bat_cap"]) - ((df1["V2G_time_min"] / 60) / 2) * 6.6) / (df1["bat_cap"]) * 100
    df1.loc[df1["V2G_SOC_half_6k"] < 0, "V2G_SOC_half_6k"] = 0
    df1["V2G_cap_6k"] = (abs(df1["V2G_SOC_half_6k"] - (df1["battery[soc][end][charging]"] + 10)) / 100) * df1["bat_cap"]
    df1["V2G_cycle_6k_time"] = (((df1["battery[soc][end][charging]"] / 100) * df1["bat_cap"]) / 6.6) + ((((df1["battery[soc][end][charging]"] - 10) / 100) * df1["bat_cap"]) / 6.6)
    df1.replace({float('inf'): 0, float('-inf'): 0, pd.NaT: 0}, inplace=True)
    df1["V2G_max_cycle_6k"] = df1.apply(lambda row: math.floor((row["V2G_time_min"] / 60) / row["V2G_cycle_6k_time"]) if row["V2G_cycle_6k_time"] != 0 else 0, axis=1)
    df1.loc[df1["V2G_max_cycle_6k"] < 0, "V2G_max_cycle_6k"] = 0
    df1.loc[df1["V2G_max_cycle_6k"] != 0, "V2G_cap_6k"] *= df1["V2G_max_cycle_6k"]

    # Assuming df1 is your DataFrame
    df1["V2G_SOC_half_19k"] = (((df1["battery[soc][end][charging]"] / 100) * df1["bat_cap"]) - ((df1["V2G_time_min"] / 60) / 2) * 19) / (df1["bat_cap"]) * 100
    df1.loc[df1["V2G_SOC_half_19k"] < 0, "V2G_SOC_half_19k"] = 0
    df1["V2G_cap_19k"] = (abs(df1["V2G_SOC_half_19k"] - (df1["battery[soc][end][charging]"] + 10)) / 100) * df1["bat_cap"]
    df1["V2G_cycle_19k_time"] = (((df1["battery[soc][end][charging]"] / 100) * df1["bat_cap"]) / 19) + ((((df1["battery[soc][end][charging]"] - 10) / 100) * df1["bat_cap"]) / 19)
    df1.replace({float('inf'): 0, float('-inf'): 0, pd.NaT: 0}, inplace=True)
    df1["V2G_max_cycle_19k"] = df1.apply(lambda row: math.floor((row["V2G_time_min"] / 60) / row["V2G_cycle_19k_time"]) if row["V2G_cycle_19k_time"] != 0 else 0, axis=1)
    df1.loc[df1["V2G_max_cycle_19k"] < 0, "V2G_max_cycle_19k"] = 0
    df1.loc[df1["V2G_max_cycle_19k"] != 0, "V2G_cap_19k"] *= df1["V2G_max_cycle_19k"]
    return df1
##################################################################################################################
##################################################################################################################


def v2g_normal(df):
    V2G_cap_charging_rate = V2G_cap_ch_r(df).reset_index(drop=True)
    V2G_cap_charging_rate.loc[V2G_cap_charging_rate["charging_speed"] < 0, "charging_speed"] = 0
    V2G_cap_charging_rate.loc[V2G_cap_charging_rate["charging_speed"] == 0, "V2G_cap_12k"] = 0
    V2G_cap_charging_rate.loc[V2G_cap_charging_rate["charging_speed"] == 0, "V2G_cap_19k"] = 0
    V2G_cap_charging_rate.loc[V2G_cap_charging_rate["charging_speed"] == 0, "V2G_cap_6k"] = 0
    V2G_cap_charging_rate.loc[V2G_cap_charging_rate["charging_speed"] == 0, "V2G_cap"] = 0
    V2G_hourly_12, V2G_hourly_6, V2G_hourly_19, V2G_hourly_12_sum, V2G_hourly_6_sum, V2G_hourly_19_sum = storage_cap(V2G_cap_charging_rate)
    V2G_hourly_12_sum_reset = V2G_hourly_12_sum.reset_index(drop=False).drop("month", axis=1).groupby("day").sum()
    V2G_hourly_6_sum_reset = V2G_hourly_6_sum.reset_index(drop=False).drop("month", axis=1).groupby("day").sum()
    V2G_hourly_19_sum_reset = V2G_hourly_19_sum.reset_index(drop=False).drop("month", axis=1).groupby("day").sum()
    return V2G_cap_charging_rate, V2G_hourly_12, V2G_hourly_6, V2G_hourly_19, V2G_hourly_12_sum, V2G_hourly_6_sum, V2G_hourly_19_sum, V2G_hourly_12_sum_reset, V2G_hourly_6_sum_reset, V2G_hourly_19_sum_reset
##################################################################################################################
##################################################################################################################


def v2g_r5(df):
    # calculating the storage capacity based on the different charging discharging speed and SOC at the end of charging
    V2G_cap_soc_rate5 = V2G_cap_soc_r5(df).reset_index(drop=True)
    V2G_cap_soc_rate5.loc[V2G_cap_soc_rate5["charging_speed"] < 0, "charging_speed"] = 0
    V2G_cap_soc_rate5.loc[V2G_cap_soc_rate5["charging_speed"] == 0, "V2G_cap_12k"] = 0
    V2G_cap_soc_rate5.loc[V2G_cap_soc_rate5["charging_speed"] == 0, "V2G_cap_19k"] = 0
    V2G_cap_soc_rate5.loc[V2G_cap_soc_rate5["charging_speed"] == 0, "V2G_cap_6k"] = 0
    V2G_cap_soc_rate5.loc[V2G_cap_soc_rate5["charging_speed"] == 0, "V2G_cap"] = 0
    V2G_hourly_12_s5, V2G_hourly_6_s5, V2G_hourly_19_s5, V2G_hourly_12_sum_s5, V2G_hourly_6_sum_s5, V2G_hourly_19_sum_s5 = storage_cap(V2G_cap_soc_rate5)
    V2G_hourly_12_sum_reset_s5 = V2G_hourly_12_sum_s5.reset_index(drop=False).drop("month", axis=1).groupby("day").sum()
    V2G_hourly_6_sum_reset_s5 = V2G_hourly_6_sum_s5.reset_index(drop=False).drop("month", axis=1).groupby("day")
    V2G_hourly_6_sum_reset_s5 = V2G_hourly_6_sum_reset_s5.sum()
    V2G_hourly_19_sum_reset_s5 = V2G_hourly_19_sum_s5.reset_index(drop=False).drop("month", axis=1).groupby("day")
    V2G_hourly_19_sum_reset_s5 = V2G_hourly_19_sum_reset_s5.sum()
    return V2G_cap_soc_rate5, V2G_hourly_12_s5, V2G_hourly_6_s5, V2G_hourly_19_s5, V2G_hourly_12_sum_s5, V2G_hourly_6_sum_s5, V2G_hourly_19_sum_s5, V2G_hourly_12_sum_reset_s5, V2G_hourly_6_sum_reset_s5, V2G_hourly_19_sum_reset_s5
##################################################################################################################
##################################################################################################################


def v2g_r10(df):
    V2G_cap_soc_rate10 = V2G_cap_soc_r10(df).reset_index(drop=True)
    V2G_cap_soc_rate10.loc[V2G_cap_soc_rate10["charging_speed"] < 0, "charging_speed"] = 0
    V2G_cap_soc_rate10.loc[V2G_cap_soc_rate10["charging_speed"] == 0, "V2G_cap_12k"] = 0
    V2G_cap_soc_rate10.loc[V2G_cap_soc_rate10["charging_speed"] == 0, "V2G_cap_19k"] = 0
    V2G_cap_soc_rate10.loc[V2G_cap_soc_rate10["charging_speed"] == 0, "V2G_cap_6k"] = 0
    V2G_cap_soc_rate10.loc[V2G_cap_soc_rate10["charging_speed"] == 0, "V2G_cap"] = 0
    V2G_hourly_12_s10, V2G_hourly_6_s10, V2G_hourly_19_s10, V2G_hourly_12_sum_s10, V2G_hourly_6_sum_s10, V2G_hourly_19_sum_s10 = storage_cap(V2G_cap_soc_rate10)
    V2G_hourly_12_sum_reset_s10 = V2G_hourly_12_sum_s10.reset_index(drop=False).drop("month", axis=1).groupby("day").sum()
    V2G_hourly_6_sum_reset_s10 = V2G_hourly_6_sum_s10.reset_index(drop=False).drop("month", axis=1).groupby("day").sum()
    V2G_hourly_19_sum_reset_s10 = V2G_hourly_19_sum_s10.reset_index(drop=False).drop("month", axis=1).groupby("day").sum()
    return V2G_cap_soc_rate10, V2G_hourly_12_s10, V2G_hourly_6_s10, V2G_hourly_19_s10, V2G_hourly_12_sum_s10, V2G_hourly_6_sum_s10, V2G_hourly_19_sum_s10, V2G_hourly_12_sum_reset_s10, V2G_hourly_6_sum_reset_s10, V2G_hourly_19_sum_reset_s10
##################################################################################################################
##################################################################################################################


def v2g_normal_mc(df):
    # calculating the storage capacity based on the different charging discharging speed
    V2G_cap_charging_rate_mc = V2G_cap_ch_r_mc(df).reset_index(drop=True)
    V2G_cap_charging_rate_mc.loc[V2G_cap_charging_rate_mc["charging_speed"] < 0, "charging_speed"] = 0
    V2G_cap_charging_rate_mc.loc[V2G_cap_charging_rate_mc["charging_speed"] == 0, "V2G_cap"] = 0
    V2G_cap_charging_rate_mc.loc[V2G_cap_charging_rate_mc["charging_speed"] == 0, "V2G_cap_12k"] = 0
    V2G_cap_charging_rate_mc.loc[V2G_cap_charging_rate_mc["charging_speed"] == 0, "V2G_cap_6k"] = 0
    V2G_cap_charging_rate_mc.loc[V2G_cap_charging_rate_mc["charging_speed"] == 0, "V2G_cap_19k"] = 0
    V2G_hourly_12_mc, V2G_hourly_6_mc, V2G_hourly_19_mc, V2G_hourly_12_sum_mc, V2G_hourly_6_sum_mc, V2G_hourly_19_sum_mc = storage_cap(V2G_cap_charging_rate_mc)
    V2G_hourly_12_sum_reset_mc = V2G_hourly_12_sum_mc.reset_index(drop=False).drop("month", axis=1).groupby("day").sum()
    V2G_hourly_6_sum_reset_mc = V2G_hourly_6_sum_mc.reset_index(drop=False).drop("month", axis=1).groupby("day").sum()
    V2G_hourly_19_sum_reset_mc = V2G_hourly_19_sum_mc.reset_index(drop=False).drop("month", axis=1).groupby("day").sum()
    return V2G_cap_charging_rate_mc, V2G_hourly_12_mc, V2G_hourly_6_mc, V2G_hourly_19_mc, V2G_hourly_12_sum_mc, V2G_hourly_6_sum_mc, V2G_hourly_19_sum_mc, V2G_hourly_12_sum_reset_mc, V2G_hourly_6_sum_reset_mc, V2G_hourly_19_sum_reset_mc
##################################################################################################################
##################################################################################################################


def v2g_r5_mc(df):
    # calculating the storage capacity based on the different charging discharging speed and SOC at the end of charging
    V2G_cap_soc_rate5_mc = V2G_cap_soc_r5_mc(df).reset_index(drop=True)
    V2G_cap_soc_rate5_mc.loc[V2G_cap_soc_rate5_mc["charging_speed"] < 0, "charging_speed"] = 0
    V2G_cap_soc_rate5_mc.loc[V2G_cap_soc_rate5_mc["charging_speed"] == 0, "V2G_cap"] = 0
    V2G_cap_soc_rate5_mc.loc[V2G_cap_soc_rate5_mc["charging_speed"] == 0, "V2G_cap_12k"] = 0
    V2G_cap_soc_rate5_mc.loc[V2G_cap_soc_rate5_mc["charging_speed"] == 0, "V2G_cap_6k"] = 0
    V2G_cap_soc_rate5_mc.loc[V2G_cap_soc_rate5_mc["charging_speed"] == 0, "V2G_cap_19k"] = 0
    V2G_hourly_12_s5_mc, V2G_hourly_6_s5_mc, V2G_hourly_19_s5_mc, V2G_hourly_12_sum_s5_mc, V2G_hourly_6_sum_s5_mc, V2G_hourly_19_sum_s5_mc = storage_cap(V2G_cap_soc_rate5_mc)
    V2G_hourly_12_sum_reset_s5_mc = V2G_hourly_12_sum_s5_mc.reset_index(drop=False).drop("month", axis=1).groupby("day").sum()
    V2G_hourly_6_sum_reset_s5_mc = V2G_hourly_6_sum_s5_mc.reset_index(drop=False).drop("month", axis=1).groupby("day").sum()
    V2G_hourly_19_sum_reset_s5_mc = V2G_hourly_19_sum_s5_mc.reset_index(drop=False).drop("month", axis=1).groupby("day").sum()
    return V2G_cap_soc_rate5_mc, V2G_hourly_12_s5_mc, V2G_hourly_6_s5_mc, V2G_hourly_19_s5_mc, V2G_hourly_12_sum_s5_mc, V2G_hourly_6_sum_s5_mc, V2G_hourly_19_sum_s5_mc, V2G_hourly_12_sum_reset_s5_mc, V2G_hourly_6_sum_reset_s5_mc, V2G_hourly_19_sum_reset_s5_mc
##################################################################################################################
##################################################################################################################


def v2g_r10_mc(df):
    V2G_cap_soc_rate10_mc = V2G_cap_soc_r10_mc(df).reset_index(drop=True)
    V2G_cap_soc_rate10_mc.loc[V2G_cap_soc_rate10_mc["charging_speed"] < 0, "charging_speed"] = 0
    V2G_cap_soc_rate10_mc.loc[V2G_cap_soc_rate10_mc["charging_speed"] == 0, "V2G_cap"] = 0
    V2G_cap_soc_rate10_mc.loc[V2G_cap_soc_rate10_mc["charging_speed"] == 0, "V2G_cap_12k"] = 0
    V2G_cap_soc_rate10_mc.loc[V2G_cap_soc_rate10_mc["charging_speed"] == 0, "V2G_cap_19k"] = 0
    V2G_cap_soc_rate10_mc.loc[V2G_cap_soc_rate10_mc["charging_speed"] == 0, "V2G_cap_6k"] = 0
    V2G_hourly_12_s10_mc, V2G_hourly_6_s10_mc, V2G_hourly_19_s10_mc, V2G_hourly_12_sum_s10_mc, V2G_hourly_6_sum_s10_mc, V2G_hourly_19_sum_s10_mc = storage_cap(V2G_cap_soc_rate10_mc)
    V2G_hourly_12_sum_reset_s10_mc = V2G_hourly_12_sum_s10_mc.reset_index(drop=False).drop("month", axis=1).groupby("day").sum()
    V2G_hourly_6_sum_reset_s10_mc = V2G_hourly_6_sum_s10_mc.reset_index(drop=False).drop("month", axis=1).groupby("day").sum()
    V2G_hourly_19_sum_reset_s10_mc = V2G_hourly_19_sum_s10_mc.reset_index(drop=False).drop("month", axis=1).groupby("day").sum()
    return V2G_cap_soc_rate10_mc, V2G_hourly_12_s10_mc, V2G_hourly_6_s10_mc, V2G_hourly_19_s10_mc, V2G_hourly_12_sum_s10_mc, V2G_hourly_6_sum_s10_mc, V2G_hourly_19_sum_s10_mc, V2G_hourly_12_sum_reset_s10_mc, V2G_hourly_6_sum_reset_s10_mc, V2G_hourly_19_sum_reset_s10_mc
##################################################################################################################
##################################################################################################################


def total_storage(df1, df2, df3, df1_r5, df2_r5, df3_r5, df1_r10, df2_r10, df3_r10):
    data = {'6.6 kW': [df1_r5.sum().sum() / 1000, df2_r5.sum().sum() / 1000, df3_r5.sum().sum() / 1000],
            '12 kw': [df1.sum().sum() / 1000, df2.sum().sum() / 1000, df3.sum().sum() / 1000],
            '19 kW': [df1_r10.sum().sum() / 1000, df2_r10.sum().sum() / 1000, df3_r10.sum().sum() / 1000]}

    df_summary_storage = pd.DataFrame(data, index=['Total', 'Total_s5', 'Total_s10']).T

    return df_summary_storage
##################################################################################################################
##################################################################################################################


def total_storage_tou(df1, df2, df3):
    data = {'6.6 kW': [df1.sum().sum() / 1000],
            '12 kw': [df2.sum().sum() / 1000],
            '19 kW': [df3.sum().sum() / 1000]}

    df_summary_storage = pd.DataFrame(data, index=['Total']).T

    return df_summary_storage
##################################################################################################################
##################################################################################################################


def failure_estimation(df1, df2):
    ratio5_nt = df1["next_trip_fail"].value_counts(normalize=True)
    ratio5_nc = df1["next_c_fail"].value_counts(normalize=True)

    ratio10_nt = df2["next_trip_fail"].value_counts(normalize=True)
    ratio10_nc = df2["next_c_fail"].value_counts(normalize=True)

    data = {'ratio5': [ratio5_nt[1]*100, ratio5_nc[1]*100],
            'ratio10': [ratio10_nt[1]*100, ratio10_nc[1]*100]}
    data = pd.DataFrame(data, index=['next_trip', 'next_charging']).T

    return data
##################################################################################################################
##################################################################################################################


def total_capacity(df):
    total_cap_df = df.groupby('vehicle_name', as_index=False).first()[['vehicle_name', 'bat_cap']]
    total_cap = total_cap_df["bat_cap"].sum()
    return total_cap


##################################################################################################################
##################################################################################################################
def charging_c_k(df):
    df1 = df.copy()
    df1["charging_cap"] = ((df1["battery[soc][end][charging]"] - df1["battery[soc][start][charging]"]) / 100) * df["bat_cap"]
    df1["charging_cycle"] = (df1["charging_cap"] > 0).astype(int)
    df1["charging_v2g_energy_6k"] = df1["charging_cap"] + df1["V2G_cap_6k"]
    df1["charging_v2g_cycle"] = (df1["V2G_cap_6k"] > 0).astype(int)
    df1["charging_v2g_energy_12k"] = df1["charging_cap"] + df1["V2G_cap_12k"]
    df1["charging_v2g_cycle_12k"] = (df1["V2G_cap_12k"] > 0).astype(int)
    df1["charging_v2g_energy_19k"] = df1["charging_cap"] + df1["V2G_cap_19k"]
    df1["charging_v2g_cycle_19k"] = (df1["V2G_cap_19k"] > 0).astype(int)
    df1 = df1.groupby("vehicle_name")[["charging_cap", "charging_v2g_energy_6k", "charging_v2g_energy_12k", "charging_v2g_energy_19k",
                                       "charging_cycle", "charging_v2g_cycle"]].sum()
    return df1


def charging_c_st(df):
    df1 = df.copy()
    df1["charging_cap"] = 0
    df1["charging_cycle"] = (df1["charging_cap"] > 0).astype(int)
    df1["charging_v2g_energy_6k"] = df1["V2G_cap_6k"]
    df1["charging_v2g_cycle"] = (df1["V2G_cap_6k"] > 0).astype(int)
    df1["charging_v2g_energy_12k"] = df1["V2G_cap_12k"]
    df1["charging_v2g_cycle_12k"] = (df1["V2G_cap_12k"] > 0).astype(int)
    df1["charging_v2g_energy_19k"] = df1["V2G_cap_19k"]
    df1["charging_v2g_cycle_19k"] = (df1["V2G_cap_19k"] > 0).astype(int)
    df1 = df1.groupby("vehicle_name")[["charging_cap", "charging_v2g_energy_6k", "charging_v2g_energy_12k", "charging_v2g_energy_19k",
                                       "charging_cycle", "charging_v2g_cycle"]].sum()
    return df1


##################################################################################################################
##################################################################################################################
def charging_c_k_mc(df):
    df1 = df.copy()
    df1["charging_cap"] = ((df1["battery[soc][end][charging]"] - df1["battery[soc][start][charging]"]) / 100) * df["bat_cap"]
    df1["charging_cycle"] = (df1["charging_cap"] > 0).astype(int)
    df1["charging_v2g_energy_6k"] = df1["charging_cap"] + df1["V2G_cap_6k"]
    df1["charging_v2g_cycle"] = df1["V2G_max_cycle_6k"] + 1
    df1["charging_v2g_energy_12k"] = df1["charging_cap"] + df1["V2G_cap_12k"]
    df1["charging_v2g_cycle_12k"] = df1["V2G_max_cycle_12k"] + 1
    df1["charging_v2g_energy_19k"] = df1["charging_cap"] + df1["V2G_cap_19k"]
    df1["charging_v2g_cycle_19k"] = df1["V2G_max_cycle_19k"] + 1
    df1 = df1.groupby("vehicle_name")[["charging_cap", "charging_v2g_energy_6k", "charging_v2g_energy_12k", "charging_v2g_energy_19k",
                                       "charging_cycle", "charging_v2g_cycle"]].sum()
    return df1


##################################################################################################################
##################################################################################################################
def charging_c_k_tou(df):
    df1 = df.copy()
    df1["charging_cap"] = ((df1["battery[soc][end][charging]"] - df1["battery[soc][start][charging]"]) / 100) * df["bat_cap"]
    # df1.loc[df1["indicator_column"].isna(), "charging_cap"] = 0
    df1.loc[df1["indicator_column"] == True, "charging_cap"] = 0
    df1.loc[df1["energy[charge_type][type]"] == "Parking", "charging_cap"] = 0
    df1.loc[df1["energy[charge_type][type]"].isna(), "charging_cap"] = 0

    df1["charging_cycle"] = (df1["charging_cap"] > 0).astype(int)
    df1["charging_v2g_energy_6k"] = df1["charging_cap"] + df1["V2G_cap_6k_tou"]
    df1["charging_v2g_cycle"] = (df1["V2G_cap_6k_tou"] > 0).astype(int)
    df1["charging_v2g_energy_12k"] = df1["charging_cap"] + df1["V2G_cap_12k_tou"]
    df1["charging_v2g_cycle_12k"] = (df1["V2G_cap_12k_tou"] > 0).astype(int)
    df1["charging_v2g_energy_19k"] = df1["charging_cap"] + df1["V2G_cap_19k_tou"]
    df1["charging_v2g_cycle_19k"] = (df1["V2G_cap_19k_tou"] > 0).astype(int)
    df1['bat_cap'] = df.groupby('vehicle_name')['bat_cap'].transform(lambda x: x.mode().iloc[0])
    df1["start_time_local"] = pd.to_datetime(df1["start_time_local"])

    df1 = df1.groupby("vehicle_name").agg(
        charging_cap=("charging_cap", "sum"),
        charging_v2g_energy_6k=("charging_v2g_energy_6k", "sum"),
        charging_v2g_energy_12k=("charging_v2g_energy_12k", "sum"),
        charging_v2g_energy_19k=("charging_v2g_energy_19k", "sum"),
        charging_cycle=("charging_cycle", "sum"),
        charging_v2g_cycle=("charging_v2g_cycle", "sum"),
        bat_cap=("bat_cap", "first"),  # Use "first" or "mode" here
        observation_day=("start_time_local", lambda x: (x.max() - x.min()).days)  # Difference in days
    )

    return df1


def charging_c_k_tou_real(df):
    df1 = df.copy()
    df1["charging_cap"] = ((df1["battery[soc][end][charging]"] - df1["battery[soc][start][charging]"]) / 100) * df["bat_cap"]

    df1["charging_cycle"] = (df1["charging_cap"] > 0).astype(int)
    df1["charging_v2g_energy_6k"] = df1["charging_cap"] + df1["V2G_cap_6k_tou"]
    df1["charging_v2g_cycle"] = (df1["V2G_cap_6k_tou"] > 0).astype(int)
    df1["charging_v2g_energy_12k"] = df1["charging_cap"] + df1["V2G_cap_12k_tou"]
    df1["charging_v2g_cycle_12k"] = (df1["V2G_cap_12k_tou"] > 0).astype(int)
    df1["charging_v2g_energy_19k"] = df1["charging_cap"] + df1["V2G_cap_19k_tou"]
    df1["charging_v2g_cycle_19k"] = (df1["V2G_cap_19k_tou"] > 0).astype(int)
    df1['bat_cap'] = df.groupby('vehicle_name')['bat_cap'].transform(lambda x: x.mode().iloc[0])
    df1["start_time_local"] = pd.to_datetime(df1["start_time_local"])


    df1 = df1.groupby("vehicle_name").agg(
        charging_cap=("charging_cap", "sum"),
        charging_v2g_energy_6k=("charging_v2g_energy_6k", "sum"),
        charging_v2g_energy_12k=("charging_v2g_energy_12k", "sum"),
        charging_v2g_energy_19k=("charging_v2g_energy_19k", "sum"),
        charging_cycle=("charging_cycle", "sum"),
        charging_v2g_cycle=("charging_v2g_cycle", "sum"),
        bat_cap=("bat_cap", "first"),  # Use "first" or "mode" here
        observation_day=("start_time_local", lambda x: (x.max() - x.min()).days)  # Difference in days
    )

    return df1

##################################################################################################################
##################################################################################################################
# Define peak time slots
def is_peak_time(hour, minute):
    for start, end in peak_time_slots:
        if start <= hour < end or (start == hour and 0 <= minute < 60):
            return True
    return False


peak_time_slots = [(16, 21)]  # Peak time slots from 4 PM to 9 PM


# Function to calculate start and end time of discharging and charging
def calculate_v2g(row):

    discharge_start = row['end_time_charging']
    discharge_hour = discharge_start.hour
    discharge_minute = discharge_start.minute
    charge_end = row['next_departure_time']
    depart_hour = row['next_departure_time'].hour
    depart_min = row['next_departure_time'].minute
    discharge_end = None
    charge_start = None

    # Charging end and departure before peak
    if ((not is_peak_time(discharge_hour, discharge_minute)) and (discharge_hour < min(min(peak_time_slots))) and (not is_peak_time(depart_hour, depart_min)) and (depart_hour < min(min(peak_time_slots))) and
            ((row["next_departure_time"].date() - row["end_time_charging"].date()).days < 1)):
        discharge_start = row['end_time_charging']
        discharge_end = discharge_start
        charge_end = row['next_departure_time']
        charge_start = charge_end

    # Charging end before peak and departure during peak
    if ((not is_peak_time(discharge_hour, discharge_minute)) and (discharge_hour < min(min(peak_time_slots))) and (is_peak_time(depart_hour, depart_min)) and (depart_hour >= min(min(peak_time_slots))) and
            ((row["next_departure_time"].date() - row["end_time_charging"].date()).days < 1)):
        discharge_start = row['end_time_charging']
        discharge_start = discharge_start.replace(hour=min(min(peak_time_slots)), minute=0, second=0)
        discharge_end = row['next_departure_time']
        charge_end = row['next_departure_time']
        charge_start = discharge_end

    # Charging end before peak and departure after peak
    if ((not is_peak_time(discharge_hour, discharge_minute)) and (discharge_hour < min(min(peak_time_slots))) and (not is_peak_time(depart_hour, depart_min)) and (depart_hour >= max(max(peak_time_slots))) and
            ((row["next_departure_time"].date() - row["end_time_charging"].date()).days < 1)):
        discharge_start = row['end_time_charging']
        discharge_start = discharge_start.replace(hour=min(min(peak_time_slots)), minute=0, second=0)
        discharge_end = discharge_start.replace(hour=max(max(peak_time_slots)), minute=0, second=0)
        charge_end = row['next_departure_time']
        charge_start = discharge_end

    # Charging end during peak and departure during peak
    if ((is_peak_time(discharge_hour, discharge_minute)) and (discharge_hour >= min(min(peak_time_slots))) and (is_peak_time(depart_hour, depart_min)) and (depart_hour >= min(min(peak_time_slots))) and
            ((row["next_departure_time"].date() - row["end_time_charging"].date()).days < 1)):
        discharge_start = row['end_time_charging']
        discharge_end = row['next_departure_time']
        charge_end = row['next_departure_time']
        charge_start = discharge_end

    # Charging end during peak and departure after peak
    if ((is_peak_time(discharge_hour, discharge_minute)) and (discharge_hour >= min(min(peak_time_slots))) and (not is_peak_time(depart_hour, depart_min)) and (depart_hour >= max(max(peak_time_slots))) and
            ((row["next_departure_time"].date() - row["end_time_charging"].date()).days < 1)):
        discharge_start = row['end_time_charging']
        discharge_end = discharge_start.replace(hour=max(max(peak_time_slots)), minute=0, second=0)
        charge_end = row['next_departure_time']
        charge_start = discharge_end

    # Charging end after peak and departure after peak
    if ((not is_peak_time(discharge_hour, discharge_minute)) and (discharge_hour >= max(max(peak_time_slots))) and (not is_peak_time(depart_hour, depart_min)) and (depart_hour >= max(max(peak_time_slots))) and
            ((row["next_departure_time"].date() - row["end_time_charging"].date()).days < 1)):
        discharge_start = row['end_time_charging']
        discharge_end = discharge_start
        charge_end = row['next_departure_time']
        charge_start = discharge_end

    # when charging and departure happen in two days  row["next_departure_time"].date() - row["end_time_charging"].date()) == 1

    # Charging end before peak and departure before peak next day
    if ((not is_peak_time(discharge_hour, discharge_minute)) and (discharge_hour < min(min(peak_time_slots))) and (not is_peak_time(depart_hour, depart_min)) and (depart_hour < min(min(peak_time_slots))) and
            ((row["next_departure_time"].date() - row["end_time_charging"].date()).days >= 1) and ((row["next_departure_time"].date() - row["end_time_charging"].date()).days < 2)):
        discharge_start = row['end_time_charging']
        discharge_start = discharge_start.replace(hour=min(min(peak_time_slots)), minute=0, second=0)
        discharge_end = discharge_start.replace(hour=max(max(peak_time_slots)), minute=0, second=0)
        charge_end = row['next_departure_time']
        charge_start = discharge_end

    # Charging end before peak and departure during peak next day
    if ((not is_peak_time(discharge_hour, discharge_minute)) and (discharge_hour < min(min(peak_time_slots))) and (is_peak_time(depart_hour, depart_min)) and (depart_hour >= min(min(peak_time_slots))) and
            ((row["next_departure_time"].date() - row["end_time_charging"].date()).days >= 1) and ((row["next_departure_time"].date() - row["end_time_charging"].date()).days < 2)):
        discharge_start = row['end_time_charging']
        discharge_start = discharge_start.replace(hour=min(min(peak_time_slots)), minute=0, second=0)
        discharge_end = discharge_start.replace(hour=max(max(peak_time_slots)), minute=0, second=0)
        charge_end = discharge_start.replace(hour=min(min(peak_time_slots)), minute=0, second=0) + timedelta(days=1)
        charge_start = discharge_end

    # Charging end before peak and departure after peak next day
    if ((not is_peak_time(discharge_hour, discharge_minute)) and (discharge_hour < min(min(peak_time_slots))) and (not is_peak_time(depart_hour, depart_min)) and (depart_hour >= max(max(peak_time_slots))) and
            ((row["next_departure_time"].date() - row["end_time_charging"].date()).days >= 1) and ((row["next_departure_time"].date() - row["end_time_charging"].date()).days < 2)):
        discharge_start = row['end_time_charging']
        discharge_start = discharge_start.replace(hour=min(min(peak_time_slots)), minute=0, second=0)
        discharge_end = discharge_start.replace(hour=max(max(peak_time_slots)), minute=0, second=0)
        charge_end = discharge_start.replace(hour=min(min(peak_time_slots)), minute=0, second=0) + timedelta(days=1)
        charge_start = discharge_end

    # Charging end during peak and departure before peak next day
    if ((is_peak_time(discharge_hour, discharge_minute)) and (discharge_hour >= min(min(peak_time_slots))) and (not is_peak_time(depart_hour, depart_min)) and (depart_hour < min(min(peak_time_slots))) and
            ((row["next_departure_time"].date() - row["end_time_charging"].date()).days >= 1) and ((row["next_departure_time"].date() - row["end_time_charging"].date()).days < 2)):
        discharge_start = row['end_time_charging']
        discharge_end = discharge_start.replace(hour=max(max(peak_time_slots)), minute=0, second=0)
        charge_end = row["next_departure_time"]
        charge_start = discharge_end

    # Charging end during peak and departure during peak next day
    if ((is_peak_time(discharge_hour, discharge_minute)) and (discharge_hour >= min(min(peak_time_slots))) and (is_peak_time(depart_hour, depart_min)) and (depart_hour >= min(min(peak_time_slots))) and
            ((row["next_departure_time"].date() - row["end_time_charging"].date()).days >= 1) and ((row["next_departure_time"].date() - row["end_time_charging"].date()).days < 2)):
        discharge_start = row['end_time_charging']
        discharge_end = discharge_start.replace(hour=max(max(peak_time_slots)), minute=0, second=0)
        charge_end = discharge_start.replace(hour=min(min(peak_time_slots)), minute=0, second=0) + timedelta(days=1)
        charge_start = discharge_end

    # Charging end during peak and departure after peak next day
    if ((is_peak_time(discharge_hour, discharge_minute)) and (discharge_hour >= min(min(peak_time_slots))) and (not is_peak_time(depart_hour, depart_min)) and (depart_hour >= max(max(peak_time_slots))) and
            ((row["next_departure_time"].date() - row["end_time_charging"].date()).days >= 1) and ((row["next_departure_time"].date() - row["end_time_charging"].date()).days < 2)):
        discharge_start = row['end_time_charging']
        discharge_end = discharge_start.replace(hour=max(max(peak_time_slots)), minute=0, second=0)
        charge_end = discharge_start.replace(hour=min(min(peak_time_slots)), minute=0, second=0) + timedelta(days=1)
        charge_start = discharge_end

    # Charging end after peak and departure before peak next day
    if ((not is_peak_time(discharge_hour, discharge_minute)) and (discharge_hour >= max(max(peak_time_slots))) and (not is_peak_time(depart_hour, depart_min)) and (depart_hour < min(min(peak_time_slots))) and
            ((row["next_departure_time"].date() - row["end_time_charging"].date()).days >= 1) and ((row["next_departure_time"].date() - row["end_time_charging"].date()).days < 2)):
        discharge_start = row['end_time_charging']
        discharge_end = row['end_time_charging']
        charge_end = row["next_departure_time"]
        charge_start = row["next_departure_time"]

    # Charging end after peak and departure during peak next day
    if ((not is_peak_time(discharge_hour, discharge_minute)) and (discharge_hour >= max(max(peak_time_slots))) and (is_peak_time(depart_hour, depart_min)) and (depart_hour >= min(min(peak_time_slots))) and
            ((row["next_departure_time"].date() - row["end_time_charging"].date()).days >= 1) and ((row["next_departure_time"].date() - row["end_time_charging"].date()).days < 2)):
        discharge_start = discharge_start.replace(hour=min(min(peak_time_slots)), minute=0, second=0) + timedelta(days=1)
        discharge_end = row["next_departure_time"]
        charge_end = row["next_departure_time"]
        charge_start = row["next_departure_time"]

    # Charging end after peak and departure after peak next day
    if ((not is_peak_time(discharge_hour, discharge_minute)) and (discharge_hour >= max(max(peak_time_slots))) and (not is_peak_time(depart_hour, depart_min)) and (depart_hour >= max(max(peak_time_slots))) and
            ((row["next_departure_time"].date() - row["end_time_charging"].date()).days >= 1) and ((row["next_departure_time"].date() - row["end_time_charging"].date()).days < 2)):
        discharge_start = discharge_start.replace(hour=min(min(peak_time_slots)), minute=0, second=0) + timedelta(days=1)
        discharge_end = discharge_start.replace(hour=max(max(peak_time_slots)), minute=0, second=0) + timedelta(days=1)
        charge_end = row["next_departure_time"]
        charge_start = discharge_end

    # Charging end after peak and departure before peak 2 days
    if ((not is_peak_time(discharge_hour, discharge_minute)) and (discharge_hour < min(min(peak_time_slots))) and (not is_peak_time(depart_hour, depart_min)) and (depart_hour < min(min(peak_time_slots))) and
            ((row["next_departure_time"].date() - row["end_time_charging"].date()).days >= 2)):
        discharge_start = discharge_start.replace(hour=min(min(peak_time_slots)), minute=0, second=0)
        discharge_end = discharge_start.replace(hour=max(max(peak_time_slots)), minute=0, second=0)
        charge_end = row["next_departure_time"]
        charge_start = discharge_end

    # Charging end during peak and departure before peak 2 days
    if ((is_peak_time(discharge_hour, discharge_minute)) and (discharge_hour >= min(min(peak_time_slots))) and (not is_peak_time(depart_hour, depart_min)) and (depart_hour < min(min(peak_time_slots))) and
            ((row["next_departure_time"].date() - row["end_time_charging"].date()).days >= 2)):
        discharge_start = row["end_time_charging"]
        discharge_end = discharge_start.replace(hour=max(max(peak_time_slots)), minute=0, second=0)
        charge_end = row["next_departure_time"]
        charge_start = discharge_end

    # Charging end after peak and departure before peak 2 days
    if ((not is_peak_time(discharge_hour, discharge_minute)) and (discharge_hour >= max(max(peak_time_slots))) and (not is_peak_time(depart_hour, depart_min)) and (depart_hour < min(min(peak_time_slots))) and
            ((row["next_departure_time"].date() - row["end_time_charging"].date()).days >= 2)):
        discharge_start = discharge_start.replace(hour=min(min(peak_time_slots)), minute=0, second=0) + timedelta(days=1)
        discharge_end = discharge_start.replace(hour=max(max(peak_time_slots)), minute=0, second=0) + timedelta(days=1)
        charge_end = row["next_departure_time"]
        charge_start = discharge_end

    else:
        if pd.isna(discharge_end) and pd.isna(charge_start):
            discharge_start = discharge_start.replace(hour=min(min(peak_time_slots)), minute=0, second=0) + timedelta(days=1)
            discharge_end = discharge_start.replace(hour=max(max(peak_time_slots)), minute=0, second=0) + timedelta(days=1)
            charge_end = discharge_start.replace(hour=min(min(peak_time_slots)), minute=0, second=0) + timedelta(days=2)
            charge_start = discharge_end

    return discharge_start, discharge_end, charge_start, charge_end,  row['next_departure_time']


##################################################################################################################
##################################################################################################################
def v2g_tou_cap(df):
    # Apply the function to each row and add the results as new columns
    df[['discharge_start', 'discharge_end', 'charge_start', 'charge_end', "next_departure_time1"]] = df.apply(calculate_v2g, axis=1, result_type='expand')

    df["V2G_time_min"] = (df["discharge_end"] - df["discharge_start"]).dt.seconds / 60
    df["V2G_time_charge"] = (df["charge_end"] - df["charge_start"]).dt.seconds / 60

    df["V2G_SOC_tou_6k"] = ((((df["battery[soc][end][charging]"] / 100) * df["bat_cap"]) - ((df["V2G_time_min"] / 60) * 6.6)) / (df["bat_cap"])) * 100
    df.loc[df["V2G_SOC_tou_6k"] < 0, "V2G_SOC_tou_6k"] = 0
    df["V2G_cap_6k_tou"] = (abs(df["V2G_SOC_tou_6k"] - (df["battery[soc][end][charging]"])) / 100) * df["bat_cap"]

    df["V2G_SOC_tou_12k"] = ((((df["battery[soc][end][charging]"] / 100) * df["bat_cap"]) - ((df["V2G_time_min"] / 60) * 12)) / (df["bat_cap"])) * 100
    df.loc[df["V2G_SOC_tou_12k"] < 0, "V2G_SOC_tou_12k"] = 0
    df["V2G_cap_12k_tou"] = (abs(df["V2G_SOC_tou_12k"] - (df["battery[soc][end][charging]"])) / 100) * df["bat_cap"]

    df["V2G_SOC_tou_19k"] = ((((df["battery[soc][end][charging]"] / 100) * df["bat_cap"]) - ((df["V2G_time_min"] / 60) * 19)) / (df["bat_cap"])) * 100
    df.loc[df["V2G_SOC_tou_19k"] < 0, "V2G_SOC_tou_19k"] = 0
    df["V2G_cap_19k_tou"] = (abs(df["V2G_SOC_tou_19k"] - (df["battery[soc][end][charging]"])) / 100) * df["bat_cap"]

    return df


def v2g_tou_cap_20(df):
    # Apply the function to each row and add the results as new columns
    df[['discharge_start', 'discharge_end', 'charge_start', 'charge_end', "next_departure_time1"]] = df.apply(calculate_v2g, axis=1, result_type='expand')

    df["V2G_time_min"] = (df["discharge_end"] - df["discharge_start"]).dt.seconds / 60
    df["V2G_time_charge"] = (df["charge_end"] - df["charge_start"]).dt.seconds / 60

    df["V2G_SOC_tou_6k"] = ((((df["battery[soc][end][charging]"] / 100) * df["bat_cap"]) - ((df["V2G_time_min"] / 60) * 6.6)) / (df["bat_cap"])) * 100
    df.loc[df["V2G_SOC_tou_6k"] < 20, "V2G_SOC_tou_6k"] = 20
    df["V2G_cap_6k_tou"] = (abs(df["V2G_SOC_tou_6k"] - (df["battery[soc][end][charging]"])) / 100) * df["bat_cap"]

    df["V2G_SOC_tou_12k"] = ((((df["battery[soc][end][charging]"] / 100) * df["bat_cap"]) - ((df["V2G_time_min"] / 60) * 12)) / (df["bat_cap"])) * 100
    df.loc[df["V2G_SOC_tou_12k"] < 20, "V2G_SOC_tou_12k"] = 20
    df["V2G_cap_12k_tou"] = (abs(df["V2G_SOC_tou_12k"] - (df["battery[soc][end][charging]"])) / 100) * df["bat_cap"]

    df["V2G_SOC_tou_19k"] = ((((df["battery[soc][end][charging]"] / 100) * df["bat_cap"]) - ((df["V2G_time_min"] / 60) * 19)) / (df["bat_cap"])) * 100
    df.loc[df["V2G_SOC_tou_19k"] < 20, "V2G_SOC_tou_19k"] = 20
    df["V2G_cap_19k_tou"] = (abs(df["V2G_SOC_tou_19k"] - (df["battery[soc][end][charging]"])) / 100) * df["bat_cap"]

    return df


def v2g_tou_cap_30(df):
    # Apply the function to each row and add the results as new columns
    df[['discharge_start', 'discharge_end', 'charge_start', 'charge_end', "next_departure_time1"]] = df.apply(calculate_v2g, axis=1, result_type='expand')

    df["V2G_time_min"] = (df["discharge_end"] - df["discharge_start"]).dt.seconds / 60
    df["V2G_time_charge"] = (df["charge_end"] - df["charge_start"]).dt.seconds / 60

    df["V2G_SOC_tou_6k"] = ((((df["battery[soc][end][charging]"] / 100) * df["bat_cap"]) - ((df["V2G_time_min"] / 60) * 6.6)) / (df["bat_cap"])) * 100
    df.loc[df["V2G_SOC_tou_6k"] < 30, "V2G_SOC_tou_6k"] = 30
    df["V2G_cap_6k_tou"] = (abs(df["V2G_SOC_tou_6k"] - (df["battery[soc][end][charging]"])) / 100) * df["bat_cap"]

    df["V2G_SOC_tou_12k"] = ((((df["battery[soc][end][charging]"] / 100) * df["bat_cap"]) - ((df["V2G_time_min"] / 60) * 12)) / (df["bat_cap"])) * 100
    df.loc[df["V2G_SOC_tou_12k"] < 30, "V2G_SOC_tou_12k"] = 30
    df["V2G_cap_12k_tou"] = (abs(df["V2G_SOC_tou_12k"] - (df["battery[soc][end][charging]"])) / 100) * df["bat_cap"]

    df["V2G_SOC_tou_19k"] = ((((df["battery[soc][end][charging]"] / 100) * df["bat_cap"]) - ((df["V2G_time_min"] / 60) * 19)) / (df["bat_cap"])) * 100
    df.loc[df["V2G_SOC_tou_19k"] < 30, "V2G_SOC_tou_19k"] = 30
    df["V2G_cap_19k_tou"] = (abs(df["V2G_SOC_tou_19k"] - (df["battery[soc][end][charging]"])) / 100) * df["bat_cap"]

    return df


def v2g_tou_cap_40(df):
    # Apply the function to each row and add the results as new columns
    df[['discharge_start', 'discharge_end', 'charge_start', 'charge_end', "next_departure_time1"]] = df.apply(calculate_v2g, axis=1, result_type='expand')

    df["V2G_time_min"] = (df["discharge_end"] - df["discharge_start"]).dt.seconds / 60
    df["V2G_time_charge"] = (df["charge_end"] - df["charge_start"]).dt.seconds / 60

    df["V2G_SOC_tou_6k"] = ((((df["battery[soc][end][charging]"] / 100) * df["bat_cap"]) - ((df["V2G_time_min"] / 60) * 6.6)) / (df["bat_cap"])) * 100
    df.loc[df["V2G_SOC_tou_6k"] < 40, "V2G_SOC_tou_6k"] = 40
    df["V2G_cap_6k_tou"] = (abs(df["V2G_SOC_tou_6k"] - (df["battery[soc][end][charging]"])) / 100) * df["bat_cap"]

    df["V2G_SOC_tou_12k"] = ((((df["battery[soc][end][charging]"] / 100) * df["bat_cap"]) - ((df["V2G_time_min"] / 60) * 12)) / (df["bat_cap"])) * 100
    df.loc[df["V2G_SOC_tou_12k"] < 40, "V2G_SOC_tou_12k"] = 40
    df["V2G_cap_12k_tou"] = (abs(df["V2G_SOC_tou_12k"] - (df["battery[soc][end][charging]"])) / 100) * df["bat_cap"]

    df["V2G_SOC_tou_19k"] = ((((df["battery[soc][end][charging]"] / 100) * df["bat_cap"]) - ((df["V2G_time_min"] / 60) * 19)) / (df["bat_cap"])) * 100
    df.loc[df["V2G_SOC_tou_19k"] < 40, "V2G_SOC_tou_19k"] = 40
    df["V2G_cap_19k_tou"] = (abs(df["V2G_SOC_tou_19k"] - (df["battery[soc][end][charging]"])) / 100) * df["bat_cap"]

    return df


def v2g_tou_cap_50(df):
    # Apply the function to each row and add the results as new columns
    df[['discharge_start', 'discharge_end', 'charge_start', 'charge_end', "next_departure_time1"]] = df.apply(calculate_v2g, axis=1, result_type='expand')

    df["V2G_time_min"] = (df["discharge_end"] - df["discharge_start"]).dt.seconds / 60
    df["V2G_time_charge"] = (df["charge_end"] - df["charge_start"]).dt.seconds / 60

    df["V2G_SOC_tou_6k"] = ((((df["battery[soc][end][charging]"] / 100) * df["bat_cap"]) - ((df["V2G_time_min"] / 60) * 6.6)) / (df["bat_cap"])) * 100
    df.loc[df["V2G_SOC_tou_6k"] < 50, "V2G_SOC_tou_6k"] = 50
    df["V2G_cap_6k_tou"] = (abs(df["V2G_SOC_tou_6k"] - (df["battery[soc][end][charging]"])) / 100) * df["bat_cap"]

    df["V2G_SOC_tou_12k"] = ((((df["battery[soc][end][charging]"] / 100) * df["bat_cap"]) - ((df["V2G_time_min"] / 60) * 12)) / (df["bat_cap"])) * 100
    df.loc[df["V2G_SOC_tou_12k"] < 50, "V2G_SOC_tou_12k"] = 50
    df["V2G_cap_12k_tou"] = (abs(df["V2G_SOC_tou_12k"] - (df["battery[soc][end][charging]"])) / 100) * df["bat_cap"]

    df["V2G_SOC_tou_19k"] = ((((df["battery[soc][end][charging]"] / 100) * df["bat_cap"]) - ((df["V2G_time_min"] / 60) * 19)) / (df["bat_cap"])) * 100
    df.loc[df["V2G_SOC_tou_19k"] < 50, "V2G_SOC_tou_19k"] = 50
    df["V2G_cap_19k_tou"] = (abs(df["V2G_SOC_tou_19k"] - (df["battery[soc][end][charging]"])) / 100) * df["bat_cap"]

    return df


##################################################################################################################
##################################################################################################################
def v2g_tou_trip_buffer(df):
    # Apply the function to each row and add the results as new columns
    df[['discharge_start', 'discharge_end', 'charge_start', 'charge_end', "next_departure_time1"]] = df.apply(calculate_v2g, axis=1, result_type='expand')

    df["V2G_time_min"] = (df["discharge_end"] - df["discharge_start"]).dt.seconds / 60
    df["V2G_time_charge"] = (df["charge_end"] - df["charge_start"]).dt.seconds / 60

    # df["V2G_SOC_tou_6k"] = ((((df["battery[soc][end][charging]"] / 100) * df["bat_cap"]) - ((df["V2G_time_min"] / 60) * 6.6) - ((df["V2G_time_charge"] / 60) * 6.6)) / (df["bat_cap"])) * 100
    df["V2G_SOC_tou_6k"] = ((((df["battery[soc][end][charging]"] / 100) * df["bat_cap"]) - ((df["V2G_time_min"] / 60) * 6.6)) / (df["bat_cap"])) * 100

    df.loc[df["V2G_SOC_tou_6k"] < 0, "V2G_SOC_tou_6k"] = 0
    df.loc[df["V2G_SOC_tou_6k"] < df["SOC_next_trip"], "V2G_SOC_tou_6k"] = df["SOC_next_trip"]
    df["V2G_cap_6k_tou"] = (abs(df["V2G_SOC_tou_6k"] - (df["battery[soc][end][charging]"])) / 100) * df["bat_cap"]

    # df["V2G_SOC_tou_12k"] = ((((df["battery[soc][end][charging]"] / 100) * df["bat_cap"]) - ((df["V2G_time_min"] / 60) * 12) - ((df["V2G_time_charge"] / 60) * 12)) / (df["bat_cap"])) * 100
    df["V2G_SOC_tou_12k"] = ((((df["battery[soc][end][charging]"] / 100) * df["bat_cap"]) - ((df["V2G_time_min"] / 60) * 12)) / (df["bat_cap"])) * 100

    df.loc[df["V2G_SOC_tou_12k"] < 0, "V2G_SOC_tou_12k"] = 0
    df.loc[df["V2G_SOC_tou_12k"] < df["SOC_next_trip"], "V2G_SOC_tou_12k"] = df["SOC_next_trip"]
    df["V2G_cap_12k_tou"] = (abs(df["V2G_SOC_tou_12k"] - (df["battery[soc][end][charging]"])) / 100) * df["bat_cap"]

    # df["V2G_SOC_tou_19k"] = ((((df["battery[soc][end][charging]"] / 100) * df["bat_cap"]) - ((df["V2G_time_min"] / 60) * 19) - ((df["V2G_time_charge"] / 60) * 19)) / (df["bat_cap"])) * 100
    df["V2G_SOC_tou_19k"] = ((((df["battery[soc][end][charging]"] / 100) * df["bat_cap"]) - ((df["V2G_time_min"] / 60) * 19)) / (df["bat_cap"])) * 100

    df.loc[df["V2G_SOC_tou_19k"] < 0, "V2G_SOC_tou_19k"] = 0
    df.loc[df["V2G_SOC_tou_19k"] < df["SOC_next_trip"], "V2G_SOC_tou_19k"] = df["SOC_next_trip"]
    df["V2G_cap_19k_tou"] = (abs(df["V2G_SOC_tou_19k"] - (df["battery[soc][end][charging]"])) / 100) * df["bat_cap"]

    # Apply the function to each row and add the results as new columns
    return df


def v2g_tou_charging_buffer(df):
    df[['discharge_start', 'discharge_end', 'charge_start', 'charge_end', "next_departure_time1"]] = df.apply(calculate_v2g, axis=1, result_type='expand')

    df["V2G_time_min"] = (df["discharge_end"] - df["discharge_start"]).dt.seconds / 60
    df["V2G_time_charge"] = (df["charge_end"] - df["charge_start"]).dt.seconds / 60

    # df["V2G_SOC_tou_6k"] = ((((df["battery[soc][end][charging]"] / 100) * df["bat_cap"]) - ((df["V2G_time_min"] / 60) * 6.6) - ((df["V2G_time_charge"] / 60) * 6.6)) / (df["bat_cap"])) * 100
    df["V2G_SOC_tou_6k"] = ((((df["battery[soc][end][charging]"] / 100) * df["bat_cap"]) - ((df["V2G_time_min"] / 60) * 6.6)) / (df["bat_cap"])) * 100

    df.loc[df["V2G_SOC_tou_6k"] < 0, "V2G_SOC_tou_6k"] = 0
    df.loc[df["V2G_SOC_tou_6k"] < df["SOC_need_next_charge"], "V2G_SOC_tou_6k"] = df["SOC_need_next_charge"]
    df["V2G_cap_6k_tou"] = (abs(df["V2G_SOC_tou_6k"] - (df["battery[soc][end][charging]"])) / 100) * df["bat_cap"]

    # df["V2G_SOC_tou_12k"] = ((((df["battery[soc][end][charging]"] / 100) * df["bat_cap"]) - ((df["V2G_time_min"] / 60) * 12) - ((df["V2G_time_charge"] / 60) * 12)) / (df["bat_cap"])) * 100
    df["V2G_SOC_tou_12k"] = ((((df["battery[soc][end][charging]"] / 100) * df["bat_cap"]) - ((df["V2G_time_min"] / 60) * 12)) / (df["bat_cap"])) * 100

    df.loc[df["V2G_SOC_tou_12k"] < 0, "V2G_SOC_tou_12k"] = 0
    df.loc[df["V2G_SOC_tou_12k"] < df["SOC_need_next_charge"], "V2G_SOC_tou_12k"] = df["SOC_need_next_charge"]
    df["V2G_cap_12k_tou"] = (abs(df["V2G_SOC_tou_12k"] - (df["battery[soc][end][charging]"])) / 100) * df["bat_cap"]

    # df["V2G_SOC_tou_19k"] = ((((df["battery[soc][end][charging]"] / 100) * df["bat_cap"]) - ((df["V2G_time_min"] / 60) * 19) - ((df["V2G_time_charge"] / 60) * 19)) / (df["bat_cap"])) * 100
    df["V2G_SOC_tou_19k"] = ((((df["battery[soc][end][charging]"] / 100) * df["bat_cap"]) - ((df["V2G_time_min"] / 60) * 19)) / (df["bat_cap"])) * 100

    df.loc[df["V2G_SOC_tou_19k"] < 0, "V2G_SOC_tou_19k"] = 0
    df.loc[df["V2G_SOC_tou_19k"] < df["SOC_need_next_charge"], "V2G_SOC_tou_19k"] = df["SOC_need_next_charge"]
    df["V2G_cap_19k_tou"] = (abs(df["V2G_SOC_tou_19k"] - (df["battery[soc][end][charging]"])) / 100) * df["bat_cap"]

    return df



##################################################################################################################
##################################################################################################################

def storage_cap_tou(df):
    df = df.copy()
    V2G_hourly_tou = pd.DataFrame(index=df.index, columns=range(24))
    V2G_hourly_12_tou = V2G_hourly_tou.fillna(0)
    V2G_hourly_6_tou = V2G_hourly_12_tou.copy()
    V2G_hourly_19_tou = V2G_hourly_12_tou.copy()
    for i in df.index:
        start_hour = (df.loc[i, "discharge_start"].hour)
        end_hour = df.loc[i, "discharge_end"].hour
        discharging_speed = 12
        total_capacity = df.loc[i, "V2G_cap_12k_tou"]
        current_hour = start_hour
        remaining_capacity = total_capacity
        while remaining_capacity > 0 and current_hour < end_hour:
            discharge_amount = min(discharging_speed, remaining_capacity)
            V2G_hourly_12_tou.loc[i, current_hour] += discharge_amount
            remaining_capacity -= discharge_amount
            if current_hour < 21:
                current_hour += 1
            else:
                break
    for i in df.index:
        start_hour = (df.loc[i, "discharge_start"].hour)
        end_hour = df.loc[i, "discharge_end"].hour
        discharging_speed = 6.6
        total_capacity = df.loc[i, "V2G_cap_6k_tou"]
        current_hour = start_hour
        remaining_capacity = total_capacity
        while remaining_capacity > 0 and current_hour < end_hour:
            discharge_amount = min(discharging_speed, remaining_capacity)
            V2G_hourly_6_tou.loc[i, current_hour] += discharge_amount
            remaining_capacity -= discharge_amount
            if current_hour < 21:
                current_hour += 1
            else:
                break
    for i in df.index:
        start_hour = (df.loc[i, "discharge_start"].hour)
        end_hour = df.loc[i, "discharge_end"].hour
        discharging_speed = 19
        total_capacity = df.loc[i, "V2G_cap_19k_tou"]
        current_hour = start_hour
        remaining_capacity = total_capacity
        while remaining_capacity > 0 and current_hour < end_hour:
            discharge_amount = min(discharging_speed, remaining_capacity)
            V2G_hourly_19_tou.loc[i, current_hour] += discharge_amount
            remaining_capacity -= discharge_amount
            if current_hour < 21:
                current_hour += 1
            else:
                break
    V2G_hourly_12_tou = pd.merge(df[["month", "day"]], V2G_hourly_12_tou, how="left", left_index=True, right_index=True)
    V2G_hourly_12_tou_sum = V2G_hourly_12_tou.groupby(["month", "day"]).sum()
    V2G_hourly_6_tou = pd.merge(df[["month", "day"]], V2G_hourly_6_tou, how="left", left_index=True, right_index=True)
    V2G_hourly_6_tou_sum = V2G_hourly_6_tou.groupby(["month", "day"]).sum()
    V2G_hourly_19_tou = pd.merge(df[["month", "day"]], V2G_hourly_19_tou, how="left", left_index=True, right_index=True)
    V2G_hourly_19_tou_sum = V2G_hourly_19_tou.groupby(["month", "day"]).sum()
    return V2G_hourly_12_tou, V2G_hourly_6_tou, V2G_hourly_19_tou, V2G_hourly_12_tou_sum, V2G_hourly_6_tou_sum, V2G_hourly_19_tou_sum



##################################################################################################################
##################################################################################################################

def storage_cap_tou_sta(df):
    df = df.copy()
    df = df.copy()
    V2G_hourly_tou = pd.DataFrame(index=df.index, columns=range(24))
    V2G_hourly_12_tou = V2G_hourly_tou.fillna(0)
    V2G_hourly_6_tou = V2G_hourly_12_tou.copy()
    V2G_hourly_19_tou = V2G_hourly_12_tou.copy()
    for i in df.index:
        start_hour = df.loc[i, "discharge_start"].hour -1
        end_hour = df.loc[i, "discharge_end"].hour
        discharging_speed = 12
        total_capacity = df.loc[i, "V2G_cap_12k"]
        current_hour = start_hour
        remaining_capacity = total_capacity
        while remaining_capacity > 0 and current_hour < end_hour:
            discharge_amount = min(discharging_speed, remaining_capacity)
            V2G_hourly_12_tou.loc[i, current_hour] += discharge_amount
            remaining_capacity -= discharge_amount
            if current_hour < 21:
                current_hour += 1
            else:
                break
    for i in df.index:
        start_hour = df.loc[i, "discharge_start"].hour - 1
        end_hour = df.loc[i, "discharge_end"].hour
        discharging_speed = 6.6
        total_capacity = df.loc[i, "V2G_cap_6k"]
        current_hour = start_hour
        remaining_capacity = total_capacity
        while remaining_capacity > 0 and current_hour < end_hour:
            discharge_amount = min(discharging_speed, remaining_capacity)
            V2G_hourly_6_tou.loc[i, current_hour] += discharge_amount
            remaining_capacity -= discharge_amount
            if current_hour < 21:
                current_hour += 1
            else:
                break
    for i in df.index:
        start_hour = df.loc[i, "discharge_start"].hour - 1
        end_hour = df.loc[i, "discharge_end"].hour
        discharging_speed = 19
        total_capacity = df.loc[i, "V2G_cap_19k"]
        current_hour = start_hour
        remaining_capacity = total_capacity
        while remaining_capacity > 0 and current_hour < end_hour:
            discharge_amount = min(discharging_speed, remaining_capacity)
            V2G_hourly_19_tou.loc[i, current_hour] += discharge_amount
            remaining_capacity -= discharge_amount
            if current_hour < 21:
                current_hour += 1
            else:
                break
    V2G_hourly_12_tou = pd.merge(df[["day"]], V2G_hourly_12_tou, how="left", left_index=True, right_index=True)
    V2G_hourly_12_tou_sum = V2G_hourly_12_tou.groupby(["day"]).sum()
    V2G_hourly_6_tou = pd.merge(df[["day"]], V2G_hourly_6_tou, how="left", left_index=True, right_index=True)
    V2G_hourly_6_tou_sum = V2G_hourly_6_tou.groupby(["day"]).sum()
    V2G_hourly_19_tou = pd.merge(df[["day"]], V2G_hourly_19_tou, how="left", left_index=True, right_index=True)
    V2G_hourly_19_tou_sum = V2G_hourly_19_tou.groupby(["day"]).sum()
    return V2G_hourly_12_tou, V2G_hourly_6_tou, V2G_hourly_19_tou, V2G_hourly_12_tou_sum, V2G_hourly_6_tou_sum, V2G_hourly_19_tou_sum

##################################################################################################################
##################################################################################################################
def extra_extra_kwh(df):
    # Divide values by 1000 to convert kWh to MWh
    # Sort the DataFrame based on the sum of each row
    test0_MWh_sorted = df.sum(axis=1).sort_values().index
    test0_MWh_sorted_df = df.loc[test0_MWh_sorted]
    test0_MWh_sorted_df = test0_MWh_sorted_df/1000
    columns_to_include_reversed = test0_MWh_sorted_df.columns[:-4][::-1]

    # Define colors for each bar
    colors = ['orange', 'red', 'blue', 'green']

    plt.figure(figsize=(12, 8))

    # Plot each bar separately without stacking
    for i, column in enumerate(columns_to_include_reversed):
        plt.bar(test0_MWh_sorted_df.index, test0_MWh_sorted_df[column], color=colors[i], label=column)

    plt.xlabel('Vehicles', fontsize=14)
    plt.ylabel('MWh', fontsize=14)
    plt.title('Energy Consumption During Driving and V2G (MWh)', fontsize=16)
    plt.legend(title='V2G Speeds', fontsize=18)
    plt.xticks(rotation=90)
    plt.yticks(fontsize=12)  # Set font size for y-axis ticks
    plt.tight_layout()
    plt.ylim(0, 50)
    plt.grid(axis='y', alpha=0.5)
    plt.show()



def extra_extra_kwh_sta(df):
    # Divide values by 1000 to convert kWh to MWh
    # Sort the DataFrame based on the sum of each row
    test0_MWh_sorted_df = df.sort_values(by='19 kW')
    test0_MWh_sorted_df = test0_MWh_sorted_df
    columns_to_include_reversed = test0_MWh_sorted_df.columns[::-1]

    # Define colors for each bar
    colors = ['orange', 'red', 'blue']

    plt.figure(figsize=(12, 8))

    # Plot each bar separately without stacking
    for i, column in enumerate(columns_to_include_reversed):
        plt.bar(test0_MWh_sorted_df.index, (test0_MWh_sorted_df[column]/365), color=colors[i], label=column)

    plt.xlabel('Vehicles', fontsize=14)
    plt.ylabel('Energy Storage per Day (kWh)', fontsize=14)
    plt.title('Energy Consumption During Driving and V2G (kWh)', fontsize=16)
    plt.legend(title='V2G Speeds', fontsize=18)
    plt.xticks(rotation=90)
    plt.yticks(fontsize=12)  # Set font size for y-axis ticks
    plt.tight_layout()
    plt.ylim(0, 100)
    plt.grid(axis='y', alpha=0.5)
    plt.show()

#
# def extra_extra_kwh_parking(df):
#     # Assume df is your DataFrame and it contains 'Observation_days' column
#     # Convert kWh to MWh and sort the DataFrame based on the sum of each row
#     df.rename(columns={'charging_cap': 'Driving'}, inplace=True)
#     df_sorted = df.sort_values(by='Driving')
#     df_sorted.set_index('vehicle_name', inplace=True)
#     # Define columns to plot (excluding 'Observation_days' and 'Sum_MWh')
#     columns_to_plot = df_sorted.columns[:-6][::-1] # Adjust based on your DataFrame structure
#
#     # Define colors for each bar
#     colors = ['orange', 'red', 'blue', 'green']
#
#     plt.figure(figsize=(12, 8))
#     ax1 = plt.gca()  # Get current axis for the bar plot
#
#     # Plot each bar separately without stacking
#     for i, column in enumerate(columns_to_plot):  # Reversed order
#         ax1.bar(df_sorted.index, df_sorted[column], color=colors[i % len(colors)], label=column)
#
#     # Setting the primary y-axis (left) labels and title
#     ax1.set_xlabel('Vehicles', fontsize=14)
#     ax1.set_ylabel('Average Energy Consumption per Day - kWh', fontsize=14)
#     ax1.set_title('Energy Consumption During Driving and V2G (MWh)', fontsize=16)
#     ax1.legend(title='V2G Speeds', fontsize=12)
#     ax1.set_xticklabels(df_sorted.index, rotation=90)
#     ax1.set_ylim(0, 100)  # Adjust based on your data
#     ax1.grid(axis='y', alpha=0.5)
#     ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), shadow=True, ncol=2)
#     # Create a second y-axis for the line plot
#     ax2 = ax1.twinx()
#     ax2.plot(df_sorted.index.to_numpy(), df_sorted['Average Daily VMT'].to_numpy(), color='orange', label='Average Daily VMT', linewidth=2, marker='o')
#     # ax2.set_ylabel('Observation Days / Battery Capacity ', fontsize=14)  # Label for the second y-axis
#     ax2.set_ylim(0, 150)  # Adjust based on your data
#     ax2.legend(loc='upper left', fontsize=12)
#
#     ax3 = ax1.twinx()
#     ax3.plot(df_sorted.index.to_numpy(), df_sorted['bat_cap'].to_numpy(), color='red', label='Battery Capacity', linewidth=2, marker='o')
#     ax3.set_ylabel('Battery Capacity (kWh) / Daily VMT', fontsize=14)
#     # ax3.set_ylabel('Battery Capacity', fontsize=14)  # Label for the second y-axis
#     ax3.set_ylim(0, 150)  # Adjust based on your data
#     ax3.legend(loc='upper center', fontsize=12)
#
#     plt.tight_layout()
#     plt.show()


def extra_extra_kwh_parking(df):
    # Assume df is your DataFrame and it contains 'Observation_days' column
    # Convert kWh to MWh and sort the DataFrame based on the sum of each row
    df_sorted = df.sort_values(by='Driving')
    df_sorted.set_index('vehicle_name', inplace=True)

    # Define colors for each bar
    colors = ['orange', 'red', 'blue', 'green']

    # Create first plot for the first column
    plt.figure(figsize=(12, 10))
    ax1 = plt.subplot(211)
    ax1 = plt.gca()  # Get current axis for the bar plot
    ax1.bar(df_sorted.index, df_sorted[df_sorted.columns[0]], color=colors[0], label=df_sorted.columns[0])

    # Setting the primary y-axis (left) labels and title for the first plot
    ax1.set_xlabel('Vehicles', fontsize=14)
    ax1.set_ylabel('Average Battery Usage\n per Driving Day - %', fontsize=14)
    ax1.set_title('Driving mode', fontsize=16)
    ax1.legend(title='Energy Consumption', fontsize=12)
    ax1.set_xticklabels(df_sorted.index, rotation=90)
    ax1.set_ylim(0, 100)  # Adjust based on your data
    ax1.grid(axis='y', alpha=0.5)

    # Create second plot for columns 3, 4, and 5
    ax2 = plt.subplot(212)
    ax2 = plt.gca()  # Get current axis for the bar plot
    for i, column in enumerate(df_sorted.columns[1:4][::-1]):
        ax2.bar(df_sorted.index, df_sorted[column], color=colors[i + 1], label=column)

    # Setting the primary y-axis (left) labels and title for the second plot
    ax2.set_xlabel('Vehicles', fontsize=14)
    ax2.set_ylabel('Average Available power\n per Plugged-in Day - %', fontsize=14)
    ax2.set_title('V2G mode', fontsize=16)
    ax2.set_xticklabels(df_sorted.index, rotation=90)
    ax2.set_ylim(0, 100)  # Adjust based on your data
    ax2.grid(axis='y', alpha=0.5)
    ax2.legend(title='Energy Consumption', fontsize=12, loc='upper center', bbox_to_anchor=(0.5, -0.35), shadow=True, ncol=2)

    ax3 = ax2.twinx()
    ax3.plot(df_sorted.index.to_numpy(), df_sorted['bat_cap'].to_numpy(), color='red', label='Battery Capacity', linewidth=2, marker='o')
    ax3.set_ylabel('Battery Capacity (kWh)', fontsize=14)
    # ax3.set_ylabel('Battery Capacity', fontsize=14)  # Label for the second y-axis
    ax3.set_ylim(0, 120)  # Adjust based on your data
    ax3.legend(loc='upper right', fontsize=12)

    plt.tight_layout()
    plt.show()

##################################################################################################################
##################################################################################################################
def extra_extra_cycle(df):
    # Divide values by 1000 to convert kWh to MWh
    # Sort the DataFrame based on the sum of each row
    test0_MWh_sorted = df.sum(axis=1).sort_values().index
    test0_MWh_sorted_df = df.loc[test0_MWh_sorted]
    columns_to_include_reversed = test0_MWh_sorted_df.columns[-2:][::-1]

    # Define colors for each bar
    colors = ['orange', 'green']

    plt.figure(figsize=(12, 8))

    # Plot each bar separately without stacking
    for i, column in enumerate(columns_to_include_reversed):
        plt.bar(test0_MWh_sorted_df.index, test0_MWh_sorted_df[column], color=colors[i], label=column)

    plt.xlabel('Vehicles', fontsize=14)
    plt.ylabel('#', fontsize=14)
    plt.title('Number of Charging Cycle by Vehicles and V2G Speeds', fontsize=16)
    plt.legend(title='V2G Speeds', fontsize=12)
    plt.xticks(rotation=90)
    plt.yticks(fontsize=12)  # Set font size for y-axis ticks
    plt.tight_layout()
    plt.ylim(0, 700)
    plt.grid(axis='y', alpha=0.5)
    plt.show()


##############################################################################################################################################
##############################################################################################################################################
def total_v2g_cap_graph(df, df1):

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 6))  # Making the figure wider
    bar_width = 0.5
    num_cols = len(df.columns)
    index = np.arange(len(df))

    for i, col in enumerate(df.columns):
        ax.bar(index + i * bar_width / num_cols, df[col], bar_width / num_cols, label=col)  # Adjust bar position

    ax.set_xlabel('V2G Charging Speed', fontsize=18)
    ax.set_ylabel('V2G Capacity per Total Stationary Capacity %', fontsize=14)
    ax.set_title('Annual V2G Storage Capacity', fontsize=18)
    ax.set_xticks([x + 0.3 * bar_width for x in index])  # Adjusting the x-ticks position
    ax.set_xticklabels(df.index, rotation=-0, ha='left', fontsize=18)  # Rotating the x-ticks to -45 degrees and aligning to the left
    ax.set_yticks(ax.get_yticks())
    ax.set_yticklabels(ax.get_yticks(), fontsize=16)
    ax.grid(axis='y', alpha=0.5)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), shadow=True, ncol=3)
    plt.tight_layout()
    # plt.show()
    #
    # ax2 = ax.twinx()
    # num_cols_df1 = len(df1.columns)  # Number of columns in df1
    # for i, col1 in enumerate(df1.columns):
    #     ax2.bar(index + (i + num_cols) * bar_width / num_cols_df1, df1[col1], bar_width / num_cols_df1, label=col1)  # Adjust bar position
    #
    # # Setting the primary y-axis (left) labels and title for the second plot
    # ax2.set_ylabel('Annual V2G Capacity of the Fleet - MWh', fontsize=14)

    plt.tight_layout()
    plt.show()



def total_v2g_cap_graph1(df, df1):

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 6))  # Making the figure wider
    bar_width = 0.5
    num_cols = len(df.columns)
    index = np.arange(len(df))

    for i, col in enumerate(df.columns):
        ax.bar(index + i * bar_width / num_cols, df[col], bar_width / num_cols, label=col)  # Adjust bar position

    ax.set_xlabel('V2G Charging Speed', fontsize=18)
    ax.set_ylabel('Total V2G Storage Capacity (MWh)', fontsize=14)
    ax.set_title('Total V2G Storage Capacity', fontsize=18)
    ax.set_xticks([x + 0.3 * bar_width for x in index])  # Adjusting the x-ticks position
    ax.set_xticklabels(df.index, rotation=-0, ha='left', fontsize=18)  # Rotating the x-ticks to -45 degrees and aligning to the left
    ax.set_yticks(ax.get_yticks())
    ax.set_yticklabels(ax.get_yticks(), fontsize=16)
    ax.grid(axis='y', alpha=0.5)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), shadow=True, ncol=3)
    plt.tight_layout()
    # plt.show()
    #
    # ax2 = ax.twinx()
    # num_cols_df1 = len(df1.columns)  # Number of columns in df1
    # for i, col1 in enumerate(df1.columns):
    #     ax2.bar(index + (i + num_cols) * bar_width / num_cols_df1, df1[col1], bar_width / num_cols_df1, label=col1)  # Adjust bar position
    #
    # # Setting the primary y-axis (left) labels and title for the second plot
    # ax2.set_ylabel('Annual V2G Capacity of the Fleet - MWh', fontsize=14)

    plt.tight_layout()
    plt.show()

def total_v2g_cap_graph_base(df):
    # Plotting
    fig, ax = plt.subplots(figsize=(12, 6))  # Making the figure wider
    bar_width = 0.5
    num_cols = len(df.columns)
    index = np.arange(len(df))

    for i, col in enumerate(df.columns):
        ax.bar(index + i * bar_width / num_cols, df[col], bar_width / num_cols, label=col)  # Adjust bar position

    ax.set_xlabel('V2G Charging Speed', fontsize=18)
    ax.set_ylabel('Annual V2G Capacity - MWh', fontsize=14)
    ax.set_title('Annual V2G Storage Capacity', fontsize=18)
    ax.set_xticks([x + 0.3 * bar_width for x in index])  # Adjusting the x-ticks position
    ax.set_xticklabels(df.index, rotation=-0, ha='left', fontsize=18)  # Rotating the x-ticks to -45 degrees and aligning to the left
    ax.set_yticks(ax.get_yticks())
    ax.set_yticklabels(ax.get_yticks(), fontsize=16)
    ax.grid(axis='y', alpha=0.5)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), shadow=True, ncol=3)
    plt.tight_layout()
    plt.show()
##############################################################################################################################################
##############################################################################################################################################


def total_v2g_failt_graph(df):
    df = df.T
    fig, ax = plt.subplots(figsize=(12, 6))  # Making the figure wider
    bar_width = 0.2
    index = range(len(df))

    for i, col in enumerate(df.columns):
        ax.bar([x + i * bar_width for x in index], df[col], bar_width, label=col)

    ax.set_xlabel('V2G TOU Scenarios', fontsize=18)
    ax.set_ylabel('% of Failure', fontsize=18)
    ax.set_title('Impact of V2G on Subsequent Trip Success', fontsize=18)
    ax.set_xticks([x + 1.5 * bar_width for x in index])  # Adjusting the x-ticks position
    ax.set_xticklabels(df.index, rotation=-45, ha='left')  # Rotating the x-ticks to -45 degrees and aligning to the left
    ax.set_ylim(0, 12)  # Setting a limit for the y-axis
    ax.grid(axis='y', alpha=0.5)
    ax.legend()

    plt.tight_layout()  # Adjusting the layout to ensure all elements are properly displayed
    plt.show()


def total_v2g_failc_graph(df):
    df = df.T
    fig, ax = plt.subplots(figsize=(12, 6))  # Making the figure wider
    bar_width = 0.2
    index = range(len(df))

    for i, col in enumerate(df.columns):
        ax.bar([x + i * bar_width for x in index], df[col], bar_width, label=col)

    ax.set_xlabel('V2G Scenarios', fontsize=18)
    ax.set_ylabel('% of Failure', fontsize=18)
    ax.set_title(' Impact of V2G on Reaching Next Charging Event', fontsize=18)
    ax.set_xticks([x + 1.5 * bar_width for x in index])  # Adjusting the x-ticks position
    ax.set_xticklabels(df.index, rotation=-45, ha='left')  # Rotating the x-ticks to -45 degrees and aligning to the left
    ax.set_ylim(0, 20)  # Setting a limit for the y-axis
    ax.grid(axis='y', alpha=0.5)
    ax.legend()

    plt.tight_layout()  # Adjusting the layout to ensure all elements are properly displayed
    plt.show()


##############################################################################################################################################
##############################################################################################################################################

def v2g_fail(df):
    # Filter rows where charging duration is not NaN

    df1 = df.copy()
    # Calculate minimum range for different scenarios
    df1["minrange_6k"] = (df1["bat_cap"] * (df1["SOC_after_char_V2G_6k"] / 100)) / 0.28
    df1["minrange_12k"] = (df1["bat_cap"] * (df1["SOC_after_char_V2G_12k"] / 100)) / 0.28
    df1["minrange_19k"] = (df1["bat_cap"] * (df1["SOC_after_char_V2G_19k"] / 100)) / 0.28

    df1["minrange_need"] = (df1["bat_cap"] * (df1["SOC_next_trip"] / 100)) / 0.28
    df1["minrange_need_nextc"] = (df1["bat_cap"] * (df1["SOC_need_next_charge"] / 100)) / 0.28

    # next trip fail indicator
    df1.loc[:, "next_trip_fail_6"] = df1.loc[:, "minrange_6k"] < df1.loc[:, "minrange_need"]
    # next charging sessions fail indicator
    df1.loc[:, "next_c_fail_6"] = df1.loc[:, "minrange_6k"] < df1.loc[:, "minrange_need_nextc"]

    df1.loc[:, "next_trip_fail_12"] = df1.loc[:, "minrange_12k"] < df1.loc[:, "minrange_need"]
    # next charging sessions fail indicator
    df1.loc[:, "next_c_fail_12"] = df1.loc[:, "minrange_12k"] < df1.loc[:, "minrange_need_nextc"]

    df1.loc[:, "next_trip_fail_19"] = df1.loc[:, "minrange_19k"] < df1.loc[:, "minrange_need"]
    # next charging sessions fail indicator
    df1.loc[:, "next_c_fail_19"] = df1.loc[:, "minrange_19k"] < df1.loc[:, "minrange_need_nextc"]

    # Calculate ratio for "next_trip_fail_6" with zero fill
    ratio6_nt = df1["next_trip_fail_6"].value_counts(normalize=True)
    ratio6_nt = ratio6_nt.reindex([True, False], fill_value=0)

    # Calculate ratio for "next_c_fail_6" with zero fill
    ratio6_nc = df1["next_c_fail_6"].value_counts(normalize=True)
    ratio6_nc = ratio6_nc.reindex([True, False], fill_value=0)

    # Calculate ratio for "next_trip_fail_12" with zero fill
    ratio12_nt = df1["next_trip_fail_12"].value_counts(normalize=True)
    ratio12_nt = ratio12_nt.reindex([True, False], fill_value=0)

    # Calculate ratio for "next_c_fail_12" with zero fill
    ratio12_nc = df1["next_c_fail_12"].value_counts(normalize=True)
    ratio12_nc = ratio12_nc.reindex([True, False], fill_value=0)

    # Calculate ratio for "next_trip_fail_19" with zero fill
    ratio19_nt = df1["next_trip_fail_19"].value_counts(normalize=True)
    ratio19_nt = ratio19_nt.reindex([True, False], fill_value=0)

    # Calculate ratio for "next_c_fail_19" with zero fill
    ratio19_nc = df1["next_c_fail_19"].value_counts(normalize=True)
    ratio19_nc = ratio19_nc.reindex([True, False], fill_value=0)

    data = {'6': [ratio6_nt[1] * 100, ratio6_nc[1] * 100],
            '12': [ratio12_nt[1] * 100, ratio12_nc[1] * 100],
            '19': [ratio19_nt[1] * 100, ratio19_nc[1] * 100]}

    data = pd.DataFrame(data, index=['next_trip', 'next_charging']).T

    return data

##############################################################################################################################################
##############################################################################################################################################
def v2g_participate(df):
    # df = v2g_tou.copy()
    df["discharge_end1"] = pd.to_datetime(df["discharge_end"])
    df["discharge_start1"] = pd.to_datetime(df["discharge_start"])
    df["charge_end1"] = pd.to_datetime(df["charge_end"])
    df["charge_start1"] = pd.to_datetime(df["charge_start"])

    df["SOC_after_char_V2G_6k"] = (df["V2G_SOC_tou_6k"] + (((((df["charge_end1"] - df["charge_start1"]).dt.seconds/3600) * 6.6)/df["bat_cap"])*100))
    df["SOC_after_char_V2G_12k"] = (df["V2G_SOC_tou_12k"] + (((((df["charge_end1"] - df["charge_start1"]).dt.seconds/3600) * 12)/df["bat_cap"])*100))
    df["SOC_after_char_V2G_19k"] = (df["V2G_SOC_tou_19k"] + (((((df["charge_end1"] - df["charge_start1"]).dt.seconds/3600) * 19)/df["bat_cap"])*100))

    df["V2G_participate"] = False
    df.loc[(df["discharge_end1"] - df["discharge_start1"]).dt.seconds > 0, "V2G_participate"] = True

    return df


##############################################################################################################################################
##############################################################################################################################################
def parking_sessions(df):
    # df = final_dataframes.copy()
    parking_dataframe = df.groupby(['vehicle_name', "year", "month", "day"]).tail(n=1)
    parking_dataframe = parking_dataframe.sort_values(by=["vehicle_name", "year", "month", "day"])

    # Convert 'year', 'month', and 'day' columns to datetime
    parking_dataframe['date'] = pd.to_datetime(parking_dataframe[['year', 'month', 'day']])

    # Sort the dataframe by 'vehicle_name' and 'date'
    parking_dataframe = parking_dataframe.sort_values(by=['vehicle_name', 'date']).reset_index(drop=True)

    # Create an empty DataFrame to store the result
    result_df = pd.DataFrame(columns=parking_dataframe.columns)

    # Iterate over each group of vehicle_name
    for vehicle_name, group in parking_dataframe.groupby('vehicle_name'):
        # Calculate the expected date range
        expected_dates = pd.date_range(start=group['date'].min(), end=group['date'].max(), freq='D')
        # Find the missing dates
        missing_dates = expected_dates.difference(group['date'])
        # Create NaN rows for missing dates and append them to the result DataFrame
        if not missing_dates.empty:
            nan_rows = pd.DataFrame({'vehicle_name': vehicle_name,
                                     'year': missing_dates.year,
                                     'month': missing_dates.month,
                                     'day': missing_dates.day,
                                     'indicator_column': np.nan})
            result_df = pd.concat([result_df, group, nan_rows]).sort_values(by='date').reset_index(drop=True)
        else:
            result_df = pd.concat([result_df, group]).reset_index(drop=True)

    # Drop the 'date' column if no longer needed
    result_df.drop(columns=['date'], inplace=True)

    result_df = result_df.sort_values(by=["vehicle_name", "year", "month", "day"]).reset_index(drop=True)
    result_df["indicator_column"] = False
    result_df.loc[result_df["vehicle_model"].isna(), "indicator_column"] = True
    # Parking sessions indicator_column = False are those are existing in the dataset and could participate in V2G
    result_df.loc[((result_df["energy[charge_type][type]"].isna()) & (~result_df["destination_label"].isna())), "energy[charge_type][type]"] = "Parking"
    result_df.loc[((result_df["energy[charge_type][type]"].isna()) & (~result_df["destination_label"].isna())), "charge_type"] = "Parking"

    result_df.loc[(result_df["indicator_column"] == False) & (result_df["energy[charge_type][type]"] == "Parking"), "battery[soc][start][charging]"] = \
        result_df.loc[(result_df["indicator_column"] == False) & (result_df["energy[charge_type][type]"] == "Parking"), "battery[soc][end][trip]"]

    result_df.loc[(result_df["indicator_column"] == False) & (result_df["energy[charge_type][type]"] == "Parking"), "battery[soc][end][charging]"] = \
        result_df.loc[(result_df["indicator_column"] == False) & (result_df["energy[charge_type][type]"] == "Parking"), "battery[soc][end][trip]"]

    result_df.loc[(result_df["indicator_column"] == False) & (result_df["energy[charge_type][type]"] == "Parking"), "start_time_charging"] = \
        result_df.loc[(result_df["indicator_column"] == False) & (result_df["energy[charge_type][type]"] == "Parking"), "end_time_local"]

    result_df.loc[(result_df["indicator_column"] == False) & (result_df["energy[charge_type][type]"] == "Parking"), "end_time_charging"] = \
        result_df.loc[(result_df["indicator_column"] == False) & (result_df["energy[charge_type][type]"] == "Parking"), "end_time_local"]

    # Parking sessions indicator_column = True are those Trips that we generated
    result_df.loc[result_df["indicator_column"] == True, "duration_trip"] = 0
    result_df.loc[result_df["indicator_column"] == True, "distance"] = 0
    result_df.loc[result_df["indicator_column"] == True, "battery[soc][start][trip]"] = 100
    result_df.loc[result_df["indicator_column"] == True, "battery[soc][end][trip]"] = 100
    result_df.loc[result_df["indicator_column"] == True, "battery[soc][start][charging]"] = 100
    result_df.loc[result_df["indicator_column"] == True, "battery[soc][end][charging]"] = 100
    result_df.loc[result_df["indicator_column"] == True, "duration_charging"] = 0
    result_df.loc[result_df["indicator_column"] == True, "energy[charge_type][type]"] = "Parking"
    result_df.loc[result_df["indicator_column"] == True, "charge_type"] = "Parking"

    result_df.fillna(method='ffill', inplace=True)

    return result_df

##############################################################################################################################################
##############################################################################################################################################


def v2g_tou_parking_function(df):

    v2g_tou_p = df.copy()
    v2g_tou_p = v2g_tou_p[v2g_tou_p["energy[charge_type][type]"] == "Parking"]

    # v2g_tou_p = v2g_tou_cap(v2g_tou_p)
    # v2g_tou_p = v2g_tou_trip_buffer(v2g_tou_p)
    v2g_tou_p = v2g_tou_charging_buffer(v2g_tou_p)

    v2g_tou_p.loc[v2g_tou_p["indicator_column"] == True, "discharge_start"] = v2g_tou_p.loc[v2g_tou_p["indicator_column"] == True, "discharge_start"].dt.tz_convert(None).dt.strftime('%Y-%m-%d 16:00:00')
    v2g_tou_p.loc[v2g_tou_p["indicator_column"] == True, "discharge_end"] = v2g_tou_p.loc[v2g_tou_p["indicator_column"] == True, "discharge_end"].dt.tz_convert(None).dt.strftime('%Y-%m-%d 20:59:59')
    v2g_tou_p.loc[v2g_tou_p["indicator_column"] == True, "charge_start"] = (v2g_tou_p.loc[v2g_tou_p["indicator_column"] == True, "charge_start"].dt.tz_convert(None).dt.strftime('%Y-%m-%d 21:00:00'))
    v2g_tou_p.loc[v2g_tou_p["indicator_column"] == True, "charge_end"] = (v2g_tou_p.loc[v2g_tou_p["indicator_column"] == True, "charge_end"].dt.tz_convert(None).dt.strftime('%Y-%m-%d 16:00:00'))
    v2g_tou_p.loc[v2g_tou_p["indicator_column"] == True, "charge_start"] = pd.to_datetime(v2g_tou_p.loc[v2g_tou_p["indicator_column"] == True, "charge_start"]) + timedelta(days=1)
    v2g_tou_p.loc[v2g_tou_p["indicator_column"] == True, "charge_end"] = pd.to_datetime(v2g_tou_p.loc[v2g_tou_p["indicator_column"] == True, "charge_end"]) + timedelta(days=1)
    v2g_tou_p.loc[v2g_tou_p["indicator_column"] == True, "discharge_start"] = pd.to_datetime(v2g_tou_p.loc[v2g_tou_p["indicator_column"] == True, "discharge_start"])
    v2g_tou_p.loc[v2g_tou_p["indicator_column"] == True, "discharge_end"] = pd.to_datetime(v2g_tou_p.loc[v2g_tou_p["indicator_column"] == True, "discharge_end"])
    v2g_tou_p.loc[v2g_tou_p["indicator_column"] == True, "charge_start"] = pd.to_datetime(v2g_tou_p.loc[v2g_tou_p["indicator_column"] == True, "charge_start"])
    v2g_tou_p.loc[v2g_tou_p["indicator_column"] == True, "charge_end"] = pd.to_datetime(v2g_tou_p.loc[v2g_tou_p["indicator_column"] == True, "charge_end"])

    v2g_tou_p = v2g_tou_p.loc[(v2g_tou_p["discharge_start"].dt.hour < 21)].reset_index(drop=True).sort_values(by=["vehicle_name", "year", "month", "day"])

    v2g_tou_p.loc[(v2g_tou_p["indicator_column"] == False), "discharge_start"] = v2g_tou_p.loc[v2g_tou_p["indicator_column"] == False, "end_time_local"]
    v2g_tou_p.loc[(v2g_tou_p["indicator_column"] == False) & (v2g_tou_p["discharge_start"].dt.hour < 16), "discharge_start"] = v2g_tou_p.loc[v2g_tou_p["indicator_column"] == False, "discharge_start"].dt.tz_convert(None).dt.strftime('%Y-%m-%d 16:00:00')

    v2g_tou_p.loc[(v2g_tou_p["indicator_column"] == False), "discharge_end"] = v2g_tou_p.loc[v2g_tou_p["indicator_column"] == False, "next_departure_time"]
    v2g_tou_p.loc[(v2g_tou_p["indicator_column"] == False), "discharge_end"] = v2g_tou_p.loc[v2g_tou_p["indicator_column"] == False, "discharge_start"].dt.tz_convert(None).dt.strftime('%Y-%m-%d 20:59:59')

    v2g_tou_p.loc[v2g_tou_p["indicator_column"] == False, "charge_start"] = (v2g_tou_p.loc[v2g_tou_p["indicator_column"] == False, "charge_start"].dt.tz_convert(None).dt.strftime('%Y-%m-%d 21:00:00'))
    v2g_tou_p.loc[v2g_tou_p["indicator_column"] == False, "charge_end"] = (v2g_tou_p.loc[v2g_tou_p["indicator_column"] == False, "next_departure_time"])

    v2g_tou_p.loc[v2g_tou_p["indicator_column"] == False, "discharge_start"] = pd.to_datetime(v2g_tou_p.loc[v2g_tou_p["indicator_column"] == False, "discharge_start"])
    v2g_tou_p.loc[v2g_tou_p["indicator_column"] == False, "discharge_end"] = pd.to_datetime(v2g_tou_p.loc[v2g_tou_p["indicator_column"] == False, "discharge_end"])
    v2g_tou_p.loc[v2g_tou_p["indicator_column"] == False, "charge_start"] = pd.to_datetime(v2g_tou_p.loc[v2g_tou_p["indicator_column"] == False, "charge_start"])
    v2g_tou_p.loc[v2g_tou_p["indicator_column"] == False, "charge_end"] = pd.to_datetime(v2g_tou_p.loc[v2g_tou_p["indicator_column"] == False, "charge_end"])

    v2g_tou_p = v2g_tou_p[v2g_tou_p["destination_label"] != "Other"]

    v2g_tou_p.loc[v2g_tou_p["indicator_column"] == True, "V2G_SOC_tou_6k"] = 100
    v2g_tou_p.loc[v2g_tou_p["indicator_column"] == True, "V2G_SOC_tou_12k"] = 100
    v2g_tou_p.loc[v2g_tou_p["indicator_column"] == True, "V2G_SOC_tou_19k"] = 100

    v2g_tou_p.loc[v2g_tou_p["indicator_column"] == True, "V2G_SOC_half_6k"] = 100
    v2g_tou_p.loc[v2g_tou_p["indicator_column"] == True, "V2G_SOC_half_12k"] = 100
    v2g_tou_p.loc[v2g_tou_p["indicator_column"] == True, "V2G_SOC_half_19k"] = 100

    v2g_tou_p.loc[v2g_tou_p["indicator_column"] == True, "V2G_cap_6k_tou"] = 33
    v2g_tou_p.loc[v2g_tou_p["indicator_column"] == True, "V2G_cap_12k_tou"] = 60
    v2g_tou_p.loc[v2g_tou_p["indicator_column"] == True, "V2G_cap_19k_tou"] = 95

    v2g_tou_p['discharge_start'] = pd.to_datetime(v2g_tou_p['discharge_start']).dt.tz_localize(None)
    v2g_tou_p['discharge_end'] = pd.to_datetime(v2g_tou_p['discharge_end']).dt.tz_localize(None)
    # Then, set 'discharge_end' to 21:59:00 on the same day as 'discharge_start'
    # for records meeting your conditions
    condition = (v2g_tou_p["energy[charge_type][type]"] == "Parking") & (v2g_tou_p["discharge_start"].dt.hour < 21)
    v2g_tou_p.loc[condition, 'discharge_end'] = v2g_tou_p.loc[condition, 'discharge_start'].dt.floor('d') + pd.Timedelta(hours=21, minutes=00)

    v2g_tou_p.loc[v2g_tou_p["energy[charge_type][type]"] == "Parking", "V2G_cap_6k_tou"] = (((v2g_tou_p.loc[v2g_tou_p["energy[charge_type][type]"] == "Parking", "discharge_end"] - v2g_tou_p.loc[v2g_tou_p["energy[charge_type][type]"] == "Parking", "discharge_start"]).dt.seconds)/3600) * 6.6
    v2g_tou_p.loc[v2g_tou_p["energy[charge_type][type]"] == "Parking", "V2G_cap_12k_tou"] = (((v2g_tou_p.loc[v2g_tou_p["energy[charge_type][type]"] == "Parking", "discharge_end"] - v2g_tou_p.loc[v2g_tou_p["energy[charge_type][type]"] == "Parking", "discharge_start"]).dt.seconds)/3600) * 12
    v2g_tou_p.loc[v2g_tou_p["energy[charge_type][type]"] == "Parking", "V2G_cap_19k_tou"] = (((v2g_tou_p.loc[v2g_tou_p["energy[charge_type][type]"] == "Parking", "discharge_end"] - v2g_tou_p.loc[v2g_tou_p["energy[charge_type][type]"] == "Parking", "discharge_start"]).dt.seconds)/3600) * 19

    v2g_tou_p.loc[v2g_tou_p["V2G_cap_6k_tou"] > v2g_tou_p["bat_cap"], "V2G_cap_6k_tou"] = v2g_tou_p.loc[v2g_tou_p["V2G_cap_6k_tou"] > v2g_tou_p["bat_cap"], "bat_cap"]
    v2g_tou_p.loc[v2g_tou_p["V2G_cap_12k_tou"] > v2g_tou_p["bat_cap"], "V2G_cap_12k_tou"] = v2g_tou_p.loc[v2g_tou_p["V2G_cap_12k_tou"] > v2g_tou_p["bat_cap"], "bat_cap"]
    v2g_tou_p.loc[v2g_tou_p["V2G_cap_19k_tou"] > v2g_tou_p["bat_cap"], "V2G_cap_19k_tou"] = v2g_tou_p.loc[v2g_tou_p["V2G_cap_19k_tou"] > v2g_tou_p["bat_cap"], "bat_cap"]

    return v2g_tou_p



def conditionally_update_start_time(row):
    from datetime import datetime

    # Check if charge_type is 'parking'
    if row["energy[charge_type][type]"] == 'Parking':
        start_time = datetime.strptime(row['start_time_local'], '%Y-%m-%d %H:%M:%S%z')
        start_day = start_time.day

        # Proceed only if the day from start_time and the day column are different
        if start_day != row['day']:
            # Update the day in start_time to match the 'day' column
            updated_start_time = start_time.replace(day=row['day'])

            # Update the start_time in the data
            row['start_time_local'] = updated_start_time.strftime('%Y-%m-%d %H:%M:%S%z')
    return row

def conditionally_update_end_time(row):
    from datetime import datetime, timedelta

    # Check if charge_type is 'parking'
    if row["energy[charge_type][type]"] == 'Parking':
        end_time = datetime.strptime(row['end_time_local'], '%Y-%m-%d %H:%M:%S%z')
        end_day = end_time.day

        # Proceed only if the day from end_time and the day column are different
        if end_day != row['day']:
            try:
                # Attempt to update the day in end_time to match the 'day' column
                updated_end_time = end_time.replace(day=row['day'])
            except ValueError:
                # If the day is out of range for the month, adjust to the last day of the month
                # First, find the first day of the next month
                first_of_next_month = (end_time.replace(day=28) + timedelta(days=4)).replace(day=1)
                # Then, subtract one day to get the last day of the current month
                last_day_of_month = first_of_next_month - timedelta(days=1)
                updated_end_time = end_time.replace(day=last_day_of_month.day)

            # Update the end_time in the data
            row['end_time_local'] = updated_end_time.strftime('%Y-%m-%d %H:%M:%S%z')

    return row


def conditionally_update_start_time_charging(row):
    from datetime import datetime, timedelta

    # Check if charge_type is 'parking'
    if row["energy[charge_type][type]"] == 'Parking':
        end_time = datetime.strptime(row['start_time_charging'], '%Y-%m-%d %H:%M:%S%z')
        end_day = end_time.day

        # Proceed only if the day from end_time and the day column are different
        if end_day != row['day']:
            try:
                # Attempt to update the day in end_time to match the 'day' column
                updated_end_time = end_time.replace(day=row['day'])
            except ValueError:
                # If the day is out of range for the month, adjust to the last day of the month
                # First, find the first day of the next month
                first_of_next_month = (end_time.replace(day=28) + timedelta(days=4)).replace(day=1)
                # Then, subtract one day to get the last day of the current month
                last_day_of_month = first_of_next_month - timedelta(days=1)
                updated_end_time = end_time.replace(day=last_day_of_month.day)

            # Update the end_time in the data
            row['start_time_charging'] = updated_end_time.strftime('%Y-%m-%d %H:%M:%S%z')
    return row

def conditionally_update_end_time_charging(row):
    from datetime import datetime, timedelta

    # Check if charge_type is 'parking'
    if row["energy[charge_type][type]"] == 'Parking':
        end_time = datetime.strptime(row['end_time_charging'], '%Y-%m-%d %H:%M:%S%z')
        end_day = end_time.day

        # Proceed only if the day from end_time and the day column are different
        if end_day != row['day']:
            try:
                # Attempt to update the day in end_time to match the 'day' column
                updated_end_time = end_time.replace(day=row['day'])
            except ValueError:
                # If the day is out of range for the month, adjust to the last day of the month
                # First, find the first day of the next month
                first_of_next_month = (end_time.replace(day=28) + timedelta(days=4)).replace(day=1)
                # Then, subtract one day to get the last day of the current month
                last_day_of_month = first_of_next_month - timedelta(days=1)
                updated_end_time = end_time.replace(day=last_day_of_month.day)

            # Update the end_time in the data
            row['end_time_charging'] = updated_end_time.strftime('%Y-%m-%d %H:%M:%S%z')

    return row


def conditionally_adjust_charging_end(row):

    # Check if charge_type is 'parking'
    if row["energy[charge_type][type]"] != 'Parking':
        end_time = row['discharge_end']
        end_day = end_time.day

        # Proceed only if the day from end_time and the day column are different
        if end_day != row['day']:

            # Update the end_time in the data
            row['discharge_end'] = row['discharge_start']

    return row

