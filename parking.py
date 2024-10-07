# %%
import pytz
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import math
from datetime import timedelta
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import os
##################################################################################################################
##################################################################################################################
# %%


def read_clean(value):
    home_dir = os.path.expanduser("~")

    # Define the relative path to your file from the user's home directory
    relative_path = os.path.join('Library', 'CloudStorage', 'GoogleDrive-htayarani@ucdavis.edu', 'My Drive', 'PycharmProjects', 'LSTM', 'bev_trips_full.csv')
    full_trips = os.path.join(home_dir, relative_path)
    data_trips_full = pd.read_csv(full_trips, low_memory=False)

    relative_path1 = os.path.join('Library', 'CloudStorage', 'GoogleDrive-htayarani@ucdavis.edu', 'My Drive', 'PycharmProjects', 'LSTM', 'bev_trips.csv')
    full_trips1 = os.path.join(home_dir, relative_path1)
    data_trips = pd.read_csv(full_trips1).drop("destination_label", axis=1)
    data_trips = pd.merge(data_trips, data_trips_full[["id", "Lat", "Long", "cluster_rank", "destination_label"]], how="left", on="id")

    relative_path2 = os.path.join('Library', 'CloudStorage', 'GoogleDrive-htayarani@ucdavis.edu', 'My Drive', 'PycharmProjects', 'LSTM', 'bev_zcharges.csv')
    full_trips2 = os.path.join(home_dir, relative_path2)
    data_charge = pd.read_csv(full_trips2, low_memory=False)
    data_charge = data_charge.sort_values(by="start_time_ (local)").reset_index(drop=True)
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
    data_charge_grouped = data_charge_grouped.reset_index()

    data_charge_grouped["id"] = data_charge_grouped["last_trip_id"]
    data = pd.merge(data_trips, data_charge_grouped, on="id", how="left")
    # Rename columns for clarity
    data.rename(columns={'duration_y': 'duration_charging',
                         'start_time_y': 'start_time_charging',
                         'end_time_y': 'end_time_charging',
                         'duration_x': 'duration_trip',
                         'start_time_x': 'start_time_trip',
                         'end_time_x': 'end_time_trip',
                         'total_energy': 'Energy_charged',
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
                     "P_1281", "P_1285", "P_1288", "P_1294", "P_1295", "P_1296", "P_1304", "P_1307", "P_1375",
                     "P_1088a", "P_1122", "P_1264", "P_1267", "P_1276", "P_1289", "P_1290", "P_1300", "P_1319"]
    df = pd.DataFrame()
    for vehicle_name in vehicle_names:
        df_full_trips = read_clean(vehicle_name)  # done
        df_full_trips_short = trip_summary(df_full_trips)  # done
        df_soc_req = soc_next(df_full_trips_short) # done
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


# New function to adjust battery size based on specific conditions
def adjust_battery_size(row):
    model_name = row['Model']
    previous_battery_size = row['bat_cap']  # Using the result from extract_last_chars

    # Only adjust if the model starts with 'Model S'
    if model_name.startswith("Model S"):
        # Define the possible battery sizes
        battery_sizes = [70, 75, 85, 100]

        # Search for a battery size in the model name
        for size in battery_sizes:
            if str(size) in model_name:
                return size

    # If no specific size is found or model does not start with 'Model S', return the previous size
    return previous_battery_size
##################################################################################################################
##################################################################################################################


def trip_summary(df):
    data3 = df[["ts_start", "ts_end", "start_time_local", "end_time_local", "Model", "vehicle_name", "vehicle_model", "year", "month", "day", "hour", "duration_trip", "distance", "battery[soc][start][trip]",
               "battery[soc][end][trip]", "Lat", "Long", "destination_label", "origin_label", "Energy_charged", "energy[charge_type][type]", "battery[soc][start][charging]", "battery[soc][end][charging]", "start_time", "end_time", "duration_charging"]].copy()
    ""
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
    # data3["bat_cap"] = data3.apply(adjust_battery_size, axis=1)

    return data3
##################################################################################################################
##################################################################################################################


# selecting only the trips that have charging session at the end
def charging_dataframe(df, time):
    # trip_data = pd.read_csv("data.csv")
    # df= trip_data.copy()
    final_dataframes_charging = charging_selection(df)
    # determine teh charging speed based on the parking time, charging time and SOC before and after charging
    final_dataframes_charging = charging_speed(final_dataframes_charging)
    # range indicator is indicating if the trip will fail or not
    final_dataframes_charging = range_indicator(final_dataframes_charging)
    final_dataframes_charging = v2g_draw(final_dataframes_charging)
    # final_dataframes_charging = final_dataframes_charging.loc[final_dataframes_charging["V2G_time_min"] >= time]
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
    df = df[~df["duration_charging_min"].isna()]
    average_duration = df.groupby(['origin_label', 'destination_label'])['duration_charging_min'].mean().reset_index(name='Average Charging Time')
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
        palette='viridis'
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
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    # Adjust the legend for the hue
    legend = scatterplot.legend_
    legend.set_bbox_to_anchor((1, 1.05))
    legend.set_loc('upper left')
    legend.get_frame().set_facecolor('#f0f0f0')  # Set the background color of the legend box

    # Decrease the font size of the legend items
    for text in legend.get_texts():
        text.set_fontsize(10)

    # Save the plot as an image file (e.g., PNG)
    plt.savefig('bubble_chart_with_charging.png', bbox_inches='tight')
    plt.show()
##################################################################################################################
##################################################################################################################
# %%
def draw_combined(df):
    # Calculate average time spent at each location for parking and charging
    df_parking = df.groupby(['origin_label', 'destination_label'])['parking_time_minute'].mean().reset_index(name='Average Parking Time')
    df_parking['Frequency'] = df.groupby(['origin_label', 'destination_label']).size().reset_index(name='Frequency')['Frequency']
    df_charging0 = df[~df["duration_charging_min"].isna()]
    df_charging = df_charging0.groupby(['origin_label', 'destination_label'])['duration_charging_min'].mean().reset_index(name='Average Charging Time')
    df_charging['Frequency'] = df_charging0.groupby(['origin_label', 'destination_label']).size().reset_index(name='Frequency')['Frequency']
    # Set the size limits based on the combined frequency values
    parking_min, charging_min = (df_parking['Frequency'].min()), (df_charging['Frequency'].min())
    parking_max, charging_max = (df_parking['Frequency'].max()), (df_charging['Frequency'].max())
    # Create a grid for the figure
    fig = plt.figure(figsize=(12, 8))
    # Add hatched background to the entire figure
    fig.patch.set_facecolor('white')
    fig.patch.set_hatch('//')
    gs = gridspec.GridSpec(3, 2, width_ratios=[4, 1], height_ratios=[3, 0.1, 0.1])
    # Create a subplot for the main scatter plot
    ax_main = fig.add_subplot(gs[0, 0])
    # Plot parking data
    scatterplot_parking = sns.scatterplot(
        data=df_parking,
        x='origin_label',
        y='destination_label',
        size='Frequency',
        hue='Average Parking Time',
        sizes=(parking_min * 0.65, parking_max * 0.65),  # Adjusted sizes for better visibility
        palette='cividis',
        legend=False,
        edgecolor='w',
        linewidth=1,
        alpha=0.9,  # Add transparency
        marker='o',
        ax=ax_main
    )
    # Overlay charging data
    scatterplot_charging = sns.scatterplot(
        data=df_charging,
        x='origin_label',
        y='destination_label',
        size='Frequency',
        hue='Average Charging Time',
        sizes=(charging_min * 0.65, charging_max * 0.65),  # Adjusted sizes for better visibility
        palette='vlag',
        legend=False,
        edgecolor='w',
        linewidth=1,
        alpha=0.9,  # Add transparency
        marker='o',  # Square marker for charging
        ax=ax_main)
    # Reverse the order of the y-axis
    ax_main.invert_yaxis()
    # Adjust the space between the first ticks and the origin on both axes
    ax_main.margins(x=0.3, y=0.3)
    # Automatically adjust the scale considering the margins
    ax_main.autoscale()
    # Customize the layout
    ax_main.set_xlabel('Origin', fontsize=20)
    ax_main.set_ylabel('Destination', fontsize=20)
    ax_main.tick_params(axis='both', which='major', labelsize=18)
    ax_main.grid(True)
    # Create color bars for parking and charging time
    norm_parking = plt.Normalize(df_parking['Average Parking Time'].min(), df_parking['Average Parking Time'].max())
    sm_parking = plt.cm.ScalarMappable(cmap='cividis', norm=norm_parking)
    sm_parking.set_array([])
    norm_charging = plt.Normalize(df_charging['Average Charging Time'].min(), df_charging['Average Charging Time'].max())
    sm_charging = plt.cm.ScalarMappable(cmap='vlag', norm=norm_charging)
    sm_charging.set_array([])
    # Create subplots for the color bars
    ax_cbar_parking = fig.add_subplot(gs[1, 0])
    ax_cbar_charging = fig.add_subplot(gs[2, 0])
    cbar_parking = plt.colorbar(sm_parking, cax=ax_cbar_parking, orientation='horizontal')
    cbar_parking.set_label('Average Parking Time (min)', fontsize=16)
    ax_cbar_parking.tick_params(labelsize=16)  # Increase the size of the color bar tick labels
    cbar_charging = plt.colorbar(sm_charging, cax=ax_cbar_charging, orientation='horizontal')
    cbar_charging.set_label('Average Charging Time (min)', fontsize=16)
    ax_cbar_charging.tick_params(labelsize=16)  # Increase the size of the color bar tick labels
    min_val = charging_min
    max_val = parking_max
    # Manually specify size legend elements
    size_values = [(min_val), (max_val) / 5, (max_val) * 3 / 5, (max_val) * 4 / 5, max_val]
    # size_labels = [str(round(int(s))) for s in size_values]
    size_labels = [str(round(s / 100) * 100) for s in size_values]
    handles = [plt.scatter([], [], s=s * 0.55, color='k', edgecolor='w') for s in size_values]
    ax_legend = fig.add_subplot(gs[0:3, 1])  # Create a subplot for the legend that spans the first two rows
    ax_legend.axis('off')  # Turn off the axis
    ax_legend.set_ylim(-35, 4)
    for i, (size, label) in enumerate(zip(size_values, size_labels)):
        y = 1 - (i * 1.2 + 2.5) * i
        ax_legend.scatter(0.5, y, s=size * 0.55, color='k', edgecolor='w')
        ax_legend.text(0.53, y, label, horizontalalignment='left', verticalalignment='center', fontsize=18)
    #
    ax_legend.set_title('Parking & Charging\nFrequency', fontsize=16, loc='center')
    plt.tight_layout(rect=[0, 0, 0.98, 0.98])
    # Save the plot as an image file (e.g., PNG)
    plt.savefig('combined_bubble_chart.png', bbox_inches='tight')
    plt.show()
# draw_combined(final_dataframes)

##################################################################################################################
##################################################################################################################
# %%

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
    fig, ax1 = plt.subplots(figsize=(20, 12))
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
    df.loc[mask, 'battery[soc][end][trip]'] = df.loc[mask, 'battery[soc][start][trip]'] - ((0.3 * df.loc[mask, 'distance']) * (100 / df["bat_cap"]))
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
    # trip_data = pd.read_csv("data.csv")
    #
    # # Make a copy of the DataFrame
    # df = trip_data.copy()

    # Filter rows where charging duration is not NaN
    final_df_charging = df.loc[~df["energy[charge_type][type]"].isna()].copy()

    # Fill NaN values in 'start_time_charging' with 'end_time_local'
    final_df_charging['start_time_charging'].fillna(final_df_charging['end_time_local'], inplace=True)

    # Fill NaN values in 'end_time_charging' with 'next_departure_time'
    final_df_charging['end_time_charging'].fillna(final_df_charging['next_departure_time'], inplace=True)


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



def charging_dataframe_parking(df, time):
    final_dataframes_charging = charging_selection_parking(df)
    # determine teh charging speed based on the parking time, charging time and SOC before and after charging
    final_dataframes_charging = charging_speed_parking(final_dataframes_charging)
    return final_dataframes_charging


def charging_selection_parking(df):

    # Filter rows where charging duration is not NaN
    final_df_charging = df.loc[(~df["energy[charge_type][type]"].isna()) | ((df["destination_label"] != "Other") & (df["energy[charge_type][type]"].isna()))].copy()
    # Fill NaN values in 'start_time_charging' with 'end_time_local'
    final_df_charging['start_time_charging'].fillna(final_df_charging['end_time_local'], inplace=True)

    # Fill NaN values in 'end_time_charging' with 'next_departure_time'
    final_df_charging['end_time_charging'].fillna(final_df_charging['next_departure_time'], inplace=True)
    final_df_charging['charge_type'].fillna("LEVEL_2", inplace=True)
    final_df_charging['energy[charge_type][type]'].fillna("LEVEL_2", inplace=True)

    final_df_charging['battery[soc][start][charging]'].fillna(final_df_charging['battery[soc][end][trip]'], inplace=True)
    final_df_charging['battery[soc][end][charging]'].fillna(final_df_charging['battery[soc][end][trip]'], inplace=True)

    return final_df_charging


def charging_speed_parking(df):
    df["charging_speed"] = ((((df["battery[soc][end][charging]"] - df["battery[soc][start][charging]"]) / 100) * df["bat_cap"]) / (df["duration_charging_min"] / 60))
    df.loc[df["charging_speed"] <= 1.6, "charge_type"] = "LEVEL_1"
    df.loc[(df["charging_speed"] > 1.6) & (df["charging_speed"] < 21), "charge_type"] = "LEVEL_2"
    df.loc[df["charging_speed"] >= 21, "charge_type"] = "DC_FAST"
    return df


def xlsx_read(dic):

    # List of Excel file names
    excel_files = [f for f in os.listdir(dic) if f.endswith('.xlsx')]

    # Dictionary to store dataframes
    all_dataframes = {}

    # Iterate over each Excel file
    for excel_file_name in excel_files:
        excel_file_path = os.path.join(dic, excel_file_name)
        print(f"Reading Excel file '{excel_file_path}'...")

        # Read each sheet into a separate dataframe
        with pd.ExcelFile(excel_file_path) as xls:
            sheet_names = xls.sheet_names  # Get the names of all sheets in the Excel file

            # Read each sheet into a dataframe and store it in the dictionary
            for sheet_name in sheet_names:
                df = pd.read_excel(xls, sheet_name=sheet_name)
                new_df_name = f"{excel_file_name[:-5]}_{sheet_name}"  # Add sheet name to the file name
                all_dataframes[new_df_name] = df

    # Create a new dataframe to store total cost data
    total_costs_df = pd.DataFrame()

    # Iterate over the dataframes and extract total costs
    for df_name, df in all_dataframes.items():
        if "Total Costs" in df_name:
            # Extract charging type and speed from the dataframe name
            charging_type = "smart" if "smart" in df_name else "v2g"
            charging_speed = df_name.split("_")[2][:-1]
            ghg_cost = df_name.split("_")[3][:-2]

            # Add a column indicating the charging type (smart or v2g)
            df['Charging Type'] = charging_type
            # Add columns indicating charging speed and GHG cost
            df['Charging Speed'] = charging_speed
            df['GHG Cost'] = ghg_cost

            # Concatenate this dataframe with the total_costs_df
            total_costs_df = pd.concat([total_costs_df, df])

    print("Total cost data has been extracted.")
    total_costs_df = total_costs_df.reset_index(drop=True)
    # Display the new dataframe
    print(total_costs_df)

    # Create a new dataframe to store total cost data
    individual_cost_df = pd.DataFrame()

    # Iterate over the dataframes and extract total costs
    for df_name, df in all_dataframes.items():
        if "Individual Cost" in df_name:
            # Extract charging type and speed from the dataframe name
            charging_type = "smart" if "smart" in df_name else "v2g"
            charging_speed = df_name.split("_")[2][:-1]
            ghg_cost = df_name.split("_")[3][:-2]

            # Add a column indicating the charging type (smart or v2g)
            df['Charging Type'] = charging_type
            # Add columns indicating charging speed and GHG cost
            df['Charging Speed'] = charging_speed
            df['GHG Cost'] = ghg_cost

            # Concatenate this dataframe with the total_costs_df
            individual_cost_df = pd.concat([individual_cost_df, df])

    individual_cost_df = individual_cost_df.reset_index(drop=True)
    # Display the new dataframe
    return total_costs_df, individual_cost_df
#


def json_file_read(dic, flatten_veh):

    # List of JSON file names
    json_files = [f for f in os.listdir(dic) if f.endswith('.json')]
    # Dictionary to store dataframes
    all_dataframes1 = {}

    # Iterate over each JSON file
    for json_file_name in json_files:
        json_file_path = os.path.join(dic, json_file_name)
        print(f"Reading JSON file '{json_file_path}'...")

        # Read each JSON file into a dataframe
        df = pd.read_json(json_file_path)
        # Add the dataframe to the dictionary with the file name as key
        all_dataframes1[json_file_name[:-5]] = df

    # Create a new dataframe to store total cost data
    hourly_data = pd.DataFrame()
    # Iterate over the dataframes and extract total costs
    for df_name, df in all_dataframes1.items():
        # Extract charging type and speed from the dataframe name
        charging_type = "smart" if "smart" in df_name else "v2g"
        charging_speed = df_name.split("_")[2][:-1]
        ghg_cost = df_name.split("_")[3][:-2]

        # Add a column indicating the charging type (smart or v2g)
        df['Charging Type'] = charging_type
        # Add columns indicating charging speed and GHG cost
        df['Charging Speed'] = charging_speed
        df['GHG Cost'] = ghg_cost

        # Concatenate this dataframe with the hourly_data dataframe
        hourly_data = pd.concat([hourly_data, df], ignore_index=True)

    hourly_data = hourly_data.reset_index(drop=True)
    hourly_data = pd.merge(hourly_data, flatten_veh[["Vehicle", "Hour", "charging_indicator", "location"]], how="left", on=["Vehicle", "Hour"])

    hourly_data["Charging Speed"] = hourly_data["Charging Speed"].astype(float)
    hourly_data["charging_indicator"] = hourly_data["charging_indicator"].fillna(0).astype(int)
    hourly_data["GHG Cost"] = hourly_data["GHG Cost"].astype(float)
    hourly_data_discharging = hourly_data[(hourly_data["X_CHR"] <= 0) & (hourly_data["charging_indicator"] == 1)]
    hourly_data_charging = hourly_data[(hourly_data["X_CHR"] > 0)]
    # Group the DataFrame by 'hour'
    hourly_data_discharging = hourly_data_discharging[hourly_data_discharging["Charging Type"] == "v2g"]
    grouped_df = hourly_data_discharging.groupby(['Hour', "Charging Speed", "GHG Cost", "Batt_cap", "location"])
    # Calculate sum and size (count) for each group
    result = grouped_df.agg({'X_CHR': ['sum', 'count']})
    result = result.reset_index(drop=False)

    # Create two new columns from the MultiIndex
    result['X_CHR_Sum'] = result[('X_CHR', 'sum')]
    result['X_CHR_Count'] = result[('X_CHR', 'count')]

    # Drop the original MultiIndex column
    result = result.drop(columns=[('X_CHR', 'sum'), ('X_CHR', 'count')])

    # Calculate Total_power
    result["Total_power"] = result["Charging Speed"] * result["X_CHR_Count"]
    result["Utilization Rate"] = abs(result["X_CHR_Sum"] / result["Total_power"])*100

    # Convert the hour values to modulo 24 to represent a 24-hour clock
    result['Hour_of_day'] = result['Hour'] % 24

    # Identify the peak hours between 4 PM and 9 PM (16:00 to 21:00)
    result['Peak'] = (result['Hour_of_day'] >= 16) & (result['Hour_of_day'] <= 21)

    # Convert boolean values to 'Peak' and 'Non-Peak' strings
    result['Peak'] = result['Peak'].map({True: 'Peak', False: 'Non-Peak'})

    return result, hourly_data


def plot_price_chart_TOU(off_peak_price, peak_price, bar_width, colors, font_size=12):
    # Define the time periods and corresponding prices
    time_periods = ['12 a.m.', '1 a.m.', '2 a.m.', '3 a.m.', '4 a.m.', '5 a.m.', '6 a.m.', '7 a.m.', '8 a.m.', '9 a.m.', '10 a.m.', '11 a.m.',
                    '12 p.m.', '1 p.m.', '2 p.m.', '3 p.m.', '4 p.m.', '5 p.m.', '6 p.m.', '7 p.m.', '8 p.m.', '9 p.m.', '10 p.m.', '11 p.m.']
    prices = []
    periods = []

    # Assign prices and periods based on time
    for hour in range(24):
        if hour >= 0 and hour < 3:
            prices.append(off_peak_price)
            periods.append('Off-Peak')
        elif hour >= 16 and hour < 21:
            prices.append(peak_price)
            periods.append('Peak')
        else:
            prices.append(off_peak_price)
            periods.append('Off-Peak')

    # Create DataFrame
    data = {
        'Time': time_periods,
        'Price': prices,
        'Period': periods
    }
    df = pd.DataFrame(data)

    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot the bars with hatch pattern and without edges
    bars = plt.bar(range(len(df)), df['Price'], color=[colors[period] for period in df['Period']], width=bar_width, edgecolor='none', hatch='//')

    # Set xticks and labels
    plt.xticks(range(len(df)), df['Time'], rotation=45, ha='right', fontsize=font_size)
    plt.yticks(fontsize=font_size)

    # Add annotations for each section
    for i in range(len(df)):
        if i == 0 or df['Period'].iloc[i] != df['Period'].iloc[i - 1]:
            # Find the middle of the current section
            section_start = i
            while i < len(df) and df['Period'].iloc[i] == df['Period'].iloc[section_start]:
                i += 1
            section_end = i - 1
            annotation_x = section_start + (section_end - section_start) / 2
            plt.text(annotation_x, df['Price'].iloc[section_start], f'{int(df["Price"].iloc[section_start])}¢', ha='center', va='bottom', fontsize=font_size, color='black')

    # Add labels and title
    plt.ylabel('¢/kWh', fontsize=(font_size + 2))
    plt.xlabel('Time', fontsize=(font_size + 2))
    plt.title('', fontsize=font_size)

    # Add legend with hatch pattern
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[label], hatch='//') for label in colors]
    labels = colors.keys()
    plt.legend(handles, labels, loc='upper left', fontsize=font_size)

    # Set x-axis range
    plt.xlim(-0.5, len(df) - 0.5)
    plt.ylim(0, 70)

    # Show the plot
    plt.tight_layout()
    plt.show()




def plot_price_chart_EVRATE(off_peak_price, part_peak_price, peak_price, bar_width, colors, font_size=12):
    # Define the time periods and corresponding prices
    time_periods = ['12 a.m.', '1 a.m.', '2 a.m.', '3 a.m.', '4 a.m.', '5 a.m.', '6 a.m.', '7 a.m.', '8 a.m.', '9 a.m.', '10 a.m.', '11 a.m.',
                    '12 p.m.', '1 p.m.', '2 p.m.', '3 p.m.', '4 p.m.', '5 p.m.', '6 p.m.', '7 p.m.', '8 p.m.', '9 p.m.', '10 p.m.', '11 p.m.']
    prices = []
    periods = []

    # Assign prices and periods based on time
    for hour in range(24):
        if hour >= 0 and hour < 3:
            prices.append(off_peak_price)
            periods.append('Off-Peak')
        elif hour == 15:
            prices.append(part_peak_price)
            periods.append('Part-Peak')
        elif hour >= 16 and hour < 21:
            prices.append(peak_price)
            periods.append('Peak')
        elif hour == 21 or hour == 22:
            prices.append(part_peak_price)
            periods.append('Part-Peak')
        else:
            prices.append(off_peak_price)
            periods.append('Off-Peak')

    # Create DataFrame
    data = {
        'Time': time_periods,
        'Price': prices,
        'Period': periods
    }
    df = pd.DataFrame(data)

    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot the bars with hatch pattern and without edges
    bars = plt.bar(range(len(df)), df['Price'], color=[colors[period] for period in df['Period']], width=bar_width, edgecolor='none', hatch='//')

    # Set xticks and labels
    plt.xticks(range(len(df)), df['Time'], rotation=45, ha='right', fontsize=font_size)
    plt.yticks(fontsize=font_size)

    # Add annotations for each section
    for i in range(len(df)):
        if i == 0 or df['Period'].iloc[i] != df['Period'].iloc[i - 1]:
            # Find the middle of the current section
            section_start = i
            while i < len(df) and df['Period'].iloc[i] == df['Period'].iloc[section_start]:
                i += 1
            section_end = i - 1
            annotation_x = section_start + (section_end - section_start) / 2
            plt.text(annotation_x, df['Price'].iloc[section_start], f'{int(df["Price"].iloc[section_start])}¢', ha='center', va='bottom', fontsize=font_size, color='black')

    # Add labels and title
    plt.ylabel('¢/kWh', fontsize=(font_size + 2))
    plt.xlabel('Time', fontsize=(font_size + 2))
    plt.title("", fontsize=font_size)

    # Add legend with hatch pattern
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[label], hatch='//') for label in colors]
    labels = colors.keys()
    plt.legend(handles, labels, loc='upper left', fontsize=font_size)

    # Set x-axis range
    plt.xlim(-0.5, len(df) - 0.5)
    plt.ylim(0, 70)

    # Show the plot
    plt.tight_layout()
    plt.show()

##

def plot_cost_comparison_EV(df, num_vehicles, title_size=14, axis_text_size=12, y_axis_title='Cost ($)', barhight = 500):
    df1 = df.copy()
    df1['Scenario'] = df1['Charging Type'] + ' - ' + df1['Charging Speed'].astype(str) + ' - ' + df1['Plugged-in Sessions'] + ' - ' + df1['V2G_Location']

    # Calculate average costs
    df1[["Electricity_Cost", "Degradation_Cost"]] = df1[["Electricity_Cost", "Degradation_Cost"]].div(num_vehicles, axis=0)

    # Extract the Actual Behavior rows
    actual_behavior_df = df1[(df1['Charging Type'] == 'Actual Behavior - EV Rate')]

    # Define the plot structure
    fig, axes = plt.subplots(2, 2, figsize=(16, 8), sharey=True)
    plt.subplots_adjust(hspace=0.4, wspace=0.4)

    # Define custom colors
    colors = {
        'Actual': ['#1f77b4', '#ff0000'],  # Blue for electricity, red for degradation
        'smart': ['#2ca02c', '#ff0000'],  # Different shade of green for electricity, red for degradation
        'v2g': ['#ff7f0e', '#ff0000']  # Different shade of orange for electricity, red for degradation
    }

    # Define the axes
    axes_dict = {
        'Home-Actual': axes[0, 0],
        'Home_Work-Actual': axes[0, 1],
        'Home-Potential': axes[1, 0],
        'Home_Work-Potential': axes[1, 1]
    }

    # Plotting each subplot
    for v2g_location in ['Home', 'Home_Work']:
        for plugged_in in ['Actual', 'Potential']:
            ax = axes_dict[f'{v2g_location}-{plugged_in}']
            subset = df1[(df1['V2G_Location'] == v2g_location) & (df1['Plugged-in Sessions'] == plugged_in)]

            # Include Actual Behavior rows in each subplot
            subset = pd.concat([subset, actual_behavior_df], ignore_index=True)

            # Plot Actual Behavior
            actual_subset_E = subset[subset['Charging Type'] == 'Actual Behavior - EV Rate']
            if not actual_subset_E.empty:
                electricity_cost = actual_subset_E['Electricity_Cost'].values[0]
                degradation_cost = actual_subset_E['Degradation_Cost'].values[0]

                ax.bar('Actual-EV Rate', max(0, electricity_cost), color=colors['Actual'][0], label='Electricity Cost')
                ax.bar('Actual-EV Rate', max(0, degradation_cost), bottom=max(0, electricity_cost), color=colors['Actual'][1], label='Degradation Cost')
                ax.bar('Actual-EV Rate', min(0, electricity_cost), color=colors['Actual'][0])
                ax.bar('Actual-EV Rate', min(0, degradation_cost), bottom=min(0, electricity_cost), color=colors['Actual'][1])

                # Add text annotations for sum of costs
                total_cost = electricity_cost + degradation_cost
                bar_height = max(0, electricity_cost) + max(0, degradation_cost)
                ax.text('Actual-EV Rate', barhight if total_cost > 0 else -barhight, f"${total_cost:.2f}", ha='center', va='bottom' if total_cost > 0 else 'top', color='black', fontsize=13, rotation=90)

                # Plot Smart Charging
            for speed in ['6.6', '12', '19']:
                smart_subset = subset[(subset['Charging Type'] == 'smart') & (subset['Charging Speed'] == float(speed))]
                if not smart_subset.empty:
                    electricity_cost = smart_subset['Electricity_Cost'].values[0]
                    degradation_cost = smart_subset['Degradation_Cost'].values[0]

                    ax.bar(f'Smart-{speed}', max(0, electricity_cost), color=colors['smart'][0], label=f'Smart {speed} kW')
                    ax.bar(f'Smart-{speed}', max(0, degradation_cost), bottom=max(0, electricity_cost), color=colors['smart'][1], alpha=0.7)
                    ax.bar(f'Smart-{speed}', min(0, electricity_cost), color=colors['smart'][0])
                    ax.bar(f'Smart-{speed}', min(0, degradation_cost), bottom=min(0, electricity_cost), color=colors['smart'][1], alpha=0.7)

                    # Add text annotations for sum of costs
                    total_cost = electricity_cost + degradation_cost
                    bar_height = max(0, electricity_cost) + max(0, degradation_cost)
                    ax.text(f'Smart-{speed}', barhight if total_cost > 0 else barhight, f"${total_cost:.2f}", ha='center', va='bottom' if total_cost > 0 else 'bottom', color='black', fontsize=13, rotation=90)

                # Plot V2G Charging
            for speed in ['6.6', '12', '19']:
                v2g_subset = subset[(subset['Charging Type'] == 'v2g') & (subset['Charging Speed'] == float(speed))]
                if not v2g_subset.empty:
                    electricity_cost = v2g_subset['Electricity_Cost'].values[0]
                    degradation_cost = v2g_subset['Degradation_Cost'].values[0]

                    ax.bar(f'V2G-{speed}', max(0, electricity_cost), color=colors['v2g'][0], label=f'V2G {speed} kW')
                    ax.bar(f'V2G-{speed}', max(0, degradation_cost), bottom=max(0, electricity_cost), color=colors['v2g'][1], alpha=0.7)
                    ax.bar(f'V2G-{speed}', min(0, electricity_cost), color=colors['v2g'][0])
                    ax.bar(f'V2G-{speed}', min(0, degradation_cost), bottom=min(0, electricity_cost), color=colors['v2g'][1], alpha=0.7)

                    # Add text annotations for sum of costs
                    total_cost = electricity_cost + degradation_cost
                    bar_height = max(0, electricity_cost) + max(0, degradation_cost)
                    ax.text(f'V2G-{speed}', barhight if total_cost > 0 else barhight, f"${total_cost:.2f}", ha='center', va='bottom' if total_cost > 0 else 'bottom', color='black', fontsize=13, rotation=90)

            # Set axis labels and title sizes
            ax.set_ylabel(y_axis_title, fontsize=title_size)
            ax.set_xlabel('Charging Scenario', fontsize=title_size)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=axis_text_size)
            ax.tick_params(axis='both', which='major', labelsize=axis_text_size)
            ax.grid(True)

    # Add column titles with grey background
    for ax, title in zip([axes[0, 0], axes[0, 1]], ['V2G Location: Home', 'V2G Location: Home and Work']):
        rect = plt.Rectangle((0, 1.05), 1, 0.13, color='grey', transform=ax.transAxes, clip_on=False)
        ax.add_patch(rect)
        ax.annotate(title, xy=(0.5, 1.08), xytext=(5, 0), textcoords='offset points', xycoords='axes fraction', ha='center', fontsize=14, weight='bold', color='white')

    # Add row titles with grey background
    for ax, title in zip([axes[0, 0], axes[1, 0]], ['Actual\nCharging Sessions', 'Potential\nCharging Sessions']):
        rect = plt.Rectangle((-0.38, 0), 0.15, 1.1, color='grey', transform=ax.transAxes, clip_on=False)
        ax.add_patch(rect)
        ax.annotate(title, xy=(-0.34, 0.54), xytext=(-0.34, 0.5), textcoords='offset points', xycoords='axes fraction', va='center', rotation='vertical', fontsize=14, weight='bold', color='white')

    # Add overall title
    # fig.suptitle('Cost Comparison under Different Charging Scenarios', fontsize=16, weight='bold')

    # Add legend
    handles = [
        plt.Rectangle((-1, 0), 20, 20, color=colors['Actual'][0]),
        plt.Rectangle((-1, 0), 20, 20, color=colors['smart'][0]),
        plt.Rectangle((-1, 0), 20, 20, color=colors['v2g'][0]),
        plt.Rectangle((-1, 0), 20, 20, color=colors['Actual'][1])
    ]
    labels = ['Actual Behavior \n Electricity Cost', 'Smart Charging \n Electricity Cost', 'V2G Charging \n Electricity Cost', 'Degradation Cost']
    fig.legend(handles, labels, loc='center right', fontsize=14)

    # Show the plot
    plt.tight_layout(rect=[0.05, 0, 0.8, 0.95])
    plt.show()


def plot_cost_comparison_TOU(df, num_vehicles, title_size=14, axis_text_size=12, y_axis_title='Cost ($)', barhight=500):
    df1 = df.copy()
    df1['Scenario'] = df1['Charging Type'] + ' - ' + df1['Charging Speed'].astype(str) + ' - ' + df1['Plugged-in Sessions'] + ' - ' + df1['V2G_Location']

    # Calculate average costs
    df1[["Electricity_Cost", "Degradation_Cost"]] = df1[["Electricity_Cost", "Degradation_Cost"]].div(num_vehicles, axis=0)

    # Extract the Actual Behavior rows
    actual_behavior_df = df1[(df1['Charging Type'] == 'Actual Behavior - TOU Rate')]

    # Define the plot structure
    fig, axes = plt.subplots(2, 2, figsize=(16, 8), sharey=True)
    plt.subplots_adjust(hspace=0.4, wspace=0.4)

    # Define custom colors
    colors = {
        'Actual': ['#1f77b4', '#ff0000'],  # Blue for electricity, red for degradation
        'smart': ['#2ca02c', '#ff0000'],  # Different shade of green for electricity, red for degradation
        'v2g': ['#ff7f0e', '#ff0000']  # Different shade of orange for electricity, red for degradation
    }

    # Define the axes
    axes_dict = {
        'Home-Actual': axes[0, 0],
        'Home_Work-Actual': axes[0, 1],
        'Home-Potential': axes[1, 0],
        'Home_Work-Potential': axes[1, 1]
    }

    # Plotting each subplot
    for v2g_location in ['Home', 'Home_Work']:
        for plugged_in in ['Actual', 'Potential']:
            ax = axes_dict[f'{v2g_location}-{plugged_in}']
            subset = df1[(df1['V2G_Location'] == v2g_location) & (df1['Plugged-in Sessions'] == plugged_in)]

            # Include Actual Behavior rows in each subplot
            subset = pd.concat([subset, actual_behavior_df], ignore_index=True)

            # Plot Actual Behavior
            actual_subset_T = subset[subset['Charging Type'] == 'Actual Behavior - TOU Rate']
            if not actual_subset_T.empty:
                electricity_cost = actual_subset_T['Electricity_Cost'].values[0]
                degradation_cost = actual_subset_T['Degradation_Cost'].values[0]

                ax.bar('Actual-TOU', max(0, electricity_cost), color=colors['Actual'][0], label='Electricity Cost')
                ax.bar('Actual-TOU', max(0, degradation_cost), bottom=max(0, electricity_cost), color=colors['Actual'][1], label='Degradation Cost')
                ax.bar('Actual-TOU', min(0, electricity_cost), color=colors['Actual'][0])
                ax.bar('Actual-TOU', min(0, degradation_cost), bottom=min(0, electricity_cost), color=colors['Actual'][1])

                # Add text annotations for sum of costs
                total_cost = electricity_cost + degradation_cost
                bar_height = max(0, electricity_cost) + max(0, degradation_cost)
                ax.text('Actual-TOU', barhight if total_cost > 0 else -barhight, f"${total_cost:.2f}", ha='center', va='bottom' if total_cost > 0 else 'top', color='black', fontsize=13, rotation=90)

                # Plot Smart Charging
                for speed in ['6.6', '12', '19']:
                    smart_subset = subset[(subset['Charging Type'] == 'smart') & (subset['Charging Speed'] == float(speed))]
                    if not smart_subset.empty:
                        electricity_cost = smart_subset['Electricity_Cost'].values[0]
                        degradation_cost = smart_subset['Degradation_Cost'].values[0]

                        ax.bar(f'Smart-{speed}', max(0, electricity_cost), color=colors['smart'][0], label=f'Smart {speed} kW')
                        ax.bar(f'Smart-{speed}', max(0, degradation_cost), bottom=max(0, electricity_cost), color=colors['smart'][1], alpha=0.7)
                        ax.bar(f'Smart-{speed}', min(0, electricity_cost), color=colors['smart'][0])
                        ax.bar(f'Smart-{speed}', min(0, degradation_cost), bottom=min(0, electricity_cost), color=colors['smart'][1], alpha=0.7)

                        # Add text annotations for sum of costs
                        total_cost = electricity_cost + degradation_cost
                        bar_height = max(0, electricity_cost) + max(0, degradation_cost)
                        ax.text(f'Smart-{speed}', barhight if total_cost > 0 else barhight, f"${total_cost:.2f}", ha='center', va='bottom' if total_cost > 0 else 'bottom', color='black', fontsize=13, rotation=90)

                # Plot V2G Charging
                for speed in ['6.6', '12', '19']:
                    v2g_subset = subset[(subset['Charging Type'] == 'v2g') & (subset['Charging Speed'] == float(speed))]
                    if not v2g_subset.empty:
                        electricity_cost = v2g_subset['Electricity_Cost'].values[0]
                        degradation_cost = v2g_subset['Degradation_Cost'].values[0]

                        ax.bar(f'V2G-{speed}', max(0, electricity_cost), color=colors['v2g'][0], label=f'V2G {speed} kW')
                        ax.bar(f'V2G-{speed}', max(0, degradation_cost), bottom=max(0, electricity_cost), color=colors['v2g'][1], alpha=0.7)
                        ax.bar(f'V2G-{speed}', min(0, electricity_cost), color=colors['v2g'][0])
                        ax.bar(f'V2G-{speed}', min(0, degradation_cost), bottom=min(0, electricity_cost), color=colors['v2g'][1], alpha=0.7)

                        # Add text annotations for sum of costs
                        total_cost = electricity_cost + degradation_cost
                        bar_height = max(0, electricity_cost) + max(0, degradation_cost)
                        ax.text(f'V2G-{speed}', barhight if total_cost > 0 else barhight, f"${total_cost:.2f}", ha='center', va='bottom' if total_cost > 0 else 'bottom', color='black', fontsize=13, rotation=90)


            # Set axis labels and title sizes
            ax.set_ylabel(y_axis_title, fontsize=title_size)
            ax.set_xlabel('Charging Scenario', fontsize=title_size)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=axis_text_size)
            ax.tick_params(axis='both', which='major', labelsize=axis_text_size)
            ax.grid(True)

        # Add column titles with grey background
    for ax, title in zip([axes[0, 0], axes[0, 1]], ['V2G Location: Home', 'V2G Location: Home and Work']):
        rect = plt.Rectangle((0, 1.05), 1, 0.13, color='grey', transform=ax.transAxes, clip_on=False)
        ax.add_patch(rect)
        ax.annotate(title, xy=(0.5, 1.08), xytext=(5, 0), textcoords='offset points', xycoords='axes fraction', ha='center', fontsize=14, weight='bold', color='white')

        # Add row titles with grey background
    for ax, title in zip([axes[0, 0], axes[1, 0]], ['Actual\nCharging Sessions', 'Potential\nCharging Sessions']):
        rect = plt.Rectangle((-0.38, 0), 0.15, 1.1, color='grey', transform=ax.transAxes, clip_on=False)
        ax.add_patch(rect)
        ax.annotate(title, xy=(-0.34, 0.54), xytext=(-0.34, 0.5), textcoords='offset points', xycoords='axes fraction', va='center', rotation='vertical', fontsize=14, weight='bold', color='white')

        # Add overall title
        # fig.suptitle('Cost Comparison under Different Charging Scenarios', fontsize=16, weight='bold')

        # Add legend
    handles = [
        plt.Rectangle((-1, 0), 20, 20, color=colors['Actual'][0]),
        plt.Rectangle((-1, 0), 20, 20, color=colors['smart'][0]),
        plt.Rectangle((-1, 0), 20, 20, color=colors['v2g'][0]),
        plt.Rectangle((-1, 0), 20, 20, color=colors['Actual'][1])
    ]
    labels = ['Actual Behavior \n Electricity Cost', 'Smart Charging \n Electricity Cost', 'V2G Charging \n Electricity Cost', 'Degradation Cost']
    fig.legend(handles, labels, loc='center right', fontsize=14)

    # Show the plot
    plt.tight_layout(rect=[0.05, 0, 0.8, 0.95])
    plt.show()


def plot_cost_comparison_RT(df, num_vehicles, title_size=14, axis_text_size=12, y_axis_title='Cost ($)', barhight = 500):
    df1 = df.copy()
    df1['Scenario'] = df1['Charging Type'] + ' - ' + df1['Charging Speed'].astype(str) + ' - ' + df1['Plugged-in Sessions'] + ' - ' + df1['V2G_Location']

    # Calculate average costs
    df1[["Electricity_Cost", "Degradation_Cost"]] = df1[["Electricity_Cost", "Degradation_Cost"]].div(num_vehicles, axis=0)

    # Extract the Actual Behavior rows
    actual_behavior_df = df1[(df1['Charging Type'] == 'Actual Behavior - TOU Rate') | (df1['Charging Type'] == 'Actual Behavior - EV Rate')]

    # Define the plot structure
    fig, axes = plt.subplots(2, 2, figsize=(16, 8), sharey=True)
    plt.subplots_adjust(hspace=0.4, wspace=0.4)

    # Define custom colors
    colors = {
        'Actual': ['#1f77b4', '#ff0000'],  # Blue for electricity, red for degradation
        'smart': ['#2ca02c', '#ff0000'],  # Different shade of green for electricity, red for degradation
        'v2g': ['#ff7f0e', '#ff0000']  # Different shade of orange for electricity, red for degradation
    }

    # Define the axes
    axes_dict = {
        'Home-Actual': axes[0, 0],
        'Home_Work-Actual': axes[0, 1],
        'Home-Potential': axes[1, 0],
        'Home_Work-Potential': axes[1, 1]
    }

    # Plotting each subplot
    for v2g_location in ['Home', 'Home_Work']:
        for plugged_in in ['Actual', 'Potential']:
            ax = axes_dict[f'{v2g_location}-{plugged_in}']
            subset = df1[(df1['V2G_Location'] == v2g_location) & (df1['Plugged-in Sessions'] == plugged_in)]

            # Include Actual Behavior rows in each subplot
            subset = pd.concat([subset, actual_behavior_df], ignore_index=True)

            # Plot Actual Behavior
            actual_subset_T = subset[subset['Charging Type'] == 'Actual Behavior - TOU Rate']
            if not actual_subset_T.empty:
                electricity_cost = actual_subset_T['Electricity_Cost'].values[0]
                degradation_cost = actual_subset_T['Degradation_Cost'].values[0]

                ax.bar('Actual-TOU', max(0, electricity_cost), color=colors['Actual'][0], label='Electricity Cost')
                ax.bar('Actual-TOU', max(0, degradation_cost), bottom=max(0, electricity_cost), color=colors['Actual'][1], label='Degradation Cost')
                ax.bar('Actual-TOU', min(0, electricity_cost), color=colors['Actual'][0])
                ax.bar('Actual-TOU', min(0, degradation_cost), bottom=min(0, electricity_cost), color=colors['Actual'][1])

                # Add text annotations for sum of costs
                total_cost = electricity_cost + degradation_cost
                bar_height = max(0, electricity_cost) + max(0, degradation_cost)
                ax.text('Actual-TOU', barhight if total_cost > 0 else -barhight, f"${total_cost:.2f}", ha='center', va='bottom' if total_cost > 0 else 'top', color='black', fontsize=13, rotation=90)

            # Plot Actual Behavior
            actual_subset_E = subset[subset['Charging Type'] == 'Actual Behavior - EV Rate']
            if not actual_subset_E.empty:
                electricity_cost = actual_subset_E['Electricity_Cost'].values[0]
                degradation_cost = actual_subset_E['Degradation_Cost'].values[0]

                ax.bar('Actual-EV Rate', max(0, electricity_cost), color=colors['Actual'][0], label='Electricity Cost')
                ax.bar('Actual-EV Rate', max(0, degradation_cost), bottom=max(0, electricity_cost), color=colors['Actual'][1], label='Degradation Cost')
                ax.bar('Actual-EV Rate', min(0, electricity_cost), color=colors['Actual'][0])
                ax.bar('Actual-EV Rate', min(0, degradation_cost), bottom=min(0, electricity_cost), color=colors['Actual'][1])

                # Add text annotations for sum of costs
                total_cost = electricity_cost + degradation_cost
                bar_height = max(0, electricity_cost) + max(0, degradation_cost)
                ax.text('Actual-EV Rate', barhight if total_cost > 0 else -barhight, f"${total_cost:.2f}", ha='center', va='bottom' if total_cost > 0 else 'top', color='black', fontsize=13, rotation=90)

                    # Plot Smart Charging
                for speed in ['6.6', '12', '19']:
                    smart_subset = subset[(subset['Charging Type'] == 'smart') & (subset['Charging Speed'] == float(speed))]
                    if not smart_subset.empty:
                        electricity_cost = smart_subset['Electricity_Cost'].values[0]
                        degradation_cost = smart_subset['Degradation_Cost'].values[0]

                        ax.bar(f'Smart-{speed}', max(0, electricity_cost), color=colors['smart'][0], label=f'Smart {speed} kW')
                        ax.bar(f'Smart-{speed}', max(0, degradation_cost), bottom=max(0, electricity_cost), color=colors['smart'][1], alpha=0.7)
                        ax.bar(f'Smart-{speed}', min(0, electricity_cost), color=colors['smart'][0])
                        ax.bar(f'Smart-{speed}', min(0, degradation_cost), bottom=min(0, electricity_cost), color=colors['smart'][1], alpha=0.7)

                        # Add text annotations for sum of costs
                        total_cost = electricity_cost + degradation_cost
                        bar_height = max(0, electricity_cost) + max(0, degradation_cost)
                        ax.text(f'Smart-{speed}', barhight if total_cost > 0 else barhight, f"${total_cost:.2f}", ha='center', va='bottom' if total_cost > 0 else 'bottom', color='black', fontsize=13, rotation=90)

                    # Plot V2G Charging
                for speed in ['6.6', '12', '19']:
                    v2g_subset = subset[(subset['Charging Type'] == 'v2g') & (subset['Charging Speed'] == float(speed))]
                    if not v2g_subset.empty:
                        electricity_cost = v2g_subset['Electricity_Cost'].values[0]
                        degradation_cost = v2g_subset['Degradation_Cost'].values[0]

                        ax.bar(f'V2G-{speed}', max(0, electricity_cost), color=colors['v2g'][0], label=f'V2G {speed} kW')
                        ax.bar(f'V2G-{speed}', max(0, degradation_cost), bottom=max(0, electricity_cost), color=colors['v2g'][1], alpha=0.7)
                        ax.bar(f'V2G-{speed}', min(0, electricity_cost), color=colors['v2g'][0])
                        ax.bar(f'V2G-{speed}', min(0, degradation_cost), bottom=min(0, electricity_cost), color=colors['v2g'][1], alpha=0.7)

                        # Add text annotations for sum of costs
                        total_cost = electricity_cost + degradation_cost
                        bar_height = max(0, electricity_cost) + max(0, degradation_cost)
                        ax.text(f'V2G-{speed}', barhight if total_cost > 0 else barhight, f"${total_cost:.2f}", ha='center', va='bottom' if total_cost > 0 else 'bottom', color='black', fontsize=13, rotation=90)

                # Set axis labels and title sizes
            ax.set_ylabel(y_axis_title, fontsize=title_size)
            ax.set_xlabel('Charging Scenario', fontsize=title_size)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=axis_text_size)
            ax.tick_params(axis='both', which='major', labelsize=axis_text_size)
            ax.grid(True)

        # Add column titles with grey background
    for ax, title in zip([axes[0, 0], axes[0, 1]], ['V2G Location: Home', 'V2G Location: Home and Work']):
        rect = plt.Rectangle((0, 1.05), 1, 0.13, color='grey', transform=ax.transAxes, clip_on=False)
        ax.add_patch(rect)
        ax.annotate(title, xy=(0.5, 1.08), xytext=(5, 0), textcoords='offset points', xycoords='axes fraction', ha='center', fontsize=14, weight='bold', color='white')

        # Add row titles with grey background
    for ax, title in zip([axes[0, 0], axes[1, 0]], ['Actual\nCharging Sessions', 'Potential\nCharging Sessions']):
        rect = plt.Rectangle((-0.38, 0), 0.15, 1.1, color='grey', transform=ax.transAxes, clip_on=False)
        ax.add_patch(rect)
        ax.annotate(title, xy=(-0.34, 0.54), xytext=(-0.34, 0.5), textcoords='offset points', xycoords='axes fraction', va='center', rotation='vertical', fontsize=14, weight='bold', color='white')

        # Add overall title
        # fig.suptitle('Cost Comparison under Different Charging Scenarios', fontsize=16, weight='bold')

        # Add legend
    handles = [
        plt.Rectangle((-1, 0), 20, 20, color=colors['Actual'][0]),
        plt.Rectangle((-1, 0), 20, 20, color=colors['smart'][0]),
        plt.Rectangle((-1, 0), 20, 20, color=colors['v2g'][0]),
        plt.Rectangle((-1, 0), 20, 20, color=colors['Actual'][1])
    ]
    labels = ['Actual Behavior \n Electricity Cost', 'Smart Charging \n Electricity Cost', 'V2G Charging \n Electricity Cost', 'Degradation Cost']
    fig.legend(handles, labels, loc='center right', fontsize=14)

    # Show the plot
    plt.tight_layout(rect=[0.05, 0, 0.8, 0.95])
    plt.show()


def violin_plot(df):
        # Create a color palette dictionary based on tariff types and social costs
        palette = {
            'RT Rate-0.05': '#6FCFEB',
            'RT Rate-0.191': '#FFDC00',
            'EV Rate-0.05': 'orange',
            'EV Rate-0.191': 'teal',
            'TOU Rate-0.05': '#FF8189',
            'TOU Rate-0.191': '#C6007E'
        }

        # Get unique scenarios and sort them if necessary
        unique_scenarios = sorted(df['Scenario'].unique())

        # Create a new figure for the plot
        plt.figure(figsize=(16, 8))

        # Separate the data based on the Tariff column
        ev_rate_data = df[df['Tariff'].str.contains('EV Rate')]
        rt_rate_data = df[df['Tariff'].str.contains('RT Rate')]
        tou_rate_data = df[df['Tariff'].str.contains('TOU Rate')]

        # Plot EV Rate data
        sns.violinplot(x='Scenario', y='Total_Cost',
                       hue='Tariff_Social_Cost',
                       data=ev_rate_data, bw=0.2,
                       split=True, inner='quart', linewidth=1.5,
                       palette=palette, width=1, dodge=True, order=unique_scenarios
                       )

        # Plot RT Rate data
        sns.violinplot(x='Scenario', y='Total_Cost',
                       hue='Tariff_Social_Cost',
                       data=rt_rate_data, bw=0.2,
                       split=True, inner='quart', linewidth=1.5,
                       palette=palette, width=1, dodge=True, order=unique_scenarios
                       )

        sns.violinplot(x='Scenario', y='Total_Cost',
                       hue='Tariff_Social_Cost',
                       data=tou_rate_data, bw=0.2,
                       split=True, inner='quart', linewidth=1.5,
                       palette=palette, width=1, dodge=True, order=unique_scenarios
                       )

        # Rotate x-axis labels and align them with their respective violins
        plt.xticks(ticks=np.arange(len(unique_scenarios)), labels=unique_scenarios, rotation=45, ha='right', fontsize=14)

        # Set plot labels and title
        plt.xlabel('Scenario', fontsize=16)
        plt.ylabel('Annual Cost / Revenue ($) Per Vehicle', fontsize=16)
        plt.title('', fontsize=14)
        # Add text box with scenario information
        textstr = 'All scenarios include different charging speeds and V2G charger locations.'
        plt.text(0.01, 0.3, textstr, transform=plt.gca().transAxes, fontsize=14,
                 verticalalignment='center', bbox=dict(facecolor='white', alpha=0.5))

        # Get the legend handles and labels
        handles, labels = plt.gca().get_legend_handles_labels()
        new_labels = []
        for label in labels:
            tariff, cost_per_kg = label.rsplit('-', 1)
            cost_per_tonne = float(cost_per_kg) * 1000
            new_label = f"{tariff} - {cost_per_tonne:.0f} $/tonne of CO₂"
            new_labels.append(new_label)

        # Manually add legend at the bottom of the plot with modified labels
        plt.legend(handles, new_labels, title='Tariff - Social Cost of Carbon', loc='lower left', fontsize=14, ncol=2, title_fontsize=16)

        # Increase y-axis tick font size
        plt.yticks(np.arange(-12000, 12000, 3000), fontsize=14)
        # Add horizontal grid lines
        plt.grid()

        # Adjust layout to fit legend properly
        plt.tight_layout()

        # Show plot
        plt.show()


def plot_ghg_distribution_seasons(GHG_dict, Power_dict):
    # Convert the dictionaries to DataFrame
    df = pd.DataFrame(GHG_dict.items(), columns=['Hour', 'GHG_per_MWh'])
    df3 = pd.DataFrame(Power_dict.items(), columns=['Hour', 'MWh'])

    # Sort by hour and select the first 8760 samples (one year of hourly data)
    df = df.sort_values(by='Hour').reset_index(drop=True)
    df = df.iloc[:8760]

    # Create a column for the hour of the day and for the day of the year
    df['Hour_of_Day'] = df.index % 24
    df['Day_of_Year'] = df.index // 24

    # Define seasons based on the day of the year (Winter, Spring, Summer, Fall)
    df['Season'] = pd.cut(df['Day_of_Year'],
                          bins=[0, 80, 172, 264, 355, 365],  # Adjusted bins to cover full year
                          labels=['Winter', 'Spring', 'Summer', 'Fall', 'Winter-End'],
                          right=False)

    # Remove the "Winter-End" label for the last days to avoid confusion
    df = df[df['Season'] != 'Winter-End']

    # Add the average MWh from df3 to the violin plot data
    df['Average_MWh'] = df3['MWh']

    # Normalize the Average MWh for color mapping
    norm = plt.Normalize(df['Average_MWh'].min(), df['Average_MWh'].max())
    sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=norm)

    # Plot for each season
    seasons = ['Winter', 'Spring', 'Summer', 'Fall']

    for season in seasons:
        plt.figure(figsize=(10, 6))
        plt.title(f'{season} GHG Distribution', fontsize=14)

        for hour in range(24):
            # Select data for the current hour and season
            data = df[(df['Hour_of_Day'] == hour) & (df['Season'] == season)]

            # Plot violin with color based on the average MWh for that hour
            sns.violinplot(x='Hour_of_Day', y='GHG_per_MWh', data=data,
                           inner='quart', linewidth=1.5,
                           palette=[sm.to_rgba(data['Average_MWh'].mean())],
                           scale='width', bw=0.2)

        # Set labels and limits for each subplot
        plt.xlabel('Hour', fontsize=16)
        plt.ylabel('kg of CO\u2082/MWh', fontsize=16)
        plt.xticks(range(0, 24, 2), fontsize=14)
        plt.yticks(range(50, 400, 50), fontsize=14)
        plt.xlim(-0.5, 23.5)
        plt.ylim(50, 400)
        plt.grid(axis="y")

        # Add a single color bar for each figure to reflect the generation mix (Average MWh)
        cbar = plt.colorbar(sm, ax=plt.gca())
        cbar.set_label('Average Electricity Generation (MWh)', fontsize=18)
        cbar.ax.tick_params(labelsize=16)

        plt.tight_layout()
        plt.show()

def add_tariff_name(df, tariff_name):
    df['Tariff'] = tariff_name
    return df


def add_tariff_name2(df, tariff_name, charging_behavior):
    df['Tariff'] = tariff_name
    df['Charging_Behavior'] = charging_behavior
    return df

def add_tariff_name3(df, tariff_name, charging_behavior):
    df['Tariff'] = tariff_name
    df['Charging_Behavior'] = charging_behavior
    df['GHG Cost'] = 0
    return df
# %%
from matplotlib.lines import Line2D
# Function to plot the stacked violin plots
# Function to plot the stacked violin plots
def stacked_violin_plot(df):
    # Create a color palette dictionary based on tariff types and social costs
    palette = {
        'EV Rate-0.05': 'orange',
        'EV Rate-0.191': 'teal',
        'RT Rate-0.05': '#6FCFEB',
        'RT Rate-0.191': '#FFDC00',
        'TOU Rate-0.05': '#FF8189',
        'TOU Rate-0.191': '#C6007E'
    }

    # Create a new column combining the Tariff and Social Cost for color coding
    df['Tariff_Social_Cost'] = df['Tariff'] + '-' + df['Social Cost of Carbon'].astype(str)

    # Get unique scenarios and sort them if necessary
    unique_scenarios = sorted(df['Scenario'].unique())

    # Create a new figure for the plot
    fig, ax = plt.subplots(figsize=(16, 8))

    # Collect handles and labels for the legend
    handles_list = []
    labels_list = []
    means_dict = {0: {}, 6.6: {}, 12: {}, 19: {}}  # Dictionary to store the mean values for each speed

    # Plot data for each tariff type and charging speed
    for speed in [0, 6.6, 12, 19]:
        for tariff in df['Tariff'].unique():
            subset = df[(df['Charging Speed'] == speed) & (df['Tariff'] == tariff)]
            if not subset.empty:
                sns.violinplot(x='Scenario', y='Total_Cost',
                               hue='Tariff_Social_Cost',
                               data=subset, bw=0.2,
                               split=True, inner='quart', linewidth=1.5,
                               palette=palette, width=1, dodge=True, order=unique_scenarios, ax=ax)

                # Add mean points in red without affecting the legend
                sns.pointplot(x='Scenario', y='Total_Cost', hue='Tariff_Social_Cost',
                              data=subset, dodge=0.5, join=False, palette={k: 'red' for k in palette.keys()},
                              markers='D', scale=0.75, order=unique_scenarios, legend=False)

                # Calculate the mean value and store in the dictionary
                mean_values = subset.groupby('Scenario')['Total_Cost'].mean().to_dict()
                means_dict[speed].update(mean_values)

                # Collect handles and labels for the legend
                handles, labels = ax.get_legend_handles_labels()
                handles_list.extend(handles)
                labels_list.extend(labels)

    # Remove duplicates from the legend
    by_label = dict(zip(labels_list, handles_list))
    new_labels = []
    for label in by_label.keys():
        tariff, cost_per_kg = label.rsplit('-', 1)
        cost_per_tonne = float(cost_per_kg) * 1000
        new_label = f"{tariff} - {cost_per_tonne:.0f} $/tonne of CO₂"
        new_labels.append(new_label)

    # Rotate x-axis labels and align them with their respective violins
    plt.xticks(ticks=np.arange(len(unique_scenarios)), labels=unique_scenarios, rotation=45, ha='right', fontsize=14)

    # Set plot labels and title
    plt.xlabel('Scenario', fontsize=16)
    plt.ylabel('Annual Cost / Revenue ($) Per Vehicle', fontsize=16)
    plt.title('', fontsize=14)

    # Add text box with scenario information
    textstr = 'All scenarios include different charging speeds and V2G charger locations.'
    plt.text(0.01, 0.94, textstr, transform=plt.gca().transAxes, fontsize=14,
             verticalalignment='center', bbox=dict(facecolor='white', alpha=0.5))

    # Create custom legend entries for the mean points for each speed
    mean_legends = []
    for speed in means_dict:
        for scenario, mean_value in means_dict[speed].items():
            if speed in [0]:  # For the first two scenarios, exclude the speed
                label = f'{scenario}: ${mean_value:.1f}'
            else:  # Include the speed for other scenarios
                label = f'{scenario}-{speed} kW: ${mean_value:.1f}'
            mean_legends.append(Line2D([0], [0], color='red', marker='D', linestyle='None', markersize=8, label=label))

    # Manually add the main legend at the bottom of the plot with modified labels
    main_legend = ax.legend(by_label.values(), new_labels, title='Tariff - Social Cost of Carbon', loc='lower left', fontsize=11, ncol=1, title_fontsize=14, bbox_to_anchor=(1, 0.5))
    ax.add_artist(main_legend)

    # Add custom legend for mean points outside the box to the left
    # plt.legend(handles=mean_legends, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=11, title='Mean Points', title_fontsize=14)

    # Increase y-axis tick font size
    plt.yticks(np.arange(-0, 9000, 2000), fontsize=14)
    # Add horizontal grid lines
    plt.grid()

    # Adjust layout to fit legend properly
    plt.tight_layout()
    plt.tight_layout(rect=[0.0, 0, 0.82, 0.98])
    # Show plot
    plt.show()


# Example usage
# Assuming violin_input_data is your prepared DataFrame
# violin_input_data = violin_input(All_rates_total)
# stacked_violin_plot(violin_input_data)
# Call the function
# # stacked_violin_plot(voilin_input_data)
# stacked_violin_plot(voilin_input_data_V2G)
# stacked_violin_plot(voilin_input_data_smart)

# %%
#
# def xlsx_read(dic):
#
#     # List of Excel file names
#     excel_files = [f for f in os.listdir(dic) if f.endswith('.xlsx')]
#
#     # Dictionary to store dataframes
#     all_dataframes = {}
#
#     # Iterate over each Excel file
#     for excel_file_name in excel_files:
#         excel_file_path = os.path.join(dic, excel_file_name)
#         print(f"Reading Excel file '{excel_file_path}'...")
#
#         # Read each sheet into a separate dataframe
#         with pd.ExcelFile(excel_file_path) as xls:
#             sheet_names = xls.sheet_names  # Get the names of all sheets in the Excel file
#
#             # Read each sheet into a dataframe and store it in the dictionary
#             for sheet_name in sheet_names:
#                 df = pd.read_excel(xls, sheet_name=sheet_name)
#                 new_df_name = f"{excel_file_name[:-5]}_{sheet_name}"  # Add sheet name to the file name
#                 all_dataframes[new_df_name] = df
#
#     # Create a new dataframe to store total cost data
#     total_costs_df = pd.DataFrame()
#
#     # Iterate over the dataframes and extract total costs
#     for df_name, df in all_dataframes.items():
#         if "Total Costs" in df_name:
#             # Extract charging type and speed from the dataframe name
#             charging_type = "smart" if "smart" in df_name else "v2g"
#             charging_speed = df_name.split("_")[2][:-1]
#             ghg_cost = df_name.split("_")[3][:-2]
#
#             # Add a column indicating the charging type (smart or v2g)
#             df['Charging Type'] = charging_type
#             # Add columns indicating charging speed and GHG cost
#             df['Charging Speed'] = charging_speed
#             df['GHG Cost'] = ghg_cost
#
#             # Concatenate this dataframe with the total_costs_df
#             total_costs_df = pd.concat([total_costs_df, df])
#
#     print("Total cost data has been extracted.")
#     total_costs_df = total_costs_df.reset_index(drop=True)
#     # Display the new dataframe
#     print(total_costs_df)
#
#     # Create a new dataframe to store total cost data
#     individual_cost_df = pd.DataFrame()
#
#     # Iterate over the dataframes and extract total costs
#     for df_name, df in all_dataframes.items():
#         if "Individual Cost" in df_name:
#             # Extract charging type and speed from the dataframe name
#             charging_type = "smart" if "smart" in df_name else "v2g"
#             charging_speed = df_name.split("_")[2][:-1]
#             ghg_cost = df_name.split("_")[3][:-2]
#
#             # Add a column indicating the charging type (smart or v2g)
#             df['Charging Type'] = charging_type
#             # Add columns indicating charging speed and GHG cost
#             df['Charging Speed'] = charging_speed
#             df['GHG Cost'] = ghg_cost
#
#             # Concatenate this dataframe with the total_costs_df
#             individual_cost_df = pd.concat([individual_cost_df, df])
#
#     individual_cost_df = individual_cost_df.reset_index(drop=True)
#     # Display the new dataframe
#     return total_costs_df


# %%

def plotting(df, num_vehicles):
    df1 = df.copy()
    # Create a new column called `stacked_index` by concatenating the columns `Charging Type`, `Charging Speed`, and `GHG Cost` with hyphens as separators.
    df1['stacked_index'] = df1['Charging Type'] + ' - ' + df1['Charging Speed'] + ' - ' + df1['GHG Cost'].astype(str) + df1.apply(lambda row: '' if row['Charging Type'] == 'smart' else ' - ' + row['V2G Location'], axis=1)

    # Define custom order for charging speed and ghg cost
    charging_speed_order = ['6.6', '12', '19']
    ghg_cost_order = ['0.05', '0.191']

    # Create categorical columns with the specified orders
    df1['Charging Speed'] = pd.Categorical(df1['Charging Speed'], categories=charging_speed_order, ordered=True)
    df1['GHG Cost'] = pd.Categorical(df1['GHG Cost'], categories=ghg_cost_order, ordered=True)

    # Sort by charging type first, then by the ordered categorical columns
    df1 = df1.sort_values(by=['V2G Location', 'Charging Type', 'Charging Speed', 'GHG Cost'])
    df1[["Electricity_Cost", "Degradation_Cost", "GHG_Cost", "X_CHR"]] = df1[["Electricity_Cost", "Degradation_Cost", "GHG_Cost", "X_CHR"]].div(num_vehicles, axis=0)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    components = ['Electricity_Cost', 'Degradation_Cost', 'GHG_Cost']
    colors = ["#008EAA", "#C10230", "#8A532F"]
    bottom_pos = [0] * len(df1)
    bottom_neg = [0] * len(df1)

    # Keep track of plotted components to avoid duplicate legends
    plotted_components = set()

    # Calculate total cost per stack
    total_costs = df1[components].sum(axis=1)

    for i, component in enumerate(components):
        if component not in plotted_components:
            ax1.bar(df1['stacked_index'], df1[component].clip(lower=0), color=colors[i], bottom=bottom_pos, label=component)
            ax1.bar(df1['stacked_index'], df1[component].clip(upper=0), color=colors[i], bottom=bottom_neg)
            plotted_components.add(component)

        # Update bottom values for stacking (using stacked_index)
        bottom_pos = [bottom_pos[j] + df1[component].clip(lower=0).iloc[j] for j in range(len(df1))]
        bottom_neg = [bottom_neg[j] + df1[component].clip(upper=0).iloc[j] for j in range(len(df1))]

    # Plot total cost line (on ax1, the primary axis)
    ax1.plot(df1['stacked_index'], total_costs, color='black', marker='o', linestyle='-', label='Total Cost')

    # Rotate x-axis labels (optional)
    plt.xticks(rotation=45, ha='right')

    # Set other labels, title, legends, grid, and layout (with adjustments for new legend)
    ax1.set_ylabel('Cost/Revenue ($)')
    ax1.set_xlabel('Charging Scenarios')
    ax1.set_ylim(-3500, 3500)
    ax2 = ax1.twinx()
    ax2.plot(df1['stacked_index'], df1['X_CHR'], color='blue', label='Charging Demand')
    ax2.set_ylim(0, 5000)  # Adjust the secondary y-axis limits if needed
    ax2.set_ylabel('Charging Demand (kWh)', color='blue')

    # Set the tick parameters for the second y-axis to be blue
    ax2.tick_params(axis='y', colors='blue')
    ax2.spines['right'].set_color('blue')

    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='lower left')

    ax1.set_title(f'Average Cost Breakdown per Vehicle (for {num_vehicles} BEVs) Under Different Scenarios')

    ax1.grid(True)
    # Remove the grid from the secondary axis (ax2)
    ax2.grid(False)
    # Add V2G location annotations

    plt.subplots_adjust(bottom=0.35)
    # plt.savefig('plot_output.png')
    plt.show()

# %%


def draw_RT(df):

    # Create a sample array representing electricity price for 8760 hours
    electricity_price = [] # Random values between 0 and 1
    for key, value in df.items():
        electricity_price.append(value)
    electricity_price = electricity_price[:8760]
    # Create time labels for the x-axis (assuming hourly data)
    hours = np.arange(0, 8760, 1)  # Array of hours from 0 to 8759

    # Plot the electricity price line
    plt.figure(figsize=(10, 6))
    plt.plot(hours, electricity_price, label='Electricity Price ($ / MWh)')

    # Set labels and title
    plt.xlabel('Hour', fontsize=18)
    plt.ylabel('Electricity Price ($ / MWh)',fontsize=18)
    # plt.title('Electricity Price for year 2021 - PG&E territory ')

    # Add grid and legend
    plt.grid(True)
    plt.legend(fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlim(0,8760)
    # Show the plot
    plt.tight_layout()
    plt.show()

# %%

def draw_util(df):
    # Assuming you have a DataFrame named 'df' with columns 'Charging_Speed', 'GHG_Cost', and 'Peak'
    # Filter data for peak and non-peak hours
    peak_data = df[df['Peak'] == 'Peak']
    non_peak_data = df[df['Peak'] == 'Non-Peak']

    # Create the box plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

    # Box plot for peak hours
    sns.boxplot(x='Charging Speed', y='Utilization Rate', hue='GHG Cost', data=peak_data, ax=axes[0], whis=[10, 90])
    axes[0].set_title('Peak Hours')
    axes[0].set_xlabel('Discharging Speed')
    axes[0].set_ylabel('%')

    # Box plot for non-peak hours
    sns.boxplot(x='Charging Speed', y='Utilization Rate', hue='GHG Cost', data=non_peak_data, ax=axes[1], whis=[10, 90])
    axes[1].set_title('Non-Peak Hours')
    axes[1].set_xlabel('Discharging Speed')
    axes[1].set_ylabel('%')
    # Add annotation explaining whiskers
    annotation_text = "Whiskers extend to the 5th and 95th percentiles of the data"
    plt.annotate(annotation_text, xy=(0.5, -0.15), xytext=(0, -50), ha='center', fontsize=12,
                 xycoords='axes fraction', textcoords='offset points', bbox=dict(facecolor='white', alpha=0.5))

    plt.tight_layout()
    plt.show()

# %%


def draw_util_rt(df):
    # Assuming you have a DataFrame named 'df' with columns 'Charging_Speed', 'GHG_Cost', 'Peak', and 'Utilization Rate'

    # Create a new column to distinguish between peak and non-peak hours
    df['Hour Type'] = df['Peak'].map({'Peak': 'Peak Hours', 'Non-Peak': 'Non-Peak Hours'})

    # Create the box plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Charging Speed', y='Utilization Rate', hue='GHG Cost', data=df, whis=[5, 95])
    plt.title('Utilization Rate by Charging Speed')
    plt.xlabel('Charging Speed')
    plt.ylabel('Utilization Rate (%)')

    # Add annotation explaining whiskers
    annotation_text = "Whiskers extend to the 5th and 95th percentiles of the data"
    plt.annotate(annotation_text, xy=(0.5, -0.15), xytext=(0, -50), ha='center', fontsize=12,
                 xycoords='axes fraction', textcoords='offset points', bbox=dict(facecolor='white', alpha=0.5))

    plt.tight_layout()
    plt.show()

# %%


def draw_profile(charging_cost, hourly_data):

    combined_price_PGE_average_df = pd.DataFrame([charging_cost])
    combined_price_PGE_average_df = combined_price_PGE_average_df.T
    combined_price_PGE_average_df = combined_price_PGE_average_df.copy()
    combined_price_PGE_average_df = combined_price_PGE_average_df.rename(columns={0: 'Price'}).reset_index(drop=False)
    combined_price_PGE_average_df = combined_price_PGE_average_df.rename(columns={'index': 'Hour'})
    combined_price_PGE_average_df["Hour"] = combined_price_PGE_average_df["Hour"].astype(int)
    hourly_data_1087 = hourly_data[(hourly_data["Vehicle"] == "P_1087") & (hourly_data["Charging Type"] == "v2g") & (hourly_data["Charging Speed"] == 6.6)]
    # Data Preparation

    optimal_result = pd.merge(hourly_data_1087, combined_price_PGE_average_df, on='Hour', how='left').copy()
    optimal_result['Hour'] = pd.to_numeric(optimal_result['Hour']) + 1

    # Specify the year
    year = 2021

    # Convert the hour index to timedelta
    optimal_result['Hour_timedelta'] = pd.to_timedelta(optimal_result['Hour'], unit='h')

    # Add the timedelta to the start of the year to get datetime values
    optimal_result['Datetime'] = pd.Timestamp(year=year, month=1, day=1) + optimal_result['Hour_timedelta']
    optimal_result = optimal_result[8025:8700].reset_index(drop=True)

    # Plotting
    fig, ax1 = plt.subplots(figsize=(12, 10))

    # Plot each vehicle as a separate line on the primary y-axis
    for vehicle in optimal_result['Vehicle'].unique():
        vehicle_data = optimal_result[optimal_result['Vehicle'] == vehicle]
        ax1.plot(vehicle_data['Datetime'], vehicle_data['X_CHR'], label=vehicle, marker='o')

    ax1.set_xlabel('Datetime', fontsize=20)
    ax1.set_ylabel('Charging (kW)', fontsize=20)
    ax1.set_title('Vehicle Charging vs Electricity Price', fontsize=20)
    ax1.legend(loc="upper left", fontsize=20)
    ax1.set_ylim([0, 150])
    # Increase the size of the tick labels
    ax1.tick_params(axis='both', which='major', labelsize=12)
    # Create a secondary y-axis for electricity price
    ax2 = ax1.twinx()

    # Plot the electricity price on the secondary y-axis
    ax2.plot(optimal_result['Datetime'], optimal_result['Price']/1000, label='Electricity Price', color='red', linestyle='dashed', marker='x')
    ax2.set_ylabel('Price ($/kWh)', fontsize=20)
    # Increase the size of the tick labels on the secondary y-axis
    ax2.tick_params(axis='both', which='major', labelsize=18)
    # Combine the legends
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + [lines2[0]], labels + [labels2[0]], loc='upper center', fontsize=18)
    ax2.set_ylim([0.2, 0.6])
    plt.grid(axis='y', linestyle='--')
    plt.tight_layout()

    plt.show()

# %%

#
# def demand_response(df, include_locations):
#     # Ensure exclude_locations is a list
#     if not isinstance(include_locations, list):
#         include_locations = [include_locations]
#
#     # Filter the DataFrame to exclude specified locations
#     df = df[df["location"].isin(include_locations)].copy()
#     df["%10increase"] = df["Total_power"] * 1.10 - df["Total_power"]
#     df["%20increase"] = df["Total_power"] * 1.20 - df["Total_power"]
#     df["%50increase"] = df["Total_power"] * 1.50 - df["Total_power"]
#
#     df["%10response"] = (df["Total_power"] - (df["Utilization Rate"]/100) * df["Total_power"]) > df["%10increase"]
#     df["%20response"] = (df["Total_power"] - (df["Utilization Rate"]/100) * df["Total_power"]) > df["%20increase"]
#     df["%50response"] = (df["Total_power"] - (df["Utilization Rate"]/100) * df["Total_power"]) > df["%50increase"]
#
#     # Group by 'hour' and calculate percentages
#     grouped10 = df.groupby(['Hour_of_day', "Charging Speed"])['%10response'].value_counts(normalize=True).unstack().fillna(0) * 100
#     grouped20 = df.groupby(['Hour_of_day', "Charging Speed"])['%20response'].value_counts(normalize=True).unstack().fillna(0) * 100
#     grouped50 = df.groupby(['Hour_of_day', "Charging Speed"])['%50response'].value_counts(normalize=True).unstack().fillna(0) * 100
#
#     # Rename columns for clarity
#     grouped10.columns = ['Percentage_False', 'Percentage_True']
#     grouped20.columns = ['Percentage_False', 'Percentage_True']
#     grouped50.columns = ['Percentage_False', 'Percentage_True']
#
#     # Reset index to make 'hour' a column again
#     grouped10 = grouped10.reset_index()
#     grouped20 = grouped20.reset_index()
#     grouped50 = grouped50.reset_index()
#
#     return grouped10, grouped20, grouped50
#

def demand_response(df, include_locations):
    # Ensure include_locations is a list
    if not isinstance(include_locations, list):
        include_locations = [include_locations]

    # Filter the DataFrame to include specified locations
    df = df[df["location"].isin(include_locations)].copy()

    # Ensure the DataFrame is sorted for efficient grouping
    df = df.sort_values(by=['Hour_of_day', 'Charging Speed'])

    increments = range(1, 101)  # 1% to 100%
    results = []

    for inc in increments:
        increase_amount = df["Total_power"] * (inc / 100)
        response = (df["Total_power"] - (df["Utilization Rate"] / 100) * df["Total_power"]) > increase_amount

        grouped = df.groupby(['Hour_of_day', 'Charging Speed'], sort=False, group_keys=False).apply(
            lambda x: pd.Series({
                'Percentage_False': (~response[x.index]).mean() * 100,
                'Percentage_True': response[x.index].mean() * 100
            }), include_groups=False
        ).reset_index()

        grouped['Increase_Percentage'] = inc
        results.append(grouped)

    # Concatenate all results into a single DataFrame
    final_df = pd.concat(results, ignore_index=True)
    return final_df


# Assuming All_rates_total is your DataFrame with columns: Charging Type, Plugged-in Sessions, Electricity_Cost, Degradation_Cost, GHG_Cost
# Calculate Total Cost
def violin_input(df):
    df["Charging Speed"] = df["Charging Speed"].astype(float)
    df["GHG Cost"] = df["GHG Cost"].astype(float)
    # df.loc[df["GHG Cost"] == 0, "GHG Optimized"] = "False"
    # df.loc[df["GHG Cost"] != 0, "GHG Optimized"] = "Ture"
    df['Total_Cost'] = df['Electricity_Cost'] + df['Degradation_Cost'] + df['GHG_Cost']
    df.loc[df['Charging Type'] == "smart", "Charging Type"] = "Smart"
    df.loc[df['Charging Type'] == "v2g", "Charging Type"] = "V2G"
    df.loc[(df['Charging Type'] == "V2G") | (df['Charging Type'] == "Smart"), "Scenario"] = df['Charging Type'] + '-' + df['Plugged-in Sessions'] + '-' + df['Tariff']
    df.loc[(df['Charging Type'] == "Actual - TOU") | (df['Charging Type'] == "Actual - EV Rate"), "Scenario"] = "Actual" + '-' + df['Plugged-in Sessions'] + '-' + df['Tariff']

    # Calculate Total Cost
    df['Social Cost of Carbon'] = df['GHG Cost']
    df.loc[df['Social Cost of Carbon'] == 0, "Social Cost of Carbon"] = 0.05
    df_long = df[["Scenario", "Charging Speed", "Total_Cost", "Social Cost of Carbon", "Tariff"]].sort_values(by='Scenario')
    # Create a new column for the combined hue
    df_long['Tariff_Social_Cost'] = df_long['Tariff'] + '-' + df_long['Social Cost of Carbon'].astype(str)
    df_long = df_long.sort_values(["Scenario", "Charging Speed", "Social Cost of Carbon", "Tariff"])
    df_long["Charging Speed"].fillna(0, inplace=True)
    return df_long


def filter_rows_by_word(df, column_name, words):
    # Create a boolean mask for rows that match any of the specified words
    mask = df[column_name].apply(lambda x: x.split('-')[0].strip() in words)

    # Return the filtered DataFrame
    return df[mask]

def process_charging_data(df):
    # Group by initial columns and sum X_CHR
    grouped_df = df.groupby(["Hour", "Charging Type", "Charging Speed", "GHG Cost", "Tariff", "Charging_Behavior"])["X_CHR"].sum().reset_index()

    # Add Hour_Day column
    grouped_df["Hour_Day"] = grouped_df["Hour"] % 24

    # Group by Hour_Day and other columns, then divide by 50
    final_grouped_df = grouped_df.groupby(["Hour_Day", "Charging Type", "Charging Speed", "GHG Cost", "Tariff", "Charging_Behavior"])["X_CHR"].mean().reset_index()
    final_grouped_df["X_CHR"] = final_grouped_df["X_CHR"] / 50

    return final_grouped_df


def process_charging_data1(df):
    # df = all_hourly_charging_N_data.copy()
    # Group by initial columns and sum X_CHR
    df = df[(df["X_CHR"] > 0) | (df["X_CHR"] < 0)]
    df["daily_hour"] = df["Hour"] % 24

    grouped_df = df.groupby(["Vehicle", "daily_hour", "Charging Type", "Charging Speed", "GHG Cost", "Tariff", "Charging_Behavior", "charging_indicator"])["X_CHR"].sum().reset_index()

    # Group by Hour_Day and other columns, then divide by 50
    final_grouped_df = grouped_df.groupby(["daily_hour", "Charging Type", "Charging Speed", "GHG Cost", "Tariff", "Charging_Behavior", "charging_indicator"])["X_CHR"].mean().reset_index()

    return final_grouped_df

def plotting_demand_heatmap(df_actual, df_potential, charging_type, color_palette="viridis"):
    # Filter data by Charging Type
    df_actual_type = df_actual[df_actual["Charging Type"] == charging_type]
    df_potential_type = df_potential[df_potential["Charging Type"] == charging_type]

    # Pivot the data for heatmap plotting
    df_actual_pivot = df_actual_type.pivot_table(index=["Charging Speed", "Tariff"], columns="Hour_Day", values="X_CHR", aggfunc='sum').fillna(0)
    df_potential_pivot = df_potential_type.pivot_table(index=["Charging Speed", "Tariff"], columns="Hour_Day", values="X_CHR", aggfunc='sum').fillna(0)

    # Ensure the data for all Charging Speeds is available
    speeds = sorted(set(df_actual_type["Charging Speed"]).union(set(df_potential_type["Charging Speed"])))
    n_speeds = len(speeds)

    fig, axes = plt.subplots(n_speeds, 2, figsize=(15, 4 * n_speeds), sharex=True, sharey=True)

    for i, speed in enumerate(speeds):
        # Plot for Actual data
        if speed in df_actual_pivot.index:
            sns.heatmap(df_actual_pivot.loc[speed], ax=axes[i, 0], cmap=color_palette, cbar=False, vmin=df_actual["X_CHR"].min(), vmax=df_actual["X_CHR"].max())
            axes[i, 0].set_title(f'Actual - Speed: {speed}', fontsize=14)
            axes[i, 0].set_ylabel('', fontsize=12, labelpad=20)
            axes[i, 0].set_xlabel('', fontsize=12, labelpad=20)
            axes[i, 0].tick_params(axis='both', which='major', labelsize=14)
            axes[i, 0].set_xticks(range(0, 24, 2))
            axes[i, 0].set_xticklabels(range(0, 24, 2), fontsize=14)
        # else:
        #     axes[i, 0].set_visible(False)

        # Plot for Potential data
        if speed in df_potential_pivot.index:
            sns.heatmap(df_potential_pivot.loc[speed], ax=axes[i, 1], cmap=color_palette, cbar=False, vmin=df_potential["X_CHR"].min(), vmax=df_potential["X_CHR"].max())
            axes[i, 1].set_title(f'Potential - Speed: {speed}', fontsize=14)
            axes[i, 1].tick_params(axis='both', which='major', labelsize=14)
            axes[i, 1].set_ylabel('', fontsize=12, labelpad=20)
            axes[i, 1].set_xlabel('', fontsize=12, labelpad=20)
            axes[i, 1].set_xticks(range(0, 24, 2))
            axes[i, 1].set_xticklabels(range(0, 24, 2), fontsize=14)
        # else:
        #     axes[i, 1].set_visible(False)

    fig.text(0.5, 0.04, 'Hour of the Day', ha='center', fontsize=16)
    fig.text(0.02, 0.5, 'Tariff', va='center', rotation='vertical', fontsize=16)
    plt.subplots_adjust(left=0.2, right=0.88, top=0.9, bottom=0.1, wspace=0.3, hspace=0.3)
    plt.suptitle(f'Load Demand for Each Hour by Tariff and Charging Speed ({charging_type.capitalize()})', fontsize=20)

    # Create a common color bar
    cbar_ax = fig.add_axes([.91, .3, .03, .4])
    norm = plt.Normalize(min(df_actual_type["X_CHR"].min(), df_potential_type["X_CHR"].min()), max(df_actual_type["X_CHR"].max(), df_potential_type["X_CHR"].max()))
    sm = plt.cm.ScalarMappable(cmap=color_palette, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax)

    plt.show()
# %%

def dr_plot(data1, data2, data3, data4, data5, data6):
    dfs = [data1, data2, data3, data4, data5, data6]
    fig, axes = plt.subplots(2, 6, figsize=(20, 12), sharey=True)
    plt.subplots_adjust(hspace=0.4, wspace=0.3)  # Adjust space between rows and columns

    bar_width = 1  # Set bar width to reduce distance between bars

    # Define the colors
    blue_color = sns.color_palette("Blues", 10)[5]  # Ensure we get the 6th shade
    red_color = sns.color_palette("Reds", 10)[6]    # Ensure we get the 7th shade

    titles = ["Actual EV Rate", "Actual TOU Rate", "Actual RT Rate", "Potential EV Rate", "Potential TOU Rate", "Potential RT Rate"]

    for row in range(2):
        for col in range(6):
            dataset_index = (col // 2) + (3 * row)
            df = dfs[dataset_index]
            charging_speed = [6.6, 12]  # Fixed charging speeds
            speed_index = col % 2
            batt_cap = charging_speed[speed_index]

            data_filtered = df[df['Charging Speed'] == batt_cap]

            axes[row, col].bar(data_filtered['Hour_of_day'], data_filtered['Percentage_False'] / 100, color=red_color, width=bar_width, label='False', hatch='\\')
            axes[row, col].bar(data_filtered['Hour_of_day'], data_filtered['Percentage_True'] / 100, bottom=data_filtered['Percentage_False'] / 100, color=blue_color, width=bar_width, label='True', hatch='//')
            axes[row, col].set_title(f'{titles[dataset_index]} - Speed {batt_cap} kW', fontsize=10)
            axes[row, col].set_xticks(range(0, 24, 4))  # Show every other hour for readability
            axes[row, col].set_yticks([i/10 for i in range(0, 11, 2)])  # Show every other hour for readability
            # axes[row, col].set_xticklabels(range(0, 24, 2), fontsize=10)
            # axes[row, col].set_yticklabels([i/10 for i in range(0, 11, 2)], fontsize=10)
            # axes[row, col].set_xlabel('Hour of Day', fontsize=10)
            if col == 0:
                axes[row, col].set_ylabel('Response Probability to DR Signal', fontsize=14)

            if row == 0 and col == 0:
                axes[row, col].legend()

    fig.supxlabel('Hour of Day', fontsize=14)
    sns.despine()
    plt.tight_layout()
    plt.show()

def plot_contour(df):
    # Prepare the data for the contour plot
    pivot_dfs = {}
    for charging in df['Charging'].unique():
        pivot_dfs[charging] = df[df['Charging'] == charging].pivot_table(values='Percentage_True', index='Increase_Percentage', columns='Hour_of_day').fillna(0)

    # Plotting
    plt.figure(figsize=(14, 8))

    colors = sns.color_palette("viridis", len(pivot_dfs))

    for color, (charging, pivot_df) in zip(colors, pivot_dfs.items()):
        # Create the contour plot
        X, Y = np.meshgrid(pivot_df.columns, pivot_df.index)
        Z = pivot_df.values

        # Determine appropriate levels for contour
        min_value = df['Percentage_True'].min()
        # min_value = 10
        max_value = df['Percentage_True'].max()
        # max_value = 90
        levels = np.linspace(min_value, max_value, num=8)  # Create 10 levels between min and max values

        contour = plt.contour(X, Y, Z, levels=levels, colors=[color])
        plt.contourf(X, Y, Z, levels=levels, cmap="magma", alpha=0.8)  # Adjust alpha for overlapping contours

        # Add contour labels
        # plt.clabel(contour, inline=True, fontsize=10, fmt={l: charging for l in contour.levels})

    # Add color bar
    plt.colorbar(label='Percentage True')

    # Adding labels and title
    plt.xlabel('Hour of the Day', fontsize=14)
    plt.ylabel('Increase Percentage', fontsize=14)
    plt.title('Contour Plot of Percentage True with Charging Scenarios as Contours', fontsize=16)

    # Add legend for Charging scenarios
    handles = [plt.Line2D([0], [0], color=color, lw=2, label=charging) for color, charging in zip(colors, pivot_dfs.keys())]
    plt.legend(handles=handles, title="Charging Scenarios")

    plt.show()


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


def load_and_clean_data(file_path):
    # Load data from pickle file
    data = pd.read_pickle(file_path)
    # Set negative X_CHR values to 0
    data.loc[data["X_CHR"] < 0, "X_CHR"] = 0
    # Filter out rows with X_CHR less than 0 (after setting negatives to 0)
    data = data[data["X_CHR"] >= 0]

    return data


def group_charging_data(data):
    grouped_data = data.groupby(
        ["Vehicle", "Charging Type", "Charging Speed", "GHG Cost", 'Tariff', 'Charging_Behavior'])[["X_CHR", "Electricity_Cost", "Degradation_Cost", "GHG_Cost"]].sum().reset_index()

    return grouped_data


# Function to calculate years to reach 80% battery capacity
def calculate_years_to_80_percent(bt_cap_rem, annual_consumption):
    years = 0
    while bt_cap_rem > 0.8:  # Stop when battery capacity drops to 80% or less
        bt_cap_rem -= 1.42e-6 * annual_consumption  # Subtract annual consumption
        years += 1
    return years


# Function to process and clean the hourly charging data
def process_hourly_charging_data(data_N, data_P):
    # Concatenate N and P datasets
    data = pd.concat([data_N, data_P])
    # Filter rows based on conditions
    data = data[data["GHG Cost"] > 0]
    data = data[data["Charging Speed"] != 19]
    data = data[~data["Tariff"].str.contains("Home&Work")]
    # Calculate battery remaining capacity
    data["bt_cap_rem"] = -1.42e-6 * data["X_CHR"] + 0.989999999
    # Calculate years to 80% capacity
    data["years_to_80_percent"] = data.apply(lambda row: calculate_years_to_80_percent(row["bt_cap_rem"], row["X_CHR"]), axis=1)
    data.loc[data["years_to_80_percent"] > 15, "years_to_80_percent"] = 15
    return data


# Function to calculate smart average and merge
def add_smart_avg(data):
    # Calculate the average years for 'smart' charging type
    smart_avg = data[data["Charging Type"] == "smart"].groupby("Vehicle")["years_to_80_percent"].mean().reset_index()
    smart_avg = smart_avg.rename(columns={"years_to_80_percent": "average_smart_years"})
    # Merge average smart years back to the original data
    data = pd.merge(data, smart_avg, on="Vehicle", how="left")
    # Calculate percentage drop
    data["percentage_drop"] = ((data["average_smart_years"] - data["years_to_80_percent"]) / data["average_smart_years"]) * 100

    return data


# Function to merge costs and calculate total
def merge_and_calculate_costs(data, actual_cost, bev_distance):
    # Merge BEV distance
    data = pd.merge(data, bev_distance, left_on="Vehicle", right_on="vehicle_name", how="left")
    # Remove '- Home' from 'Tariff' column
    data["Tariff"] = data["Tariff"].str.replace('- Home', '', regex=False)
    # Merge TOU cost
    actual_cost_tou = actual_cost[actual_cost["Tariff"] == "TOU Rate"]
    data = pd.merge(data, actual_cost_tou[["Vehicle", "GHG Cost", "Total_cost"]], on=["Vehicle", "GHG Cost"], how="left", suffixes=('', '_TOU'))
    data["TOU Cost"] = data["Total_cost"]  # Assign TOU cost
    # Merge EV cost
    actual_cost_ev = actual_cost[actual_cost["Tariff"] == "EV Rate"]
    data = pd.merge(data, actual_cost_ev[["Vehicle", "GHG Cost", "Total_cost"]], on=["Vehicle", "GHG Cost"], how="left", suffixes=('', '_EV'))
    data["EV Cost"] = data["Total_cost_EV"]  # Assign EV cost
    # Drop unnecessary columns
    data.drop(["Total_cost", "Total_cost_EV"], axis=1, inplace=True, errors='ignore')
    # Calculate the final total cost
    data["Total_cost"] = data["Electricity_Cost"] + data["Degradation_Cost"] + data["GHG_Cost"]

    return data


def calculate_battery_price_per_kwh(year, bat_cap, df_price_estimations):
    # Find the price for the given year
    price_row = df_price_estimations[df_price_estimations['Year'] == year]

    if not price_row.empty:
        price_per_kwh = price_row['Price_per_kWh'].values[0]
    else:
        # If the year is not available, use the last available price
        price_per_kwh = df_price_estimations['Price_per_kWh'].iloc[-1]

    return bat_cap * price_per_kwh

# %%


def calculate_future_battery_price(cycle_number, current_year, bat_cap, cycles, df_price_estimations):
    # Estimate the year of battery replacement based on the cycle number
    year_of_replacement = current_year + cycle_number * cycles  # Assuming 'years_to_80_percent' is constant
    return calculate_battery_price_per_kwh(year_of_replacement, bat_cap, df_price_estimations)


def update_savings_columns(df, df_price_estimations, current_year, v2g_cost, v1g_cost):
    def calculate_total_saving(row):
        if row['Charging Type'] == 'v2g':
            # Calculate the cumulative savings for V2G
            cycles = int(row["average_smart_years"] / row["years_to_80_percent"])
            total_saving = 0
            for cycle in range(cycles):
                cycle_saving = (row['TOU Cost'] - row['Total_cost']) * row['years_to_80_percent']
                cycle_saving -= calculate_future_battery_price(cycle, current_year, row['bat_cap'], cycles, df_price_estimations)
                total_saving += cycle_saving
            total_saving -= v2g_cost * cycles
            return total_saving
        else:
            # Calculate the savings for Smart Charging
            return (row['TOU Cost'] - row['Total_cost']) * row['years_to_80_percent'] - v1g_cost

    df['Saving_TOU'] = df.apply(lambda row: calculate_total_saving(row), axis=1)

    def calculate_total_saving_ev(row):
        if row['Charging Type'] == 'v2g':
            # Calculate the cumulative savings for V2G
            cycles = int(row["average_smart_years"] / row["years_to_80_percent"])
            total_saving = 0
            for cycle in range(cycles):
                cycle_saving = (row['EV Cost'] - row['Total_cost']) * row['years_to_80_percent']
                cycle_saving -= calculate_future_battery_price(cycle, current_year, row['bat_cap'],cycles, df_price_estimations)
                total_saving += cycle_saving
            total_saving -= v2g_cost * cycles
            return total_saving
        else:
            # Calculate the savings for Smart Charging
            return (row['EV Cost'] - row['Total_cost']) * row['years_to_80_percent'] - v1g_cost

    df['Saving_EV'] = df.apply(lambda row: calculate_total_saving_ev(row), axis=1)

    return df


def plot_saving_ev_vs_distance(df, add_actual_lines=False, add_potential_lines=False, ylim=None):
    plt.figure(figsize=(10, 6))

    # Map charging type to markers
    marker_map = {'v2g': 'o', 'smart': 's'}  # Replace with actual values in your data

    # Define two colors for the charging speeds
    color_map = {0: 'blue', 1: 'green'}  # You can choose any two colors
    label_map = {'v2g': 'V2G', 'smart': 'Smart'}  # Map to desired labels

    # Assign colors based on charging speed
    speed_threshold = 6.6  # Define the threshold to split into two groups
    df['Color'] = df['Charging Speed'].apply(lambda x: color_map[0] if x <= speed_threshold else color_map[1])

    # Define discrete sizes based on battery capacity
    size_map = {66: 150, 70: 200, 75: 250, 80: 300, 85: 350, 100: 400}  # Adjust sizes as necessary

    # Map battery capacities to marker sizes
    df['Size'] = df['bat_cap'].map(size_map)
    # Plot data for each charging type
    for charging_type, marker in marker_map.items():
        subset = df[df['Charging Type'] == charging_type]
        plt.scatter(
            x=subset['distance'],
            y=subset['Saving_EV']/subset['average_smart_years'],
            c=subset['Color'],
            s=subset['Size'],  # Adjust size as needed
            marker=marker,
            alpha=0.7,
            label=label_map[charging_type]
        )
    # Set y-axis limits if provided
    if ylim is not None:
        plt.ylim(0, ylim)
        # Adding vertical lines if the option is active
    if add_actual_lines:
        # plt.axvline(x=5000, color='red', linestyle='--', linewidth=2)
        plt.axvline(x=15000, color='red', linestyle='--', linewidth=2)
        # plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
        # Adding pink shaded area between the red lines and above the horizontal line at y=0

        plt.fill_betweenx(y=[0, 1000], x1=0, x2=15000, color='green', alpha=0.3)
        plt.fill_betweenx(y=[250, 1750], x1=15000, x2=30000, color='pink', alpha=0.3)
        # Adding text above the shaded area
        # Adding text with an arrow above the shaded area
        plt.annotate(
            'Optimal Range for \nSmart Charging',
            xy=(22500, 2000),  # Point to the beginning of the arrow
            xytext=(22500, 2800),  # Text location
            fontsize=16,
            color='red',
            ha='center',
            arrowprops=dict(facecolor='red', shrink=0.05, width=2, headwidth=10, headlength=20, lw=1.5)
        )
    if add_potential_lines:
        plt.fill_betweenx(y=[750, 2800], x1=15000, x2=30000, color='pink', alpha=0.3)
        plt.fill_betweenx(y=[500, 2750], x1=0, x2=15000, color='green', alpha=0.3)
        # plt.axvline(x=15000, color='red', linestyle='--', linewidth=2)
    # Calculate and add a mean line for Saving_EV
    # mean_saving_ev = df['Saving_TOU'].mean()/subset['average_smart_years']
    # plt.axhline(y=mean_saving_ev, color='black', linestyle='-', linewidth=2, label=f'Mean Saving = {mean_saving_ev:.2f}')

    # Adding labels and title
    plt.xlabel('Distance (mile)', fontsize=15)
    plt.ylabel('Savings Compared to the Base Scenario ($/year)', fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    # Format x-ticks with commas for thousands
    plt.xticks(ticks=plt.xticks()[0], labels=[f'{int(x):,}' for x in plt.xticks()[0]])
    plt.yticks(ticks=plt.yticks()[0], labels=[f'{int(x):,}' for x in plt.yticks()[0]])
    plt.xlim(0, 30000)
    # Manually create legend entries for colors (charging speeds)
    speed_legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='6.6 kW', markerfacecolor=color_map[0], markersize=15),
        Line2D([0], [0], marker='o', color='w', label='12 kW', markerfacecolor=color_map[1], markersize=15)
    ]

    # Manually create legend entries for sizes (battery capacities)
    # Manually create legend entries for sizes (battery capacities)
    size_legend_elements = [
        Line2D([0], [0], marker='o', color='w', label=f'{key} kWh', markerfacecolor='gray', markersize=value/15)
        for key, value in size_map.items()
    ]
    # Add the legend for charging type
    type_legend = plt.legend(title='Charging Type', loc='upper right', bbox_to_anchor=(1.33, 1.02), fontsize=15, title_fontsize=17)

    # Add the legend for charging speed
    speed_legend = plt.legend(handles=speed_legend_elements, title='Charging Speed', loc='upper right', bbox_to_anchor=(1.35, 0.78), fontsize=15, title_fontsize=17)

    # Add the legend for battery sizes
    size_legend = plt.legend(handles=size_legend_elements, title='Battery Size', loc='upper right', bbox_to_anchor=(1.33, 0.55), fontsize=15, title_fontsize=17)

    # Adding all legends to the plot
    plt.gca().add_artist(type_legend)
    plt.gca().add_artist(speed_legend)
    plt.gca().add_artist(size_legend)
    plt.grid()

    # Adjust the layout to ensure everything fits within the figure
    plt.tight_layout(rect=[0, 0, 0.81, 1])  # Adjust rect to make room for legends

    # Show the plot
    plt.show()