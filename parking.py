# %%
import pytz
import pandas as pd
import re
import math
from datetime import timedelta
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import os
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.ticker as ticker
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D
##################################################################################################################
##################################################################################################################
# %%
# Set Helvetica or Arial as the default font
mpl.rcParams['font.sans-serif'] = "Helvetica"
mpl.rcParams['font.family'] = "sans-serif"
# If Helvetica is not available, fallback to Arial
mpl.rcParams['font.sans-serif'] = ["Helvetica", "Arial"]


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


def clean_data(input_data):
    vehicle_names = input_data
    df = pd.DataFrame()
    for vehicle_name in vehicle_names:
        df_full_trips = read_clean(vehicle_name)  # done
        df_full_trips_short = trip_summary(df_full_trips)  # done
        df_soc_req = soc_next(df_full_trips_short)  # done
        # Failed Next Trip
        df_soc_req["f_next_trip"] = df_soc_req["battery[soc][start][trip]"].shift(-1) - df_soc_req["SOC_next_trip"]
        # Failed Next Charging
        df_soc_req["f_next_charge"] = df_soc_req["battery[soc][start][trip]"].shift(-1) - df_soc_req["SOC_need_next_charge"]
        # Update end_time_charging based on start_time_charging and duration_charging_min
        df_soc_req["end_time_charging"] = pd.to_datetime(df_soc_req["start_time_charging"]) + pd.to_timedelta(df_soc_req["duration_charging_min"], unit='m')

        # Ensure the date format remains as requested
        df_soc_req["start_time_charging"] = pd.to_datetime(df_soc_req["start_time_charging"]).dt.strftime('%Y-%m-%d %H:%M:%S%z')
        df_soc_req["end_time_charging"] = df_soc_req["end_time_charging"].dt.strftime('%Y-%m-%d %H:%M:%S%z')
        df = pd.concat([df, df_soc_req], axis=0, ignore_index=True)
    return df


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


def draw_multilevel_pie(df, text_size=13):
    df1 = df.copy()
    df1["duration_charging_min"] = df1["duration_charging_min"].fillna(0)

    # Step 1: Prepare data for the inner circle (Destinations) with colors based on average parking time
    destination_counts = df1['destination_label'].value_counts()
    total_instances = destination_counts.sum()
    inner_ratios = destination_counts / total_instances  # Proportion of each destination
    destinations = destination_counts.index.tolist()

    # Calculate average parking time per destination for color mapping
    avg_parking_times = [df1[df1['destination_label'] == dest]['parking_time_minute'].mean() for dest in destinations]
    norm_inner = plt.Normalize(min(avg_parking_times), max(avg_parking_times))
    inner_palette = sns.color_palette("crest", as_cmap=True)
    inner_colors = inner_palette(norm_inner(avg_parking_times))

    # Step 2: Prepare data for the outer circle based on Plugged-in and Parking with colors based on charging duration
    outer_ratios = []
    outer_labels = []
    avg_charging_times = []

    for i, destination in enumerate(destinations):
        df_dest = df1[df1['destination_label'] == destination]
        total_count = len(df_dest)

        # Calculate counts for Plugged-in and Parking
        plugged_in_count = len(df_dest[df_dest['duration_charging_min'] > 0])
        parking_count = total_count - plugged_in_count

        # Calculate proportions for Plugged-in and Parking within the destination
        inner_ratio = inner_ratios.iloc[i]
        outer_ratios.extend([plugged_in_count / total_count * inner_ratio,
                             parking_count / total_count * inner_ratio])

        # Labels for outer segments
        outer_labels.extend([f"{destination} - Parked - Plugged-in", f"{destination} - Parked - not Plugged-in"])

        # Average charging duration time for Plugged-in; 0 for Parking
        avg_charging_time = df_dest[df_dest['duration_charging_min'] > 0]['duration_charging_min'].mean()
        avg_charging_times.extend([avg_charging_time if plugged_in_count > 0 else 0, 0])

    # Normalize colors for outer circle based on charging duration
    norm_outer = plt.Normalize(min(avg_charging_times), max(avg_charging_times))
    outer_palette = sns.color_palette("flare", as_cmap=True)
    outer_colors = outer_palette(norm_outer(avg_charging_times))

    # Function to calculate luminance
    def get_text_color(color):
        r, g, b = color[:3]
        luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b  # Calculate luminance
        return 'black' if luminance > 0.5 else 'white'

    # Step 3: Plot the pie chart with nested proportions
    fig, ax = plt.subplots(figsize=(10, 6))

    # Draw the inner circle (Destinations) with colors based on average parking time
    wedges_inner, texts_inner, autotexts_inner = ax.pie(inner_ratios, radius=1.1, labels=None, colors=inner_colors,
                                                        wedgeprops=dict(width=0.6, edgecolor='w'), autopct='%1.1f%%',
                                                        pctdistance=0.6, textprops={'fontsize': text_size})
    for i, wedge in enumerate(wedges_inner):
        angle = (wedge.theta2 - wedge.theta1) / 2. + wedge.theta1
        x = 0.3 * np.cos(np.radians(angle))
        y = 0.3 * np.sin(np.radians(angle))
        ax.text(x, y, destinations[i], ha='center', va='center', fontsize=text_size)

    # Set percentage text to white for inner circle
    for autotext, color in zip(autotexts_inner, inner_colors):
        autotext.set_color(get_text_color(color))
        autotext.set_fontsize(text_size + 2)

    # Draw the outer circle (Plugged-in and Parking) with colors based on charging duration
    wedges_outer, texts_outer, autotexts_outer = ax.pie(outer_ratios, radius=1.5, labels=outer_labels, colors=outer_colors,
                                                        wedgeprops=dict(width=0.6, edgecolor='w'), autopct='%1.1f%%',
                                                        pctdistance=0.8, textprops={'fontsize': text_size + 2})

    # Set percentage text to white for outer circle
    for autotext, color in zip(autotexts_outer, outer_colors):
        autotext.set_color(get_text_color(color))
        autotext.set_fontsize(text_size + 2)

    # Add color bars for both inner and outer circles
    sm_inner = plt.cm.ScalarMappable(cmap=inner_palette, norm=norm_inner)
    sm_outer = plt.cm.ScalarMappable(cmap=outer_palette, norm=norm_outer)

    cbar_inner = plt.colorbar(sm_inner, ax=ax, fraction=0.05, pad=0.1, orientation='horizontal')
    cbar_inner.set_label('Average Parking Time in Minute (Inner Circle)', fontsize=text_size)

    cbar_outer = plt.colorbar(sm_outer, ax=ax, fraction=0.05, pad=0.15, orientation='horizontal')
    cbar_outer.set_label('Average Charging Duration in Minute (Outer Circle)', fontsize=text_size)

    ax.set(aspect="equal")
    plt.savefig('multilevel_pie_chart.png', bbox_inches='tight', dpi=600)
    plt.show()


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


def charging_speed(df):
    df["charging_speed"] = ((((df["battery[soc][end][charging]"] - df["battery[soc][start][charging]"]) / 100) * df["bat_cap"]) / (df["duration_charging_min"] / 60))
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
    # level 2 12
    df["V2G_SOC_half_12k"] = (((df["battery[soc][end][charging]"] / 100) * df["bat_cap"]) - ((df["V2G_time_min"] / 60) / 2) * 12) / (df["bat_cap"]) * 100
    df.loc[df["V2G_SOC_half_12k"] < 0, "V2G_SOC_half_12k"] = 0
    df["V2G_cap_12k"] = (abs(df["V2G_SOC_half_12k"] - df["battery[soc][end][charging]"]) / 100) * df["bat_cap"]
    # with Level 2
    # Assuming df is your DataFrame
    df["V2G_SOC_half_6k"] = (((df["battery[soc][end][charging]"] / 100) * df["bat_cap"]) - ((df["V2G_time_min"] / 60) / 2) * 6.6) / (df["bat_cap"]) * 100
    df.loc[df["V2G_SOC_half_6k"] < 0, "V2G_SOC_half_6k"] = 0
    df["V2G_cap_6k"] = (abs(df["V2G_SOC_half_6k"] - df["battery[soc][end][charging]"]) / 100) * df["bat_cap"]
    # Assuming df is your DataFrame
    df["V2G_SOC_half_19k"] = (((df["battery[soc][end][charging]"] / 100) * df["bat_cap"]) - ((df["V2G_time_min"] / 60) / 2) * 19) / (df["bat_cap"]) * 100
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
    df1["V2G_SOC_half_12k"] = (((df1["battery[soc][end][charging]"] / 100) * df1["bat_cap"]) - ((df1["V2G_time_min"] / 60) / 2) * 12) / (df1["bat_cap"]) * 100
    df1.loc[df1["V2G_SOC_half_12k"] < 0, "V2G_SOC_half_12k"] = 0
    df1["V2G_cap_12k"] = (abs(df1["V2G_SOC_half_12k"] - (df1["battery[soc][end][charging]"] + 5)) / 100) * df1["bat_cap"]
    # Assuming df1 is your DataFrame
    df1["V2G_SOC_half_6k"] = (((df1["battery[soc][end][charging]"] / 100) * df1["bat_cap"]) - ((df1["V2G_time_min"] / 60) / 2) * 6.6) / (df1["bat_cap"]) * 100
    df1.loc[df1["V2G_SOC_half_6k"] < 0, "V2G_SOC_half_6k"] = 0
    df1["V2G_cap_6k"] = (abs(df1["V2G_SOC_half_6k"] - (df1["battery[soc][end][charging]"] + 5)) / 100) * df1["bat_cap"]
    # Assuming df1 is your DataFrame
    df1["V2G_SOC_half_19k"] = (((df1["battery[soc][end][charging]"] / 100) * df1["bat_cap"]) - ((df1["V2G_time_min"] / 60) / 2) * 19) / (df1["bat_cap"]) * 100
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


def V2G_cap_ch_r_mc(df):
    df = df[df["charging_speed"] != 0].fillna(0)
    df['end_time_charging'] = pd.to_datetime(df['end_time_charging'])

    # current speed
    df["V2G_SOC_half_12k"] = (((df["battery[soc][end][charging]"] / 100) * df["bat_cap"]) - ((df["V2G_time_min"] / 60) / 2) * 12) / (df["bat_cap"]) * 100
    df.loc[df["V2G_SOC_half_12k"] < 0, "V2G_SOC_half_12k"] = 0
    df["V2G_cap_12k"] = (abs(df["V2G_SOC_half_12k"] - df["battery[soc][end][charging]"]) / 100) * df["bat_cap"]
    df["V2G_cycle_12k_time"] = (((df["battery[soc][end][charging]"] / 100) * df["bat_cap"]) / 12) * 2
    df.replace({float('inf'): 0, float('-inf'): 0, pd.NaT: 0}, inplace=True)
    df.loc[df["V2G_cycle_12k_time"] < 0, "V2G_cycle_12k_time"] = 0
    df["V2G_max_cycle_12k"] = df.apply(lambda row: math.floor((row["V2G_time_min"] / 60) / row["V2G_cycle_12k_time"]) if row["V2G_cycle_12k_time"] != 0 else 0, axis=1)
    df.loc[df["V2G_max_cycle_12k"] < 0, "V2G_max_cycle_12k"] = 0
    df.loc[df["V2G_max_cycle_12k"] != 0, "V2G_cap_12k"] *= df["V2G_max_cycle_12k"]

    # Assuming df is your DataFrame
    df["V2G_SOC_half_6k"] = (((df["battery[soc][end][charging]"] / 100) * df["bat_cap"]) - ((df["V2G_time_min"] / 60) / 2) * 6.6) / (df["bat_cap"]) * 100
    df.loc[df["V2G_SOC_half_6k"] < 0, "V2G_SOC_half_6k"] = 0
    df["V2G_cap_6k"] = (abs(df["V2G_SOC_half_6k"] - df["battery[soc][end][charging]"]) / 100) * df["bat_cap"]
    df["V2G_cycle_6k_time"] = (((df["battery[soc][end][charging]"] / 100) * df["bat_cap"]) / (6.6)) * 2
    df.replace({float('inf'): 0, float('-inf'): 0, pd.NaT: 0}, inplace=True)
    df["V2G_max_cycle_6k"] = df.apply(lambda row: math.floor((row["V2G_time_min"] / 60) / row["V2G_cycle_6k_time"]) if row["V2G_cycle_6k_time"] != 0 else 0, axis=1)
    df.loc[df["V2G_max_cycle_6k"] < 0, "V2G_max_cycle_6k"] = 0
    df.loc[df["V2G_max_cycle_6k"] != 0, "V2G_cap_6k"] *= df["V2G_max_cycle_6k"]

    # Assuming df is your DataFrame
    df["V2G_SOC_half_19k"] = (((df["battery[soc][end][charging]"] / 100) * df["bat_cap"]) - ((df["V2G_time_min"] / 60) / 2) * 19) / (df["bat_cap"]) * 100
    df.loc[df["V2G_SOC_half_19k"] < 0, "V2G_SOC_half_19k"] = 0
    df["V2G_cap_19k"] = (abs(df["V2G_SOC_half_19k"] - df["battery[soc][end][charging]"]) / 100) * df["bat_cap"]
    df["V2G_cycle_19k_time"] = (((df["battery[soc][end][charging]"] / 100) * df["bat_cap"]) / 19) * 2
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
    df1["V2G_SOC_half_12k"] = (((df1["battery[soc][end][charging]"] / 100) * df1["bat_cap"]) - ((df1["V2G_time_min"] / 60) / 2) * 12) / (df1["bat_cap"]) * 100
    df1.loc[df1["V2G_SOC_half_12k"] < 0, "V2G_SOC_half_12k"] = 0
    df1["V2G_cap_12k"] = (abs(df1["V2G_SOC_half_12k"] - (df1["battery[soc][end][charging]"] + 5)) / 100) * df1["bat_cap"]
    df1["V2G_cycle_12k_time"] = (((df1["battery[soc][end][charging]"] / 100) * df1["bat_cap"]) / 12) + ((((df1["battery[soc][end][charging]"] - 5) / 100) * df1["bat_cap"]) / 12)
    df1.replace({float('inf'): 0, float('-inf'): 0, pd.NaT: 0}, inplace=True)
    df1.loc[df1["V2G_cycle_12k_time"] < 0, "V2G_cycle_12k_time"] = 0
    df1["V2G_max_cycle_12k"] = df1.apply(lambda row: math.floor((row["V2G_time_min"] / 60) / row["V2G_cycle_12k_time"]) if row["V2G_cycle_12k_time"] != 0 else 0, axis=1)
    df1.loc[df1["V2G_max_cycle_12k"] < 0, "V2G_max_cycle_12k"] = 0
    df1.loc[df1["V2G_max_cycle_12k"] != 0, "V2G_cap_12k"] *= df1["V2G_max_cycle_12k"]

    # Assuming df1 is your DataFrame
    df1["V2G_SOC_half_6k"] = (((df1["battery[soc][end][charging]"] / 100) * df1["bat_cap"]) - ((df1["V2G_time_min"] / 60) / 2) * 6.6) / (df1["bat_cap"]) * 100
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


def total_storage(df1, df2, df3, df1_r5, df2_r5, df3_r5, df1_r10, df2_r10, df3_r10):
    data = {'6.6 kW': [df1_r5.sum().sum() / 1000, df2_r5.sum().sum() / 1000, df3_r5.sum().sum() / 1000],
            '12 kw': [df1.sum().sum() / 1000, df2.sum().sum() / 1000, df3.sum().sum() / 1000],
            '19 kW': [df1_r10.sum().sum() / 1000, df2_r10.sum().sum() / 1000, df3_r10.sum().sum() / 1000]}

    df_summary_storage = pd.DataFrame(data, index=['Total', 'Total_s5', 'Total_s10']).T

    return df_summary_storage


def total_storage_tou(df1, df2, df3):
    data = {'6.6 kW': [df1.sum().sum() / 1000],
            '12 kw': [df2.sum().sum() / 1000],
            '19 kW': [df3.sum().sum() / 1000]}

    df_summary_storage = pd.DataFrame(data, index=['Total']).T

    return df_summary_storage


def failure_estimation(df1, df2):
    ratio5_nt = df1["next_trip_fail"].value_counts(normalize=True)
    ratio5_nc = df1["next_c_fail"].value_counts(normalize=True)

    ratio10_nt = df2["next_trip_fail"].value_counts(normalize=True)
    ratio10_nc = df2["next_c_fail"].value_counts(normalize=True)

    data = {'ratio5': [ratio5_nt[1] * 100, ratio5_nc[1] * 100],
            'ratio10': [ratio10_nt[1] * 100, ratio10_nc[1] * 100]}
    data = pd.DataFrame(data, index=['next_trip', 'next_charging']).T

    return data


def total_capacity(df):
    total_cap_df = df.groupby('vehicle_name', as_index=False).first()[['vehicle_name', 'bat_cap']]
    total_cap = total_cap_df["bat_cap"].sum()
    return total_cap


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

    return discharge_start, discharge_end, charge_start, charge_end, row['next_departure_time']


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


def storage_cap_tou_sta(df):
    df = df.copy()
    df = df.copy()
    V2G_hourly_tou = pd.DataFrame(index=df.index, columns=range(24))
    V2G_hourly_12_tou = V2G_hourly_tou.fillna(0)
    V2G_hourly_6_tou = V2G_hourly_12_tou.copy()
    V2G_hourly_19_tou = V2G_hourly_12_tou.copy()
    for i in df.index:
        start_hour = df.loc[i, "discharge_start"].hour - 1
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


def extra_extra_kwh(df):
    # Divide values by 1000 to convert kWh to MWh
    # Sort the DataFrame based on the sum of each row
    test0_MWh_sorted = df.sum(axis=1).sort_values().index
    test0_MWh_sorted_df = df.loc[test0_MWh_sorted]
    test0_MWh_sorted_df = test0_MWh_sorted_df / 1000
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
        plt.bar(test0_MWh_sorted_df.index, (test0_MWh_sorted_df[column] / 365), color=colors[i], label=column)

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


def v2g_participate(df):
    # df = v2g_tou.copy()
    df["discharge_end1"] = pd.to_datetime(df["discharge_end"])
    df["discharge_start1"] = pd.to_datetime(df["discharge_start"])
    df["charge_end1"] = pd.to_datetime(df["charge_end"])
    df["charge_start1"] = pd.to_datetime(df["charge_start"])

    df["SOC_after_char_V2G_6k"] = (df["V2G_SOC_tou_6k"] + (((((df["charge_end1"] - df["charge_start1"]).dt.seconds / 3600) * 6.6) / df["bat_cap"]) * 100))
    df["SOC_after_char_V2G_12k"] = (df["V2G_SOC_tou_12k"] + (((((df["charge_end1"] - df["charge_start1"]).dt.seconds / 3600) * 12) / df["bat_cap"]) * 100))
    df["SOC_after_char_V2G_19k"] = (df["V2G_SOC_tou_19k"] + (((((df["charge_end1"] - df["charge_start1"]).dt.seconds / 3600) * 19) / df["bat_cap"]) * 100))

    df["V2G_participate"] = False
    df.loc[(df["discharge_end1"] - df["discharge_start1"]).dt.seconds > 0, "V2G_participate"] = True

    return df


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

    v2g_tou_p.loc[v2g_tou_p["energy[charge_type][type]"] == "Parking", "V2G_cap_6k_tou"] = (((v2g_tou_p.loc[v2g_tou_p["energy[charge_type][type]"] == "Parking", "discharge_end"] - v2g_tou_p.loc[v2g_tou_p["energy[charge_type][type]"] == "Parking", "discharge_start"]).dt.seconds) / 3600) * 6.6
    v2g_tou_p.loc[v2g_tou_p["energy[charge_type][type]"] == "Parking", "V2G_cap_12k_tou"] = (((v2g_tou_p.loc[v2g_tou_p["energy[charge_type][type]"] == "Parking", "discharge_end"] - v2g_tou_p.loc[v2g_tou_p["energy[charge_type][type]"] == "Parking", "discharge_start"]).dt.seconds) / 3600) * 12
    v2g_tou_p.loc[v2g_tou_p["energy[charge_type][type]"] == "Parking", "V2G_cap_19k_tou"] = (((v2g_tou_p.loc[v2g_tou_p["energy[charge_type][type]"] == "Parking", "discharge_end"] - v2g_tou_p.loc[v2g_tou_p["energy[charge_type][type]"] == "Parking", "discharge_start"]).dt.seconds) / 3600) * 19

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


# def xlsx_read(dic):
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
#     return total_costs_df, individual_cost_df


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
    result["Utilization Rate"] = abs(result["X_CHR_Sum"] / result["Total_power"]) * 100

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


def plot_cost_comparison_EV(df, num_vehicles, title_size=14, axis_text_size=12, y_axis_title='Cost ($)', barhight=500):
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


def plot_cost_comparison_RT(df, num_vehicles, title_size=14, axis_text_size=12, y_axis_title='Cost ($)', barhight=500):
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


def draw_RT(df):
    # Create a sample array representing electricity price for 8760 hours
    electricity_price = []  # Random values between 0 and 1
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
    plt.ylabel('Electricity Price ($ / MWh)', fontsize=18)
    # plt.title('Electricity Price for year 2021 - PG&E territory ')

    # Add grid and legend
    plt.grid(True)
    plt.legend(fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlim(0, 8760)
    # Show the plot
    plt.tight_layout()
    plt.show()


def draw_RT_scatter(price_dict, horizon):
    """
    Plots electricity prices for multiple territories as scatter points with transparency.

    :param price_dict: A dictionary where keys are legend names and values are dictionaries of hourly electricity prices.
    """
    # Define distinct marker styles for each region
    markers = ['o', 's', 'D', '^']  # Circle, square, diamond, triangle
    colors = ['blue', 'green', 'red', 'orange']  # Colors for each region

    # Initialize the plot
    plt.figure(figsize=(12, 8))

    # Loop through the dictionary and process each price set
    for i, (name, prices_dict) in enumerate(price_dict.items()):
        # Convert the dictionary values (hourly prices) into a sorted list of the first 8760 hours
        prices = [prices_dict[hour] for hour in sorted(prices_dict.keys())[:horizon]]

        # Plot the prices as scatter points with transparency
        hours = np.arange(0, horizon, 1)  # Array of hours from 0 to 8759
        plt.scatter(
            hours, prices,
            label=name,
            alpha=0.8,  # Transparency
            marker=markers[i % len(markers)],
            color=colors[i % len(colors)],
            s=100  # Size of scatter points
        )

    # Set labels and title
    plt.xlabel('Hour', fontsize=18)
    plt.ylabel('Electricity Price ($ / MWh)', fontsize=18)
    plt.title('Electricity Prices for Various Territories', fontsize=20)

    # Add grid and legend
    plt.grid(True)
    plt.legend(fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlim(0, horizon)

    # Show the plot
    plt.tight_layout()
    plt.show()


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
    ax2.plot(optimal_result['Datetime'], optimal_result['Price'] / 1000, label='Electricity Price', color='red', linestyle='dashed', marker='x')
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


def violin_input(df):
    df["Charging Speed"] = df["Charging Speed"].astype(float)
    df["GHG Cost"] = df["GHG Cost"].astype(float)
    df['Total_Cost'] = df['Electricity_Cost'] + df['Degradation_Cost'] + df['GHG_Cost']
    df.loc[df['Charging Type'] == "smart", "Charging Type"] = "Smart"
    df.loc[df['Charging Type'] == "v2g", "Charging Type"] = "V2G"
    df.loc[(df['Charging Type'] == "V2G") | (df['Charging Type'] == "Smart"), "Scenario"] = df['Charging Type'] + '-' + df['Plugged-in Sessions'] + '-' + df['Tariff']
    df.loc[(df['Charging Type'] == "Actual - TOU") | (df['Charging Type'] == "Actual - EV Rate"), "Scenario"] = "Actual" + '-' + df['Plugged-in Sessions'] + '-' + df['Tariff']

    # Calculate Total Cost
    df['Social Cost of Carbon'] = df['GHG Cost']
    df.loc[df['Social Cost of Carbon'] == 0, "Social Cost of Carbon"] = 0.05
    df_long = df[["Vehicle", "Scenario", "Charging Speed", "Total_Cost", "Social Cost of Carbon", "Tariff"]].sort_values(by='Scenario')
    # Create a new column for the combined hue
    df_long['Tariff_Social_Cost'] = df_long['Tariff'] + '-' + df_long['Social Cost of Carbon'].astype(str)
    df_long = df_long.sort_values(["Vehicle", "Scenario", "Charging Speed", "Social Cost of Carbon", "Tariff"])
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


def dr_plot(data1, data2, data3, data4, data5, data6):
    dfs = [data1, data2, data3, data4, data5, data6]
    fig, axes = plt.subplots(2, 6, figsize=(20, 12), sharey=True)
    plt.subplots_adjust(hspace=0.4, wspace=0.3)  # Adjust space between rows and columns

    bar_width = 1  # Set bar width to reduce distance between bars

    # Define the colors
    blue_color = sns.color_palette("Blues", 10)[5]  # Ensure we get the 6th shade
    red_color = sns.color_palette("Reds", 10)[6]  # Ensure we get the 7th shade

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
            axes[row, col].set_yticks([i / 10 for i in range(0, 11, 2)])  # Show every other hour for readability
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


def load_bev_distance(file_path):
    bev_data = pd.read_csv(file_path)
    bev_distance = bev_data.groupby(["vehicle_name", "bat_cap"])["distance"].sum().reset_index()
    return bev_distance


def load_and_prepare_cost_data(file_path, sheet_name, rate_type):
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    # df = df.drop(df.columns[0], axis=1)  # Drop the unnamed column
    df["Rate_Type"] = rate_type
    df["V2G_Location"] = "None"
    df["Scenario"] = "Conventional"
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
    costs_ev_rate = load_and_prepare_cost_data(ev_cost_file, 'Individual Costs', "EV_Rate")
    costs_tou_rate = load_and_prepare_cost_data(tou_cost_file, 'Individual Costs', "TOU")
    # Adjust GHG cost for the new value of 0.191
    # costs_ev_rate_191 = adjust_ghg_cost(costs_ev_rate, new_ghg_cost=0.191)
    # costs_tou_rate_191 = adjust_ghg_cost(costs_tou_rate, new_ghg_cost=0.191)
    # Combine all cost data into a single DataFrame
    combined_costs = pd.concat([costs_ev_rate, costs_tou_rate]).reset_index(drop=True)
    # Calculate total costs
    combined_costs["Total_cost"] = combined_costs["Electricity_Cost"] + combined_costs["Degradation_Cost"]
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
    # Set any negative X_CHR values to zero
    data["X_CHR"] = data["X_CHR"].apply(lambda x: max(x, 0))

    # Group and sum the data
    grouped_data = data.groupby(
        ["Vehicle", "Charge_Type", "Charging_Speed", "Rate_Type", 'Utility', 'V2G_Location', "Scenario"]
    )[["X_CHR", "Electricity_Cost", "Degradation_Cost", "GHG_Cost"]].sum().reset_index()

    return grouped_data


def calculate_years_to_80_percent(bt_cap_rem, annual_consumption):
    years = 0
    while bt_cap_rem > 0.8:  # Stop when battery capacity drops to 80% or less
        bt_cap_rem -= 1.42e-6 * annual_consumption  # Subtract annual consumption
        years += 1
    return years


def process_hourly_charging_data(data_N, data_P):
    # Concatenate N and P datasets
    data = pd.concat([data_N, data_P])
    data['Charging_Speed'] = data['Charging_Speed'].str.replace('kW', '').astype(float)

    # Filter rows based on conditions
    # data = data[data["GHG Cost"] > 0]
    data = data[~((data["Charging_Speed"] == 19) & (data["Charge_Type"] == "Bidirectional"))]
    data = data[data["V2G_Location"] == "Home_Work"]
    # Calculate battery remaining capacity
    data["bt_cap_rem"] = -1.42e-6 * data["X_CHR"] + 0.989999999
    # Calculate years to 80% capacity
    data["years_to_80_percent"] = data.apply(lambda row: calculate_years_to_80_percent(row["bt_cap_rem"], row["X_CHR"]), axis=1)
    data.loc[data["years_to_80_percent"] > 15, "years_to_80_percent"] = 15
    return data


def add_smart_avg(data):
    # Calculate the average years for 'smart' charging type
    smart_avg = data[data["Charge_Type"] == "Smart"].groupby("Vehicle")["years_to_80_percent"].mean().reset_index()
    smart_avg = smart_avg.rename(columns={"years_to_80_percent": "average_smart_years"})
    # Merge average smart years back to the original data
    data = pd.merge(data, smart_avg, on="Vehicle", how="left")
    # Calculate percentage drop
    data["percentage_drop"] = ((data["average_smart_years"] - data["years_to_80_percent"]) / data["average_smart_years"]) * 100

    return data


def merge_and_calculate_costs(data, actual_cost, bev_distance):
    # Merge BEV distance
    data = pd.merge(data, bev_distance, left_on="Vehicle", right_on="vehicle_name", how="left")
    # Remove '- Home' from 'Tariff' column
    # data["Tariff"] = data["Tariff"].str.replace('- Home', '', regex=False)
    # Merge TOU cost
    actual_cost_tou = actual_cost[actual_cost["Rate_Type"] == "TOU"]
    data = pd.merge(data, actual_cost_tou[["Vehicle", "Total_cost"]], on="Vehicle", how="left", suffixes=('', '_TOU'))
    data["TOU Cost"] = data["Total_cost"]  # Assign TOU cost
    # Merge EV cost
    actual_cost_ev = actual_cost[actual_cost["Rate_Type"] == "EV_Rate"]
    data = pd.merge(data, actual_cost_ev[["Vehicle", "Total_cost"]], on="Vehicle", how="left", suffixes=('', '_EV'))
    data["EV Cost"] = data["Total_cost_EV"]  # Assign EV cost
    # Drop unnecessary columns
    data.drop(["Total_cost", "Total_cost_EV"], axis=1, inplace=True, errors='ignore')
    # Calculate the final total cost
    data["Total_cost"] = data["Electricity_Cost"] + data["Degradation_Cost"]

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


def calculate_future_battery_price(cycle_number, current_year, bat_cap, cycles, df_price_estimations):
    # Estimate the year of battery replacement based on the cycle number
    year_of_replacement = current_year + cycle_number * cycles  # Assuming 'years_to_80_percent' is constant
    return calculate_battery_price_per_kwh(year_of_replacement, bat_cap, df_price_estimations)


def update_savings_columns(df, df_price_estimations, current_year, v2g_cost, v1g_cost, v1g_cost_19kw):
    def calculate_total_saving(row):
        if row['Charge_Type'] == 'Bidirectional':
            # Calculate the cumulative savings for V2G
            cycles = int(row["average_smart_years"] / row["years_to_80_percent"])
            total_saving = 0
            for cycle in range(cycles):
                cycle_saving = (row['TOU Cost'] - row['Total_cost']) * row['years_to_80_percent']
                cycle_saving -= calculate_future_battery_price(cycle, current_year, row['bat_cap'], cycles, df_price_estimations)
                total_saving += cycle_saving
            total_saving -= v2g_cost
            return total_saving
        else:
            # Adjust v1g_cost for smart charging with 19 kW chargers
            if row['Charging_Speed'] == 19:
                adjusted_v1g_cost = v1g_cost_19kw
            else:
                adjusted_v1g_cost = v1g_cost
            # Calculate the savings for Smart Charging
            return (row['TOU Cost'] - row['Total_cost']) * row['years_to_80_percent'] - adjusted_v1g_cost

    df['Saving_TOU'] = df.apply(lambda row: calculate_total_saving(row), axis=1)

    def calculate_total_saving_ev(row):
        if row['Charge_Type'] == 'Bidirectional':
            # Calculate the cumulative savings for V2G
            cycles = int(row["average_smart_years"] / row["years_to_80_percent"])
            total_saving = 0
            for cycle in range(cycles):
                cycle_saving = (row['EV Cost'] - row['Total_cost']) * row['years_to_80_percent']
                cycle_saving -= calculate_future_battery_price(cycle, current_year, row['bat_cap'], cycles, df_price_estimations)
                total_saving += cycle_saving
            total_saving -= v2g_cost
            return total_saving
        else:
            # Adjust v1g_cost for smart charging with 19 kW chargers
            if row['Charging_Speed'] == 19:
                adjusted_v1g_cost = v1g_cost_19kw
            else:
                adjusted_v1g_cost = v1g_cost
            # Calculate the savings for Smart Charging
            return (row['EV Cost'] - row['Total_cost']) * row['years_to_80_percent'] - adjusted_v1g_cost

    df['Saving_EV'] = df.apply(lambda row: calculate_total_saving_ev(row), axis=1)

    return df


def update_savings_columns1(df, df_price_estimations, current_year, v2g_cost, v1g_cost, v1g_cost_19kw, interest_rate=0.05):
    def calculate_total_saving(row):
        total_saving = 0
        # Annual savings
        yearly_saving = (row['TOU Cost'] - row['Total_cost'])
        # Total years in the study
        total_years = int(row["average_smart_years"])
        # Number of cycles during the study period
        cycles = int(total_years / row["years_to_80_percent"])

        # Add yearly savings with discounting
        for year in range(1, total_years + 1):
            total_saving += yearly_saving / ((1 + interest_rate) ** year)

        # Add battery replacement costs with discounting
        for cycle in range(1, cycles):
            replacement_year = cycle * row["years_to_80_percent"]
            battery_cost = calculate_future_battery_price(cycle, current_year, row['bat_cap'], cycles, df_price_estimations)
            total_saving -= battery_cost / ((1 + interest_rate) ** replacement_year)

        # Subtract V2G infrastructure cost (applied once, year 0)
        if row['Charge_Type'] == 'Bidirectional':
            total_saving -= v2g_cost
        else:
            # Adjust V1G cost for smart charging with 19 kW chargers
            adjusted_v1g_cost = v1g_cost_19kw if row['Charging_Speed'] == 19 else v1g_cost
            total_saving -= adjusted_v1g_cost

        return total_saving

    def calculate_total_saving_ev(row):
        total_saving = 0
        # Annual savings
        yearly_saving = (row['EV Cost'] - row['Total_cost'])
        # Total years in the study
        total_years = int(row["average_smart_years"])
        # Number of cycles during the study period
        cycles = int(total_years / row["years_to_80_percent"])

        # Add yearly savings with discounting
        for year in range(1, total_years + 1):
            total_saving += yearly_saving / ((1 + interest_rate) ** year)

        # Add battery replacement costs with discounting
        for cycle in range(1, cycles):
            replacement_year = cycle * row["years_to_80_percent"]
            battery_cost = calculate_future_battery_price(cycle, current_year, row['bat_cap'], cycles, df_price_estimations)
            total_saving -= battery_cost / ((1 + interest_rate) ** replacement_year)

        # Subtract V2G infrastructure cost (applied once, year 0)
        if row['Charge_Type'] == 'Bidirectional':
            total_saving -= v2g_cost
        else:
            # Adjust V1G cost for smart charging with 19 kW chargers
            adjusted_v1g_cost = v1g_cost_19kw if row['Charging_Speed'] == 19 else v1g_cost
            total_saving -= adjusted_v1g_cost

        return total_saving

    df['Saving_TOU'] = df.apply(lambda row: calculate_total_saving(row), axis=1)
    df['Saving_EV'] = df.apply(lambda row: calculate_total_saving_ev(row), axis=1)
    return df


# def plot_saving_ev_vs_distance(df, add_actual_lines=False, add_potential_lines=False, ylim=10000, text_size=18, title='title'):
#     plt.figure(figsize=(8, 6))
#
#     # Map charging type to markers
#     marker_map = {'Bidirectional': 'o', 'Smart': 's'}  # Replace with actual values in your data
#     label_map = {'Bidirectional': 'V2G', 'Smart': 'V1G'}  # Map to desired labels
#
#     # Define colors for each charging speed
#     color_map = {6.6: 'blue', 12: 'green', 19: 'red'}  # Assign colors for each charging speed
#
#     # Assign colors based on exact charging speeds
#     df['Color'] = df['Charging_Speed'].map(color_map)
#
#     # Define discrete sizes based on battery capacity
#     size_map = {66: 150, 70: 200, 75: 250, 80: 300, 85: 350, 100: 400}  # Adjust sizes as necessary
#
#     # Map battery capacities to marker sizes
#     df['Size'] = df['bat_cap'].map(size_map)
#
#     # Plot data for each charging type without labels (we'll create custom legends)
#     for charging_type, marker in marker_map.items():
#         subset = df[df['Charge_Type'] == charging_type]
#         plt.scatter(
#             x=subset['distance'],
#             y=subset['Saving_TOU'] / subset['average_smart_years'],
#             c=subset['Color'],
#             s=subset['Size'],
#             marker=marker,
#             alpha=0.7,
#             # Remove the label to prevent automatic legend entries
#             # label=label_map[charging_type]
#         )
#
#     # Set y-axis limits if provided
#     if ylim is not None:
#         plt.ylim(0, ylim)
#
#     # Adding vertical lines if the option is active
#     if add_actual_lines:
#         plt.axvline(x=15000, color='red', linestyle='--', linewidth=2)
#         # plt.fill_betweenx(y=[0, 1000], x1=0, x2=15000, color='green', alpha=0.3)
#         plt.fill_betweenx(y=[500, 2500], x1=15000, x2=30000, color='pink', alpha=0.3)
#         plt.annotate(
#             'Optimal Range for \nSmart Charging',
#             xy=(22500, 2500),
#             xytext=(22500, 3200),
#             fontsize=16,
#             color='red',
#             ha='center',
#             arrowprops=dict(facecolor='red', shrink=0.05, width=2, headwidth=10, headlength=20, lw=1.5)
#         )
#
#     if add_potential_lines:
#         plt.fill_betweenx(y=[750, 2800], x1=15000, x2=30000, color='pink', alpha=0.3)
#         plt.fill_betweenx(y=[500, 2750], x1=0, x2=15000, color='green', alpha=0.3)
#
#     # Adding labels and title
#     plt.xlabel('Distance (mile)', fontsize=text_size)
#     plt.ylabel(f'Savings Compared to the Base Scenario ($/year)\n{title}', fontsize=text_size - 1)
#     plt.xticks(fontsize=text_size - 4, rotation=45)
#     plt.yticks(fontsize=text_size)
#
#     # Format x-ticks and y-ticks with commas for thousands
#     plt.xticks(ticks=plt.xticks()[0], labels=[f'{int(x):,}' for x in plt.xticks()[0]])
#     plt.yticks(ticks=plt.yticks()[0], labels=[f'{int(x):,}' for x in plt.yticks()[0]])
#
#     plt.xlim(0, 30000)
#     plt.grid()
#
#     # Manually create legend entries for charging type with gray markers
#     type_legend_elements = [
#         Line2D([0], [0], marker=marker_map['Bidirectional'], color='w', label='V2G', markerfacecolor='gray', markersize=text_size),
#         Line2D([0], [0], marker=marker_map['Smart'], color='w', label='V1G', markerfacecolor='gray', markersize=text_size)
#     ]
#
#     # Manually create legend entries for charging speeds
#     speed_legend_elements = [
#         Line2D([0], [0], marker='o', color='w', label='6.6 kW', markerfacecolor='blue', markersize=text_size),
#         Line2D([0], [0], marker='o', color='w', label='12 kW', markerfacecolor='green', markersize=text_size),
#         Line2D([0], [0], marker='o', color='w', label='19 kW', markerfacecolor='red', markersize=text_size),
#     ]
#
#     # Manually create legend entries for battery sizes
#     size_legend_elements = [
#         Line2D([0], [0], marker='o', color='w', label=f'{key} kWh', markerfacecolor='gray', markersize=value / 15)
#         for key, value in size_map.items()
#     ]
#
#     # Add the legend for charging type
#     type_legend = plt.legend(handles=type_legend_elements, title='Optimal\nCharging Type', loc='upper right', bbox_to_anchor=(1.50, 1.02), fontsize=text_size - 3, title_fontsize=17)
#
#     # Add the legend for charging speed
#     speed_legend = plt.legend(handles=speed_legend_elements, title='Optimal\nCharging Speed', loc='upper right', bbox_to_anchor=(1.54, 0.75), fontsize=text_size - 3, title_fontsize=17)
#
#     # Add the legend for battery sizes
#     size_legend = plt.legend(handles=size_legend_elements, title='Vehicle\nBattery Size', loc='upper right', bbox_to_anchor=(1.47, 0.41), fontsize=text_size - 3, title_fontsize=17)
#
#     # Adding all legends to the plot
#     plt.gca().add_artist(type_legend)
#     plt.gca().add_artist(speed_legend)
#     plt.gca().add_artist(size_legend)
#
#     # Adjust the layout to ensure everything fits within the figure
#     plt.tight_layout(rect=[0, 0, 0.75, 1])  # Adjust rect to make room for legends
#
#     # Save the plot
#     plt.savefig(f'{title}_{ylim}.png', dpi=300)
#
#     # Show the plot
#     plt.show()

def plot_saving_ev_vs_distance(df, ax=None, add_actual_lines=False, add_potential_lines=False, ylim=None, text_size=18, color_maps=None, title='title'):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))  # Create a new axis if none is provided

    # Map charging type to markers
    marker_map = {'Bidirectional': 'o', 'Smart': 's'}
    label_map = {'Bidirectional': 'V2G', 'Smart': 'V1G'}

    # Define colors for each charging speed
    if color_maps is None:
        color_maps = {6.6: '#FFD61F', 12: '#3370FF', 19: '#E00025'}

    # Assign colors based on exact charging speeds
    df['Color'] = df['Charging_Speed'].map(color_maps)

    # Define discrete sizes based on battery capacity
    size_map = {66: 150, 70: 200, 75: 250, 80: 300, 85: 350, 100: 400}
    df['Size'] = df['bat_cap'].map(size_map)

    # Plot data for each charging type
    for charging_type, marker in marker_map.items():
        subset = df[df['Charge_Type'] == charging_type]
        ax.scatter(
            x=subset['distance'],
            y=subset['Saving_TOU'] / subset['average_smart_years'],
            c=subset['Color'],
            s=subset['Size'],
            marker=marker,
            alpha=0.7
        )

    # Ensure ylim is a tuple and unpack it
    if ylim is not None and isinstance(ylim, tuple) and len(ylim) == 2:
        ax.set_ylim(ylim[0], ylim[1])
    else:
        raise ValueError("ylim must be a tuple of two values (min, max).")

    # Adding labels and title
    ax.set_xlabel('Distance (mile)', fontsize=text_size)
    ax.set_ylabel(f'Savings Compared to the Base Scenario ($/year)', fontsize=text_size - 1)
    ax.tick_params(axis='x', labelsize=text_size - 4, rotation=45)
    ax.tick_params(axis='y', labelsize=text_size)

    ax.set_xlim(0, 30000)
    ax.grid()

    # Set title
    ax.set_title(title, fontsize=text_size)

def plot_benefit_vs_degradation(df, num_vehicles, Utility="PGE", title='title', lb=0, ub=1000, title_size=18, axis_text_size=18):
    df1 = df.copy()
    df1 = df1[df1["Utility"] == Utility]
    # Convert Charging Speed to string and remove any trailing '.0'
    df1['Charging_Speed'] = df1['Charging_Speed'].astype(str).str.rstrip('.0')
    df1['Scenarios'] = df1['Charge_Type'] + ' - ' + df1['Charging_Speed'] + ' - ' + df1['Scenario'] + ' - ' + df1['V2G_Location']
    baseline_row = df1.iloc[-1]

    # Calculate baseline cost
    baseline_cost = baseline_row['Electricity_Cost'] + baseline_row['Degradation_Cost']

    # Calculate the net benefit (Total Cost Savings) against the baseline and degradation separately, adjusted per vehicle
    df1['Total_Benefit'] = (baseline_cost - (df1['Electricity_Cost'] + df1['Degradation_Cost'])) / num_vehicles
    df1['Degradation_Cost'] = df1['Degradation_Cost'] / num_vehicles

    # Define the speeds and locations for plotting
    speeds = ['6.6', '12', '19']  # Speeds to consider
    locations = ['Home', 'Home_Work']  # Locations for V2G scenarios
    location_labels = {'Home': '', 'Home_Work': ''}

    # Initialize plot
    fig, ax = plt.subplots(figsize=(6, 6))
    bar_height = 0.5  # Height for each grouped bar
    smart_colors = ['#fdad1a', '#fdad1a', '#fdad1a']  # Color for Smart Charging bars
    v2g_colors = {
        'Home': '#219f71',  # Color for V2G Home
        'Home_Work': '#004a6d',  # Color for V2G Home_Work
        'Degradation': '#c4383a'  # Red for degradation
    }

    y_positions = []  # Store y positions for each group
    scenario_labels = []  # Store labels for y positions
    group_positions = []  # To store the center of each group for annotation
    current_y = 2  # Current y position

    # Smart Charging Section: Add each speed separately with labels
    smart_bar_offsets = np.linspace(-bar_height / 20, bar_height / 20, len(speeds)) / 2

    for idx, speed in enumerate(speeds):
        smart_data = df1[
            (df1['Charge_Type'] == 'Smart') &
            (df1['Charging_Speed'] == speed) &
            (df1['V2G_Location'] == 'Home_Work')
            ]
        if not smart_data.empty:
            degradation_cost = -smart_data['Degradation_Cost'].values[0]
            total_benefit = smart_data['Total_Benefit'].values[0]

            # Plot Degradation and Benefit bars directly next to each other for each speed
            ax.barh(current_y + smart_bar_offsets[idx] - bar_height / 4 + 0.25, degradation_cost, height=bar_height, color=v2g_colors['Degradation'])
            ax.barh(current_y + smart_bar_offsets[idx] + bar_height / 4, total_benefit, height=bar_height, color=smart_colors[idx])

            # Add text labels for Degradation and Benefit
            ax.text(degradation_cost - 0.25, current_y + smart_bar_offsets[idx], f"${degradation_cost:.2f}", ha='right', va='center', fontsize=axis_text_size - 2)
            ax.text(total_benefit + 0.25, current_y + smart_bar_offsets[idx], f"${total_benefit:.2f}", ha='left', va='center', fontsize=axis_text_size - 2)

            # Add individual labels for each speed
            y_positions.append(current_y + smart_bar_offsets[idx])
            scenario_labels.append(f"{speed} kW")  # Only show speed for Smart Charging
            current_y += 0.6
    # Calculate center of Smart Charging section for annotation
    group_positions.append(np.mean([2.5 + offset for offset in smart_bar_offsets]))

    # Draw horizontal dashed lines to indicate separation
    first_line_position = current_y - 0.15
    second_line_position = current_y + 4.25
    ax.axhline(first_line_position, color='black', linestyle='--', linewidth=1)
    ax.axhline(second_line_position, color='black', linestyle='--', linewidth=1)
    ax.set_xlim(-lb, ub)

    # Helper function to plot side-by-side bars for V2G with locations
    def plot_v2g_section(data, section_label):
        # global current_y  # Use global instead of nonlocal since it's in the main scope
        nonlocal current_y
        centers = []

        for speed in speeds:
            bar_offsets = np.linspace(-bar_height * 1.15, bar_height * 1.15, len(locations)) / 2
            for i, loc in enumerate(locations):
                loc_data = data[(data['Charging_Speed'] == speed) & (data['V2G_Location'] == loc)]
                if not loc_data.empty:
                    degradation_cost = -loc_data['Degradation_Cost'].values[0]
                    total_benefit = loc_data['Total_Benefit'].values[0]

                    # Plot bars for degradation and benefit
                    ax.barh(current_y + bar_offsets[i] + 0.5, degradation_cost, height=bar_height, color=v2g_colors['Degradation'])
                    ax.barh(current_y + bar_offsets[i] + bar_height, total_benefit, height=bar_height, color=v2g_colors[loc])

                    # Add text labels
                    ax.text(degradation_cost - 0.2, current_y + bar_offsets[i] + 0.5, f"${degradation_cost:.2f}", ha='right', va='center', fontsize=axis_text_size - 2)
                    ax.text(total_benefit + 0.2, current_y + bar_offsets[i] + 0.5, f"${total_benefit:.2f}", ha='left', va='center', fontsize=axis_text_size - 2)

                    # Add labels
                    y_positions.append(current_y + bar_offsets[i] + 0.5)
                    scenario_labels.append(f"{speed} kW {location_labels[loc]}")
                    centers.append(current_y + bar_offsets[i])

            current_y += 1.5  # Update the y position
        return np.mean(centers)  # Return the center position

        # Plot V2G Actual section by all speeds and store its center

    v2g_actual_data = df1[(df1['Charge_Type'] == 'Bidirectional') & (df1['Scenario'] == 'No_change')]
    if not v2g_actual_data.empty:
        center = plot_v2g_section(v2g_actual_data, "V2G / No Change in Plugging Behavior")
        group_positions.append(first_line_position + 2.75)

    # Plot V2G Potential section by all speeds and store its center
    v2g_potential_data = df1[(df1['Charge_Type'] == 'Bidirectional') & (df1['Scenario'] == 'With_change')]
    if not v2g_potential_data.empty:
        center = plot_v2g_section(v2g_potential_data, "V2G / Plugging-in When Parked")
        group_positions.append(second_line_position + 2.75)

    # Customize y-axis labels with all speeds retained
    ax.set_yticks(y_positions)
    ax.set_yticklabels(scenario_labels, fontsize=axis_text_size - 2)
    ax.axvline(0, color='grey', linewidth=1)
    ax.set_xlabel('Benefit ($) / Degradation Cost ($)', fontsize=title_size)
    ax.grid(True, axis='x', linestyle='--', alpha=0.7)
    plt.xticks(fontsize=axis_text_size)

    # Determine a suitable x-position for the annotations to ensure visibility
    x_annotation_position = ub * 1.10  # This positions the text at 90% of the upper x-limit, adjust as needed

    # Add group annotations to the right side of the plot within the visible range
    group_labels = ['V1G', 'V2G\nNo Change in\nPlugging Behavior', 'V2G\nPlugging-in\nWhen Parked']
    for pos, label in zip(group_positions, group_labels):
        ax.text(x_annotation_position, pos, label, ha='center', va='center', fontsize=title_size - 2, weight='bold', rotation=90)

    # Correct legend creation by wrapping smart_patches[0] in brackets
    smart_patches = [mpatches.Patch(color=smart_colors[i], label=f"Smart Charging") for i in range(len(speeds))]
    v2g_patches = [
        mpatches.Patch(color=v2g_colors['Home'], label="Bidirectional Charger\n at Home"),
        mpatches.Patch(color=v2g_colors['Home_Work'], label="Bidirectional Charger\n at Home + Work")
    ]
    degradation_patch = mpatches.Patch(color=v2g_colors['Degradation'], label="Battery Degradation")

    ax.legend(handles=[smart_patches[0]] + v2g_patches + [degradation_patch], loc='lower right', fontsize=axis_text_size - 2)

    # Customize x-axis ticks and labels
    ax.set_xticks([ax.get_xlim()[0], 0, ax.get_xlim()[1]])
    ax.set_xticklabels([f'$ Loss', f'Baseline Cost (${round((baseline_cost / num_vehicles), 2)})', '$ Savings'], fontsize=axis_text_size - 3)
    ax.set_ylabel("Charging Scenarios by Charger Speed and Deployment Location", fontsize=title_size)
    ax.set_xlabel("Net Benefit and Associated Degradation Cost per Vehicle ($)", fontsize=title_size)

    plt.savefig(f'{title}.png', bbox_inches='tight', dpi=300)
    plt.tight_layout()
    plt.show()


def plot_cdf_by_group(df, df2, column_to_plot, xlabel, figsize=(10, 6)):
    def plot_cdf(data, label, **kwargs):
        """Helper function to plot the CDF of a data series."""
        sorted_data = np.sort(data)
        cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        plt.plot(sorted_data, cdf, label=label, linewidth=2, **kwargs)

    group_by_columns = ['Charging Type', 'Charging Speed', 'Tariff']
    # filtered_df = df[df['Tariff'] == 'RT Rate - Home']
    filtered_df = df
    # Group the DataFrame by specified columns
    grouped = filtered_df.groupby(group_by_columns)
    plt.figure(figsize=figsize)
    # Loop through each group and plot the CDF for the specified column
    for name, group in grouped:
        # Assuming 'name' is a tuple with the same order as 'group_by_columns'
        if len(name) == 3:  # Ensure there are exactly three components to unpack
            charging_type, charging_speed, tariff = name

            # Properly capitalize 'smart' to 'Smart' and 'v2g' to 'V2G'
            if charging_type.lower() == 'smart':
                charging_type = 'Smart'
            elif charging_type.lower() == 'v2g':
                charging_type = 'V2G'
            # Remove ' - Home' from the tariff name
            tariff = tariff.replace(' - Home', '')
            # Create the label
            label = f'{charging_type}, {charging_speed} kW, {tariff}'
        else:
            label = ', '.join([str(val) for val in name])  # Fallback if the structure is unexpected
        # Plot the CDF for this group
        plot_cdf(group[column_to_plot], label)
        # Plot the baseline DataFrame (df2)
    if df2 is not None and column_to_plot in df2.columns:
        plot_cdf(df2[column_to_plot], label='Baseline', linestyle='--', color='black')  # Different style for the baseline

    # Set the x and y labels with font size
    plt.xlabel(xlabel, fontsize=18)
    plt.ylabel("CDF", fontsize=18)
    # Adjust x-ticks and y-ticks font size
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    # Add legend with increased font size
    plt.legend(loc='best', fontsize=14)  # Increase the font size of the legend
    plt.grid(True)
    plt.xlim(-25, 25)  # Adjust as needed
    # Display the plot
    plt.show()


def plot_styled_box_by_scenario(df1, baseline_df1, xaxis, x_column, y_column, charge_type_column, tariff_column, behavior_column, speed_column, xlimit):
    df = df1.copy()
    baseline_df = baseline_df1.copy()
    scenarios = {
        'Baseline': (baseline_df, None),  # Baseline data as a single box
        'Smart EV Rate': (df, ('smart', 'EV Rate - Home')),
        'Smart RT Rate': (df, ('smart', 'RT Rate - Home')),
        'V2G EV Rate': (df, ('v2g', 'EV Rate - Home')),
        'V2G RT Rate': (df, ('v2g', 'RT Rate - Home'))
    }

    colors = {
        'charging': {6.6: '#80a5b6', 12: '#004a6d'},
        'discharging': {6.6: '#dc8889', 12: '#b03234'}
    }

    plt.figure(figsize=(8, 6))

    base_spacing_smart = 3
    base_spacing_v2g = 4
    stack_spacing_smart = 1.0
    stack_spacing_v2g = 2
    v2g_rt_offset = 4

    max_width = 1.5
    min_width = 0.3
    baseline_frequency = len(baseline_df)

    y_pos = 0
    y_tick_positions = []
    y_tick_labels = []

    # If the x_column is 'GHG_Value', modify the column based on abs(X_CHR)
    if x_column == 'GHG_value':
        df[x_column] = df[x_column] * df['X_CHR'].abs()
        baseline_df[x_column] = baseline_df[x_column] * baseline_df['X_CHR'].abs()

    for scenario_name, scenario_info in scenarios.items():
        scenario_data, scenario_conditions = scenario_info

        positions = []
        data_to_plot = []
        box_colors = []
        box_widths = []

        if scenario_name == 'Baseline':
            data_to_plot.append(scenario_data[x_column])
            positions.append(y_pos)
            box_colors.append('#fdad1a')
            box_widths.append(max_width)
            y_tick_positions.append(y_pos)
            y_tick_labels.append(scenario_name)
            y_pos += base_spacing_smart

            # Add text for the Baseline
            plt.text(xlimit * 0.7, y_pos - 2 * stack_spacing_smart, f"#{len(scenario_data)}", fontsize=10, va='center', color='black', ha='center')

        else:
            if 'v2g' in scenario_name.lower():
                base_spacing = base_spacing_v2g
                stack_spacing = stack_spacing_v2g
                if scenario_name == 'V2G RT Rate':
                    y_pos += v2g_rt_offset

            else:
                base_spacing = base_spacing_smart
                stack_spacing = stack_spacing_smart

            charging_type, tariff = scenario_conditions
            scenario_data_filtered = scenario_data[(scenario_data[charge_type_column] == charging_type) &
                                                   (scenario_data[tariff_column] == tariff)]

            if 'v2g' in charging_type:
                discharging_data = scenario_data_filtered[scenario_data_filtered[behavior_column] < 0]
                for j, (speed, group) in enumerate(discharging_data.groupby(speed_column)):
                    data_to_plot.append(group[x_column])
                    positions.append(y_pos + j * stack_spacing)
                    box_colors.append(colors['discharging'][speed])
                    width = min_width + (len(group) / baseline_frequency) * (max_width - min_width)
                    box_widths.append(width)

                    # Add text label for frequency
                    plt.text(xlimit * 0.80, y_pos + j * stack_spacing, f"#{len(group)}", fontsize=10, va='center', color='black')

                charging_data = scenario_data_filtered[scenario_data_filtered[behavior_column] > 0]
                for j, (speed, group) in enumerate(charging_data.groupby(speed_column)):
                    data_to_plot.append(group[x_column])
                    positions.append(y_pos + (j + 2) * stack_spacing)
                    box_colors.append(colors['charging'][speed])
                    width = min_width + (len(group) / baseline_frequency) * (max_width - min_width)
                    box_widths.append(width)

                    # Add text label for frequency
                    plt.text(xlimit * 0.80, y_pos + (j + 2) * stack_spacing, f"#{len(group)}", fontsize=10, va='center', color='black')

                y_tick_positions.append(y_pos + stack_spacing + 2)
            else:
                charging_data = scenario_data_filtered[scenario_data_filtered[behavior_column] > 0]
                for j, (speed, group) in enumerate(charging_data.groupby(speed_column)):
                    data_to_plot.append(group[x_column])
                    positions.append(y_pos + j * stack_spacing)
                    box_colors.append(colors['charging'][speed])
                    width = min_width + (len(group) / baseline_frequency) * (max_width - min_width)
                    box_widths.append(width)

                    # Add text label for frequency
                    plt.text(xlimit * 0.80, y_pos + j * stack_spacing, f"#{len(group)}", fontsize=10, va='center', color='black')

                y_tick_positions.append(y_pos + (stack_spacing / 2))

            y_tick_labels.append(scenario_name)
            y_pos += base_spacing

        bplot = plt.boxplot(data_to_plot, positions=positions, vert=False, patch_artist=True, widths=box_widths, whis=3.3)
        for patch, color in zip(bplot['boxes'], box_colors):
            patch.set_facecolor(color)

    plt.hlines(y=2, xmin=-xlimit, xmax=xlimit, linestyles='--', color='grey', linewidth=1)
    plt.hlines(y=5, xmin=-xlimit, xmax=xlimit, linestyles='--', color='grey', linewidth=1)
    plt.hlines(y=8, xmin=-xlimit, xmax=xlimit, linestyles='--', color='grey', linewidth=1)
    plt.hlines(y=16, xmin=-xlimit, xmax=xlimit, linestyles='--', color='grey', linewidth=1)

    plt.yticks(y_tick_positions, y_tick_labels, fontsize=12)
    plt.xticks(fontsize=14)
    plt.xlim(-xlimit, xlimit)
    plt.xlabel(xaxis, fontsize=16)

    # Create legend with an additional entry for box width meaning
    legend_patches = [
        Patch(color='#dc8889', label='Discharging - 6.6 kW'),
        Patch(color='#b03234', label='Discharging - 12 kW'),
        Patch(color='#80a5b6', label='Charging - 6.6 kW'),
        Patch(color='#004a6d', label='Charging - 12 kW'),
        Patch(color='#fdad1a', label='Baseline'),
        Patch(color='white', edgecolor='black', label='Box width indicates\n charging frequency')
    ]
    plt.legend(handles=legend_patches, loc='lower left', fontsize=12)

    plt.tight_layout()
    plt.show()


def plot_styled_box_by_scenarioP(df1, baseline_df1, xaxis, x_column, y_column, charge_type_column, tariff_column, behavior_column, speed_column, xlimit):
    df = df1.copy()
    baseline_df = baseline_df1.copy()
    scenarios = {
        'Baseline': (baseline_df, None),  # Baseline data as a single box
        'Smart EV Rate': (df, ('smart', 'EV Rate - Home')),
        'Smart RT Rate': (df, ('smart', 'RT Rate - Home')),
        'V2G EV Rate': (df, ('v2g', 'EV Rate - Home')),
        'V2G RT Rate': (df, ('v2g', 'RT Rate - Home'))
    }

    colors = {
        'charging': {6.6: '#80a5b6', 12: '#004a6d'},
        'discharging': {6.6: '#dc8889', 12: '#b03234'}
    }

    plt.figure(figsize=(8, 6))

    base_spacing_smart = 5
    base_spacing_v2g = 6
    stack_spacing_smart = 2.0
    stack_spacing_v2g = 6
    v2g_rt_offset = 17

    max_width = 1.5
    min_width = 0.3
    baseline_frequency = len(baseline_df)

    y_pos = 0
    y_tick_positions = []
    y_tick_labels = []

    # If the x_column is 'GHG_Value', modify the column based on abs(X_CHR)
    if x_column == 'GHG_value':
        df[x_column] = df[x_column] * df['X_CHR'].abs()
        baseline_df[x_column] = baseline_df[x_column] * baseline_df['X_CHR'].abs()

    for scenario_name, scenario_info in scenarios.items():
        scenario_data, scenario_conditions = scenario_info

        positions = []
        data_to_plot = []
        box_colors = []
        box_widths = []

        if scenario_name == 'Baseline':
            data_to_plot.append(scenario_data[x_column])
            positions.append(y_pos)
            box_colors.append('#fdad1a')
            box_widths.append(max_width)
            y_tick_positions.append(y_pos)
            y_tick_labels.append(scenario_name)
            y_pos += base_spacing_smart

            # Add text for the Baseline
            plt.text(xlimit * 0.7, y_pos - 2 * stack_spacing_smart, f"#{len(scenario_data)}", fontsize=10, va='center', color='black', ha='center')

        else:
            if 'v2g' in scenario_name.lower():
                base_spacing = base_spacing_v2g
                stack_spacing = stack_spacing_v2g
                if scenario_name == 'V2G RT Rate':
                    y_pos += v2g_rt_offset

            else:
                base_spacing = base_spacing_smart
                stack_spacing = stack_spacing_smart

            charging_type, tariff = scenario_conditions
            scenario_data_filtered = scenario_data[(scenario_data[charge_type_column] == charging_type) &
                                                   (scenario_data[tariff_column] == tariff)]

            if 'v2g' in charging_type:
                discharging_data = scenario_data_filtered[scenario_data_filtered[behavior_column] < 0]
                for j, (speed, group) in enumerate(discharging_data.groupby(speed_column)):
                    data_to_plot.append(group[x_column])
                    positions.append(y_pos + j * stack_spacing)
                    box_colors.append(colors['discharging'][speed])
                    width = min_width + (len(group) / baseline_frequency) * (max_width - min_width)
                    box_widths.append(width)

                    # Add text label for frequency
                    plt.text(xlimit * 0.80, y_pos + j * stack_spacing, f"#{len(group)}", fontsize=10, va='center', color='black')

                charging_data = scenario_data_filtered[scenario_data_filtered[behavior_column] > 0]
                for j, (speed, group) in enumerate(charging_data.groupby(speed_column)):
                    data_to_plot.append(group[x_column])
                    positions.append(y_pos + (j + 2) * stack_spacing)
                    box_colors.append(colors['charging'][speed])
                    width = min_width + (len(group) / baseline_frequency) * (max_width - min_width)
                    box_widths.append(width)

                    # Add text label for frequency
                    plt.text(xlimit * 0.80, y_pos + (j + 2) * stack_spacing, f"#{len(group)}", fontsize=10, va='center', color='black')

                y_tick_positions.append(y_pos + stack_spacing + 2)
            else:
                charging_data = scenario_data_filtered[scenario_data_filtered[behavior_column] > 0]
                for j, (speed, group) in enumerate(charging_data.groupby(speed_column)):
                    data_to_plot.append(group[x_column])
                    positions.append(y_pos + j * stack_spacing)
                    box_colors.append(colors['charging'][speed])
                    width = min_width + (len(group) / baseline_frequency) * (max_width - min_width)
                    box_widths.append(width)

                    # Add text label for frequency
                    plt.text(xlimit * 0.80, y_pos + j * stack_spacing, f"#{len(group)}", fontsize=10, va='center', color='black')

                y_tick_positions.append(y_pos + (stack_spacing / 2))

            y_tick_labels.append(scenario_name)
            y_pos += base_spacing

        bplot = plt.boxplot(data_to_plot, positions=positions, vert=False, patch_artist=True, widths=box_widths, whis=3.3)
        for patch, color in zip(bplot['boxes'], box_colors):
            patch.set_facecolor(color)

    plt.hlines(y=3, xmin=-xlimit, xmax=xlimit, linestyles='--', color='grey', linewidth=1)
    plt.hlines(y=9, xmin=-xlimit, xmax=xlimit, linestyles='--', color='grey', linewidth=1)
    plt.hlines(y=13.5, xmin=-xlimit, xmax=xlimit, linestyles='--', color='grey', linewidth=1)
    plt.hlines(y=35, xmin=-xlimit, xmax=xlimit, linestyles='--', color='grey', linewidth=1)

    plt.yticks(y_tick_positions, y_tick_labels, fontsize=12)
    plt.xticks(fontsize=14)
    # plt.ylabel(y_column, fontsize=16)
    plt.xlim(-xlimit, xlimit)
    plt.xlabel(xaxis, fontsize=16)

    # Create legend with an additional entry for box width meaning
    legend_patches = [
        Patch(color='#dc8889', label='Discharging - 6.6 kW'),
        Patch(color='#b03234', label='Discharging - 12 kW'),
        Patch(color='#80a5b6', label='Charging - 6.6 kW'),
        Patch(color='#004a6d', label='Charging - 12 kW'),
        Patch(color='#fdad1a', label='Baseline'),
        # Patch(color='white', edgecolor='black', label='Box width indicates\n charging frequency')
    ]
    plt.legend(handles=legend_patches, loc='lower left', fontsize=11)

    plt.tight_layout()
    plt.show()


def plot_filtered_data(df, ghg_value, chtype):
    # Fill NaN values in all columns (if any) to consider them in one group
    df.fillna('Actual', inplace=True)

    # Filter data based on GHG value
    filtered_df = df[((df['GHG Cost'] == ghg_value) & (df['Charging Type'] == chtype)) | (df['Charging Type'] == 'Actual')]
    filtered_df = filtered_df[(filtered_df['Tariff'].str.contains('Home') & ~filtered_df['Tariff'].str.contains('Home&Work')) | ((filtered_df['Charging Type'] == 'Actual'))]
    # Group the data based on the columns except 'daily_hour' and 'X_CHR'
    grouped = filtered_df.groupby(['Charging Type', 'Charging Speed', 'GHG Cost', 'Tariff', 'Charging_Behavior'])

    # Plot the data
    plt.figure(figsize=(12, 6))

    for name, group in grouped:
        plt.plot(group['daily_hour'], group['X_CHR'], label=str(name))

    plt.xlabel('Hour', fontsize=14)
    plt.ylabel('Power kW', fontsize=14)
    plt.title(f'Charging Demand Curve (GHG Cost: {ghg_value})')
    plt.legend(title='Groups', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.grid(True)
    plt.show()


def calculate_benefit(row, baseline_rows):
    baseline = baseline_rows[(baseline_rows['Vehicle'] == row['Vehicle']) &
                             (baseline_rows['Social Cost of Carbon'] == row['Social Cost of Carbon'])]
    baseline_cost = baseline['Total_Cost'].values[0] if not baseline.empty else None
    return baseline_cost - row['Total_Cost'] if baseline_cost is not None else None


# Define the plotting function
def plot_benefit_by_scenario(df, scenario_filter='Actual', charging_speed=19, fz=18):
    baseline_rows = df[df['Scenario'] == 'Actual-Actual-TOU Rate']
    filtered_df = df[(df['Scenario'].str.contains(scenario_filter)) & (df['Charging Speed'] == charging_speed)]

    filtered_df['Benefit'] = filtered_df.apply(lambda row: calculate_benefit(row, baseline_rows), axis=1)
    filtered_df['Charging_Type'] = filtered_df['Scenario'].apply(lambda x: 'V1G' if 'Smart' in x else 'V2G')
    filtered_df['Charging_Tariff'] = filtered_df['Charging_Type'] + "-" + filtered_df['Tariff']

    plt.figure(figsize=(8, 6))
    palette = {0.05: '#219f71', 0.191: '#004a6d'}
    sns.boxplot(data=filtered_df, x='Charging_Tariff', y='Benefit', hue='Social Cost of Carbon',
                palette=palette, order=['V1G-TOU Rate', 'V1G-EV Rate', 'V1G-RT Rate',
                                        'V2G-TOU Rate', 'V2G-EV Rate', 'V2G-RT Rate'])

    # Dashed line to separate V1G and V2G sections
    plt.axvline(2.5, color='grey', linestyle='--', linewidth=1)

    # Set up the Y-axis formatter to add dollar sign
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:,.0f}'))

    # Customize labels and legend
    plt.xlabel('Charging Type and Tariff', fontsize=fz)
    plt.ylabel('Benefit from Optimal Charging', fontsize=fz)
    plt.xticks(rotation=45, fontsize=fz)
    plt.yticks(fontsize=fz)
    plt.grid(axis='y', alpha=0.5)

    # Update legend labels
    handles, _ = plt.gca().get_legend_handles_labels()
    new_labels = ['$ 50 / metric tonne of CO\u2082', '$ 191 / metric tonne of CO\u2082']
    plt.legend(handles=handles, labels=new_labels, title='Social Cost of Carbon', loc='upper left',
               fontsize=fz, title_fontsize=fz)

    plt.tight_layout()
    plt.savefig(f'plot.png', bbox_inches='tight', dpi=300)
    plt.show()
    return filtered_df


def calculate_ghg_difference(df1, df2, df3):
    df3.rename(columns={df3.columns[0]: 'GHG'}, inplace=True)
    df2.loc[df2["Rate_Type"] == "EV", "Rate_Type"] = "EV_Rate"
    df2.loc[df2["Rate_Type"] == "RT", "Rate_Type"] = "RT_Rate"
    # Merge the GHG values based on the Hour column
    df1 = pd.merge(df1, df3, left_on="Hour", right_index=True, how="left")
    df2 = pd.merge(df2, df3, left_on="Hour", right_index=True, how="left")
    # Calculate total GHG for CDF_N and costs_A_TOU_rate_hourly_in
    df1["GHG_Total"] = df1["GHG"] * abs(df1["X_CHR"])
    df1['Charging'] = df1['X_CHR'].apply(lambda x: x if x > 0 else 0)
    df1['Discharging'] = df1['X_CHR'].apply(lambda x: -x if x < 0 else 0)
    df1['GHG_Charging'] = df1['Charging'] * df1['GHG']
    df1['GHG_Discharging'] = df1['Discharging'] * df1['GHG']
    df1['GHG_Net'] = df1['GHG_Charging'] - df1['GHG_Discharging']
    df2["GHG_Total_actual"] = df2["GHG"] * df2["X_CHR"]

    # Group CDF_N by specified columns and sum GHG_Total
    grouped = df1.groupby(["Vehicle", "Charge_Type", "Charging_Speed", "Rate_Type", "Utility", "V2G_Location"])[["GHG_Charging", "GHG_Discharging", "GHG_Net"]].sum().reset_index(drop=False)

    # Sum GHG_Total_actual by Vehicle for costs_A_TOU_rate_hourly_in
    costs_A_TOU_rate_hourly_in_sum = df2.groupby(["Vehicle", "Rate_Type", "Utility"])["GHG_Total_actual"].sum().reset_index(drop=False)

    # Merge CDF_N_grouped with costs_A_TOU_rate_hourly_in_sum on Vehicle
    grouped = pd.merge(grouped, costs_A_TOU_rate_hourly_in_sum, on=["Vehicle", "Utility", "Rate_Type"], how="left")

    # Calculate the difference between GHG_Total_actual and GHG_Total
    grouped["diff"] = (grouped["GHG_Total_actual"] - grouped["GHG_Net"]) / 1000000
    return grouped


def calculate_cost_difference(df1, df2):
    # Group CDF_N by specified columns and sum GHG_Total
    grouped = df1.groupby(["Vehicle", "Charge_Type", "Charging_Speed", "Scenario", "Rate_Type", "V2G_Location"])["Electricity_Cost"].sum().reset_index(drop=False)

    # Sum GHG_Total_actual by Vehicle for costs_A_TOU_rate_hourly_in
    costs_A_TOU_rate_hourly_in_sum = df2.groupby("Vehicle")["Electricity_Cost"].sum().reset_index(drop=False)
    costs_A_TOU_rate_hourly_in_sum.rename(columns={"Electricity_Cost": "Electricity_Cost_Actual"}, inplace=True)

    # Merge CDF_N_grouped with costs_A_TOU_rate_hourly_in_sum on Vehicle
    grouped = pd.merge(grouped, costs_A_TOU_rate_hourly_in_sum, on="Vehicle", how="left")

    # Calculate the difference between GHG_Total_actual and GHG_Total
    grouped["diff"] = (grouped["Electricity_Cost_Actual"] - grouped["Electricity_Cost"])
    return grouped


def draw_rose_chart_parking(df, text_size=13):
    # Step 1: Group by destination label and calculate the average parking time
    avg_parking_times = df.groupby('destination_label')['parking_time_minute'].mean()

    # Step 2: Calculate the proportion of trips for each destination
    destination_counts = df['destination_label'].value_counts(normalize=True)  # Get proportions

    # Step 3: Order destinations and get corresponding parking times and proportions
    destinations = ['Home', 'Work', 'Other']
    parking_times = [avg_parking_times.get(dest, 0) for dest in destinations]
    proportions = [destination_counts.get(dest, 0) for dest in destinations]

    # Step 4: Add a small constant to avoid zero ylim
    parking_times = [max(time, 1) for time in parking_times]  # Avoid zero parking times

    # Angles for each destination, adjusted for the proportions (as pie chart segments)
    total_proportions = np.cumsum([0] + proportions[:-1]) * 2 * np.pi  # Starting angles for each segment
    width = [prop * 2 * np.pi for prop in proportions]  # Width of each bar proportional to sample size
    custom_colors = ['#3062ae', '#fdad1a', '#c4383a']  # Example custom colors (red, green, blue)
    # Step 5: Use custom colors or default to seaborn palette
    if custom_colors is None:
        colors = sns.color_palette("crest", len(destinations))  # Default color scheme
    else:
        colors = custom_colors[:len(destinations)]  # Use custom colors if provided

    # Create a figure for the rose chart
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})

    # Plot bars with width proportional to destination proportions and radius to parking times
    bars = ax.bar(total_proportions, parking_times, color=colors,
                  width=width, edgecolor='w', align='edge')

    # Step 6: Add labels to the bars
    for angle, bar, dest, proportion in zip(total_proportions, bars, destinations, proportions):
        # Display average parking time and proportion as percentage
        ax.text(angle + bar.get_width() / 2, bar.get_height() + 10,
                f'{proportion * 100:.1f}%      ',
                ha='center', fontsize=text_size + 5)

    # Customize the chart
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, max(parking_times) * 1.2)  # Set the limit for radial axis
    ax.set_xticks(total_proportions + np.array(width) / 2)  # Center the xticks on each segment
    ax.set_xticklabels(destinations, fontsize=text_size + 5)

    # Add the y-axis label (radial axis)
    # ax.set_ylabel('Time by Location (minutes)', labelpad=40, fontsize=text_size + 5)  # Adjust labelpad as needed

    plt.yticks(fontsize=text_size + 2)
    plt.figtext(0.5, 0.02, 'Time by Location (minutes)', ha='center', fontsize=text_size + 5)
    plt.savefig('rose_chart_parking_with_proportions_full_circle.png', bbox_inches='tight', dpi=600)
    plt.show()


def draw_rose_chart_charging(df1, text_size=13):
    df = df1.copy()
    # Create the destination_new column based on the logic provided
    df['destination_new'] = df.apply(
        lambda row: f"{row['destination_label']} Charging" if pd.notna(row['energy[charge_type][type]']) else f"{row['destination_label']} Parking", axis=1
    )
    # Create the charging_pie column
    df['charging_pie'] = df['duration_charging_min'].fillna(0)
    df['charging_pie'] = df['charging_pie'] + 300

    # Step 1: Ensure the order of the destination_new column (Charging next to Parking for each destination)
    ordered_destinations = ['Home Charging', 'Home Parking', 'Work Charging', 'Work Parking', 'Other Charging', 'Other Parking']
    df['destination_new'] = pd.Categorical(df['destination_new'], categories=ordered_destinations, ordered=True)
    df = df.sort_values('destination_new')  # Sort by the specified order
    custom_colors = ['#3062ae', '#3fbcd9', '#c6891c', '#fdad1a', '#840965', '#c4383a']  # Custom colors for 6 segments
    # Step 2: Group by destination_new and calculate the average charging time (charging_pie)
    avg_charging_times = df.groupby('destination_new')['charging_pie'].mean()

    # Step 3: Calculate the proportion of trips for each destination_new
    destination_counts = df['destination_new'].value_counts(normalize=True)  # Get proportions

    # Step 4: Order destinations and get corresponding charging times and proportions
    destinations = df['destination_new'].unique()  # Get all unique values for destination_new
    charging_times = [avg_charging_times.get(dest, 0) for dest in destinations]

    proportions = [destination_counts.get(dest, 0) for dest in destinations]

    # Angles for each destination, adjusted for the proportions (as pie chart segments)
    total_proportions = np.cumsum([0] + proportions[:-1]) * 2 * np.pi  # Starting angles for each segment
    width = [prop * 2 * np.pi for prop in proportions]  # Width of each bar proportional to sample size

    # Step 5: Assign custom colors (you can specify different colors for Parking and Charging)
    if custom_colors is None:
        colors = sns.color_palette("crest", len(destinations))  # Default color scheme
    else:
        colors = custom_colors[:len(destinations)]  # Use custom colors if provided

    # Create a figure for the rose chart
    fig, ax = plt.subplots(figsize=(9, 8), subplot_kw={'projection': 'polar'})

    # Plot bars with width proportional to destination proportions and radius to charging times
    bars = ax.bar(total_proportions, charging_times, color=colors,
                  width=width, edgecolor='w', align='edge')

    # Step 6: Add labels to the bars with adjusted positioning for readability
    for angle, bar, dest, proportion in zip(total_proportions, bars, destinations, proportions):
        # Adjust text placement to avoid overlap
        rotation_angle = np.degrees(angle + bar.get_width() / 2)
        if rotation_angle > 180:
            alignment = 'right'
        else:
            alignment = 'left'
        ax.text(angle + bar.get_width() / 2, bar.get_height() + 20,
                f'{proportion * 100:.1f}%',
                ha=alignment, fontsize=text_size + 5)

    # Customize the chart
    ax.set_theta_offset(np.pi / 2)  # Keep the theta axis starting at the top (90 degrees)
    ax.set_theta_direction(-1)
    ax.set_ylim(-0, max(charging_times) * 1.2)  # Set the limit for radial axis from -400 minutes

    # Step 7: Remove default radial tick labels
    ax.set_yticklabels([])  # Remove the radial tick labels (e.g., -300, 0, 300)

    # Step 8: Add custom radial tick labels at the desired angle (90 degrees)
    radial_ticks = np.arange(0, 750, 100)  # Custom radial ticks, e.g., from 0 to 600
    radial_labels = ["", "", "", "0", "100", "200", "300", "400"]  # Custom labels for the radial axis
    for tick, label in zip(radial_ticks, radial_labels):
        ax.text(np.pi / 2, tick, f'{label}',  # Place custom radial labels at 90 degrees (top of the plot)
                ha='center', va='center', fontsize=text_size, rotation=45)

    # Step 9: Keep the x-axis (theta axis) labels intact
    ax.set_xticks(total_proportions + np.array(width) / 2)  # Center the xticks on each segment
    ax.set_xticklabels(destinations, fontsize=text_size + 5)

    # Add the y-axis label (radial axis) and adjust its position
    # Add the y-axis label (radial axis) and adjust its position
    # ax.set_ylabel('Time by Location (minutes)', fontsize=text_size + 5)
    ax.yaxis.set_label_coords(-0.15, 0.5)  # Adjust the position of the radial axis label
    # Add the title at the bottom of the figure
    plt.figtext(0.5, 0.02, 'Time by Location (minutes)', ha='center', fontsize=text_size + 5)

    plt.savefig('rose_chart_charging_with_proportions_full_circle.png', bbox_inches='tight', dpi=600)
    plt.show()


def calculate_charge_difference(df1):
    # Filter out rows where X_CHR is less than or equal to 0
    df1 = df1[df1["X_CHR"] > 0]

    # Group by the relevant columns and sum X_CHR
    grouped = df1.groupby(["Vehicle", "Charge_Type", "Charging_Speed", "V2G_Location", "Rate_Type", "Scenario"])["X_CHR"].sum().reset_index(drop=False)

    # Step 1: Get one row for each vehicle where the Charging Type is "smart"
    smart_charging = grouped[grouped["Charge_Type"] == "Smart"].drop_duplicates(subset=["Vehicle"], keep="first")

    # Step 2: Merge the baseline (smart charging X_CHR) back into the original grouped DataFrame
    grouped = grouped.merge(smart_charging[["Vehicle", "X_CHR"]], on="Vehicle", how="left", suffixes=("", "_baseline"))
    grouped = grouped[grouped["Charge_Type"] != "Smart"]
    grouped["diff_kwh"] = grouped["X_CHR"] - grouped["X_CHR_baseline"]
    grouped["virtual_mile"] = grouped["diff_kwh"] * 2.6
    return grouped



def draw_box_plot(df1, ax, text_size=13, enable_secondary_yaxis=True):
    df = df1.copy()
    df.loc[df["Rate_Type"] == "EV_Rate", "Rate_Type"] = "EV Rate"
    df.loc[df["Rate_Type"] == "RT_Rate", "Rate_Type"] = "RT Rate"
    # Step 2: Rename behaviors
    df['Scenario'] = df['Scenario'].replace({
        'No_change': 'No change in plugging behavior',
        'With_change': 'Plugging in when parked'
    })

    # Step 3: Convert Charging Speed to string and categorize
    df['Charging_Speed'] = df['Charging_Speed'].astype(str).str.rstrip('.0')
    df['Charging_Speed'] = pd.Categorical(df['Charging_Speed'], categories=['6.6', '12', '19'], ordered=True)

    # Step 4: Combine Charging_Behavior and Tariff
    df['Charging_Info'] = df['Scenario'] + ' | ' + df['Rate_Type']

    # Step 5: Define order for y-axis
    behavior_order = [
        'Plugging in when parked | RT Rate',
        'Plugging in when parked | EV Rate',
        'No change in plugging behavior | RT Rate',
        'No change in plugging behavior | EV Rate'
    ]
    df['Charging_Info'] = pd.Categorical(df['Charging_Info'], categories=behavior_order, ordered=True)
    df = df[~df["Charging_Info"].isna()]

    # Draw the box plot
    box_colors = ['#E00025', '#3370FF', '#FFD61F']
    sns.boxplot(
        y="Charging_Info",
        x="virtual_mile",
        hue="Charging_Speed",
        data=df,
        hue_order=['19', '12', '6.6'],
        palette=box_colors,
        ax=ax
    )

    # Remove the legend from the subplot
    ax.get_legend().remove()

    # Manually set y-axis labels
    primary_labels = [
        'Plugging in\nwhen parked',
        'Plugging in\nwhen parked',
        'No change in\nplugging behavior',
        'No change in\nplugging behavior'
    ]
    secondary_labels = ['RT Rate', 'EV Rate', 'RT Rate', 'EV Rate']

    ax.set_yticks(range(len(primary_labels)))
    ax.set_yticklabels(primary_labels, fontsize=text_size)

    # Add secondary y-axis if enabled
    if enable_secondary_yaxis:
        ax2 = ax.secondary_yaxis('right')
        ax2.set_yticks(range(len(secondary_labels)))
        ax2.set_yticklabels(secondary_labels, fontsize=text_size, va='center', ha='center', rotation=90)
        ax2.tick_params(axis='y', which='major', pad=20)  # Move the secondary y-tick labels away from the axis
    else:
        ax.secondary_yaxis('right').remove()  # Remove the secondary y-axis completely

    # Customize ticks and grid
    ax.tick_params(axis='x', labelsize=text_size)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{int(x):,}'))

    # Add horizontal lines for better readability
    ax.axhline(0.5, color='black', linestyle='--', linewidth=1)
    ax.axhline(1.5, color='black', linestyle='--', linewidth=1)
    ax.axhline(2.5, color='black', linestyle='--', linewidth=1)

    # Customize axis labels
    ax.set_xlabel("Virtual Mile", fontsize=text_size)
    ax.set_ylabel("", fontsize=text_size)

def plot_box_by_tariff(df1, df2, fz=18, figtitle="title", Rate_type="RT_Rate", show_dollar=False):
    # Concatenate the two DataFrames
    df1["Charging_Behavior"] = "No change in plugging behavior"
    df2["Charging_Behavior"] = "Plugging in when parked"
    df_combined = pd.concat([df1, df2], axis=0, ignore_index=True)

    # Filter the DataFrame based on the selected tariff
    df_filtered = df_combined[df_combined["Rate_Type"] == Rate_type].copy()

    # Rename charging behaviors and types as specified
    df_filtered["Charge_Type"] = df_filtered["Charge_Type"].replace({
        "Smart": "V1G",
        "Bidirectional": "V2G"
    })

    # Define the scenario order and add "Scenario" column for combined grouping
    scenario_order = [
        "V2G | Plugging in when parked",
        "V2G | No change in plugging behavior",
        "V1G | Plugging in when parked",
        "V1G | No change in plugging behavior"
    ]
    df_filtered['Scenario'] = df_filtered["Charge_Type"] + ' | ' + df_filtered["Charging_Behavior"]

    # Set the color palette for Charging Speed
    charging_speed_colors = {6.6: '#004a6d', 12: '#219f71', 19: '#c4383a'}

    # Set up the figure and main axis
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot the boxplot
    sns.boxplot(
        data=df_filtered,
        y="Scenario",
        x="diff",
        hue="Charging_Speed",
        palette=charging_speed_colors,
        dodge=True,
        order=scenario_order,
        ax=ax,
        hue_order=[19, 12, 6.6],  # Reversing the order within each group
        legend=False  # Disable automatic legend from Seaborn
    )

    # Adjust y-tick labels for primary y-axis
    y_labels_left = ["V2G", "V2G", "V1G", "V1G"]
    y_labels_left = ["Plugging in\nwhen\nparked", "No change in\nplugging\nbehavior", "Plugging in\nwhen\nparked", "No change in\nplugging\nbehavior"]
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(y_labels_left, ha='left', fontsize=fz)

    # Move the y-tick labels further outside the plot
    ax.tick_params(axis='y', pad=90)  # Increase pad to push y-ticks outside the plot
    ax.set_ylabel("", labelpad=16, fontsize=fz)

    # Secondary y-axis for Charging Behavior
    ax2 = ax.twinx()
    y_labels_right = ["V2G", "V2G", "V1G", "V1G"]
    ax2.set_ylim(ax.get_ylim())  # Sync limits
    ax2.set_yticks([0, 1, 2, 3])
    ax2.set_yticklabels(y_labels_right, rotation=90, va='center', ha="left", fontsize=fz + 2)

    # Add dashed lines to separate V1G and V2G groups
    plt.axhline(1.5, color='grey', linestyle='--', linewidth=1)
    plt.axhline(2.5, color='grey', linestyle='--', linewidth=1)
    plt.axhline(0.5, color='grey', linestyle='--', linewidth=1)

    # Customizing the legend to add "kW" after each speed
    legend_labels = [f"{speed} kW" for speed in charging_speed_colors.keys()]
    handles = [mpatches.Patch(color=color, label=label) for color, label in zip(charging_speed_colors.values(), legend_labels)]
    plt.legend(handles=handles, title="Charging Speed", loc='lower right', fontsize=fz, title_fontsize=fz)

    # Set title and labels
    ax.set_xlabel(figtitle, fontsize=fz + 2)

    # Adjust the x-axis ticks based on show_dollar parameter
    if show_dollar:
        x_ticks = ax.get_xticks()
        ax.set_xticklabels([f"${int(tick)}" for tick in x_ticks], fontsize=fz)
    else:
        ax.tick_params(axis='x', labelsize=fz)

    plt.tight_layout()
    plt.savefig(f'{figtitle}.png', bbox_inches='tight', dpi=300)
    plt.show()


def xlsx_read(file_path):
    try:
        # Load available sheet names
        xls = pd.ExcelFile(file_path)
        sheet_names = xls.sheet_names

        # Check if 'Total' and 'Costs' exist
        total_sheet = 'Total Costs' if 'Total Costs' in sheet_names else sheet_names[0]
        individual_sheet = 'Individual Costs' if 'Individual Costs' in sheet_names else sheet_names[0]

        # Read the sheets
        df_total = pd.read_excel(file_path, sheet_name=total_sheet)
        df_individual = pd.read_excel(file_path, sheet_name=individual_sheet)

        return df_total, df_individual

    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return pd.DataFrame(), pd.DataFrame()


def read_all_results(base_dir, rate_types, utilities, locations):
    total_costs_list = []
    costs_list = []

    for scenario in ["normal", "parking"]:
        scenario_path = os.path.join(base_dir, scenario)

        for file in os.listdir(scenario_path):
            if file.endswith(".xlsx"):
                file_path = os.path.join(scenario_path, file)

                try:
                    # Extract Rate_Type
                    match = re.search(r"(EV_Rate|RT_Rate|TOU)", file)
                    rate_type = match.group(1) if match else "Unknown"

                    # Extract Utility
                    utility = re.search(r"(pge|sce|sdge|smud)", file, re.IGNORECASE).group(0).upper()

                    # Extract V2G_Location
                    location_part = file.split("_")[3]
                    v2g_location = "Home_Work" if "Work" in location_part else "Home"

                    # Extract Charge Type (v2g → Bidirectional, otherwise Smart)
                    charge_type = "Bidirectional" if "v2g" in file.lower() else "Smart"

                    # Extract Charging Speed (6.6kW, 12kW, 19kW)
                    speed_match = re.search(r"(\d{1,2}\.?\d*)kw", file.lower())
                    charging_speed = f"{speed_match.group(1)}kW" if speed_match else "Unknown"

                    # Read the Excel file
                    df_total, df_costs = xlsx_read(file_path)

                    # Add metadata columns
                    df_total["Scenario"] = "No_change" if scenario == "normal" else "With_change"
                    df_costs["Scenario"] = "No_change" if scenario == "normal" else "With_change"
                    df_total["Rate_Type"] = rate_type
                    df_costs["Rate_Type"] = rate_type
                    df_total["Utility"] = utility
                    df_costs["Utility"] = utility
                    df_total["V2G_Location"] = v2g_location
                    df_costs["V2G_Location"] = v2g_location
                    df_total["Charge_Type"] = charge_type
                    df_costs["Charge_Type"] = charge_type
                    df_total["Charging_Speed"] = charging_speed
                    df_costs["Charging_Speed"] = charging_speed

                    total_costs_list.append(df_total)
                    costs_list.append(df_costs)

                except Exception as e:
                    print(f"Error processing file {file}: {e}")

    total_costs_combined = pd.concat(total_costs_list, ignore_index=True) if total_costs_list else pd.DataFrame()
    costs_combined = pd.concat(costs_list, ignore_index=True) if costs_list else pd.DataFrame()

    return total_costs_combined, costs_combined


def read_combined_costs(base_dir):
    """
    Reads cost files with two naming patterns:
    - 'Actual_<Utility>_<Rate>_cost_hourly_in.xlsx'
    - 'Actual_<Utility>_<Rate>_cost.xlsx'
    """
    hourly_costs_list = []
    aggregated_costs_list = []

    # Loop through all files in the base directory
    for file in os.listdir(base_dir):
        if file.endswith(".xlsx"):
            file_path = os.path.join(base_dir, file)
            try:
                # Extract metadata using regex
                utility_match = re.search(r"Actual_([A-Z]+)_", file, re.IGNORECASE)
                rate_match = re.search(r"_(EV|TOU|RT)_cost", file, re.IGNORECASE)
                utility = utility_match.group(1).upper() if utility_match else "Unknown"
                rate_type = rate_match.group(1) if rate_match else "Unknown"

                if "hourly_in" in file.lower():
                    # Hourly costs file
                    df_hourly = pd.read_excel(file_path)
                    df_hourly["Utility"] = utility
                    df_hourly["Rate_Type"] = rate_type
                    df_hourly["File_Type"] = "Hourly"
                    hourly_costs_list.append(df_hourly)

                elif "_cost.xlsx" in file.lower():
                    # Aggregated costs file
                    df_aggregated = pd.read_excel(file_path)
                    df_aggregated["Utility"] = utility
                    df_aggregated["Rate_Type"] = rate_type
                    df_aggregated["File_Type"] = "Aggregated"
                    aggregated_costs_list.append(df_aggregated)

            except Exception as e:
                print(f"Error reading {file_path}: {e}")

    # Combine all results
    costs_combined_hourly = pd.concat(hourly_costs_list, ignore_index=True) if hourly_costs_list else pd.DataFrame()
    costs_combined_aggregated = pd.concat(aggregated_costs_list, ignore_index=True) if aggregated_costs_list else pd.DataFrame()
    total_actual = costs_combined_aggregated.groupby(["Utility", "Rate_Type"])[["Electricity_Cost", "Degradation_Cost", "GHG_Cost", "Total Charge"]].sum().reset_index(drop=False)

    return costs_combined_hourly, costs_combined_aggregated, total_actual


def json_file_read_combined(base_dir, flatten_veh_data, scenario, utility_filter):
    results_list = []
    hourly_data_list = []

    # Path for the selected scenario
    scenario_path = os.path.join(base_dir, scenario)

    # Loop through JSON files for the specified utility
    for file in os.listdir(scenario_path):
        if file.endswith(".json") and utility_filter.lower() in file.lower() and "Home" in file:
            file_path = os.path.join(scenario_path, file)

            try:
                # Extract metadata from file name
                charge_type = "Bidirectional" if "v2g" in file.lower() else "Smart"

                speed_match = re.search(r"(\d{1,2}\.?\d*)kw", file.lower())
                charging_speed = f"{speed_match.group(1)}kW" if speed_match else "Unknown"

                rate_match = re.search(r"_(EV_Rate|TOU|RT_Rate)", file)
                rate_type = rate_match.group(1) if rate_match else "Unknown"

                location_part = re.search(r"(\[.*?\])", file)
                v2g_location = "Home_Work" if "Work" in location_part.group(1) else "Home"

                # Read JSON file
                df = pd.read_json(file_path)

                # Add metadata columns
                df["Charge_Type"] = charge_type
                df["Charging_Speed"] = charging_speed
                df["Rate_Type"] = rate_type
                df["Utility"] = utility_filter.upper()
                df["V2G_Location"] = v2g_location
                df["Scenario"] = "No_change" if scenario == "normal" else "With_change"

                # Merge with correct flattened data
                df = pd.merge(df, flatten_veh_data[["Vehicle", "Hour", "charging_indicator", "location"]],
                              how="left", on=["Vehicle", "Hour"])

                # Add to combined hourly data list
                hourly_data_list.append(df)

                # Process discharging data
                df_discharge = df[(df["X_CHR"] <= 0) & (df["charging_indicator"] == 1)]
                df_discharge = df_discharge[df_discharge["Charge_Type"] == "Bidirectional"]

                # Aggregate data
                grouped_df = df_discharge.groupby(['Hour', "Charging_Speed", "Rate_Type", "Utility", "V2G_Location"]).agg(
                    X_CHR_Sum=('X_CHR', 'sum'),
                    X_CHR_Count=('X_CHR', 'count')
                ).reset_index()

                # Calculate total power and utilization rate
                grouped_df["Total_power"] = grouped_df["Charging_Speed"].str.replace("kW", "").astype(float) * grouped_df["X_CHR_Count"]
                grouped_df["Utilization Rate"] = abs(grouped_df["X_CHR_Sum"] / grouped_df["Total_power"]) * 100

                # Add hour of day and peak indicator
                grouped_df["Hour_of_day"] = grouped_df["Hour"] % 24
                grouped_df["Peak"] = grouped_df["Hour_of_day"].apply(lambda x: "Peak" if 16 <= x <= 21 else "Non-Peak")

                # Append to results list
                results_list.append(grouped_df)

            except Exception as e:
                print(f"Error processing file {file_path}: {e}")

    # Combine all results
    combined_result = pd.concat(results_list, ignore_index=True) if results_list else pd.DataFrame()
    combined_hourly_data = pd.concat(hourly_data_list, ignore_index=True) if hourly_data_list else pd.DataFrame()

    return combined_result, combined_hourly_data


def process_and_save_all(base_dir, flatten_veh_data_normal, flatten_veh_data_parking):
    scenarios = ["normal", "parking"]
    utilities = ["PGE", "SCE", "SDGE", "SMUD"]

    for scenario in scenarios:
        for utility in utilities:
            print(f"Processing {scenario.upper()} scenario for {utility}...")

            # Select the correct flattened vehicle data
            flatten_veh_data = flatten_veh_data_normal if scenario == "normal" else flatten_veh_data_parking

            # Run the function to get combined results
            combined_result, combined_hourly_data = json_file_read_combined(base_dir, flatten_veh_data, scenario, utility)

            # Define file names
            result_file = f"combined_result_{scenario}_{utility}.pkl"
            hourly_file = f"combined_hourly_data_{scenario}_{utility}.pkl"

            # Save results as pickle
            combined_result.to_pickle(result_file)
            combined_hourly_data.to_pickle(hourly_file)

            print(f"Saved: {result_file} and {hourly_file}")


def all_rates(df1, df2):
    df2.rename(columns={"Total Charge": "X_CHR"}, inplace=True)
    All_rates_total_df = pd.concat([df1, df2], axis=0)
    All_rates_total_df.loc[All_rates_total_df["Scenario"].isna(), "Scenario"] = "No_change"
    All_rates_total_df.loc[All_rates_total_df["V2G_Location"].isna(), "V2G_Location"] = "None"
    All_rates_total_df.loc[All_rates_total_df["Charge_Type"].isna(), "Charge_Type"] = "Conventional"
    All_rates_total_df.loc[All_rates_total_df["Charging_Speed"].isna(), "Charging_Speed"] = "6.6kW"
    All_rates_total_df.loc[All_rates_total_df["Rate_Type"] == "EV", "Rate_Type"] = "EV_Rate"
    All_rates_total_df.loc[All_rates_total_df["Rate_Type"] == "RT", "Rate_Type"] = "RT_Rate"
    All_rates_total_df['Charging_Speed'] = All_rates_total_df['Charging_Speed'].str.replace('kW', '').astype(float)
    TOU_rates_total_df = All_rates_total_df.loc[(All_rates_total_df["Rate_Type"] == "TOU")]
    EV_rates_total_df = All_rates_total_df.loc[(All_rates_total_df["Rate_Type"] == "EV_Rate")]
    RT_rates_total_df = All_rates_total_df.loc[(All_rates_total_df["Rate_Type"] == "RT_Rate")]
    RT_rates_total_TOU_df = All_rates_total_df.loc[((All_rates_total_df["Rate_Type"] == "RT_Rate") & (All_rates_total_df["Charge_Type"] != "Conventional")) |
                                                   ((All_rates_total_df["Rate_Type"] == "TOU") & (All_rates_total_df["Charge_Type"] == "Conventional"))]
    RT_rates_total_EV_df = All_rates_total_df.loc[((All_rates_total_df["Rate_Type"] == "RT_Rate") & (All_rates_total_df["Charge_Type"] != "Conventional")) |
                                                  ((All_rates_total_df["Rate_Type"] == "EV_Rate") & (All_rates_total_df["Charge_Type"] == "Conventional"))]

    All_rates_total_6_df = All_rates_total_df.loc[(All_rates_total_df["Charge_Type"] == "Conventional") | (All_rates_total_df["Charging_Speed"] == 6.6)]
    All_rates_total_12_df = All_rates_total_df.loc[(All_rates_total_df["Charge_Type"] == "Conventional") | (All_rates_total_df["Charging_Speed"] == 12)]
    All_rates_total_19_df = All_rates_total_df.loc[(All_rates_total_df["Charge_Type"] == "Conventional") | (All_rates_total_df["Charging_Speed"] == 19)]
    return All_rates_total_df, TOU_rates_total_df, EV_rates_total_df, RT_rates_total_df, RT_rates_total_TOU_df, RT_rates_total_EV_df, All_rates_total_6_df, All_rates_total_12_df, All_rates_total_19_df


def plot_benefit_vs_degradation1(df, num_vehicles, Utility="PGE", title='title', lb=0, ub=1000, title_size=18, axis_text_size=18, ax=None, last_in_row=False, unified_x_title=False):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))  # Create a new plot if no axis is provided

    df1 = df.copy()
    df1 = df1[df1["Utility"] == Utility]
    # Convert Charging Speed to string and remove any trailing '.0'
    df1['Charging_Speed'] = df1['Charging_Speed'].astype(str).str.rstrip('.0')
    df1['Scenarios'] = df1['Charge_Type'] + ' - ' + df1['Charging_Speed'] + ' - ' + df1['Scenario'] + ' - ' + df1['V2G_Location']
    baseline_row = df1.iloc[-1]

    # Calculate baseline cost
    baseline_cost = baseline_row['Electricity_Cost'] + baseline_row['Degradation_Cost']

    # Calculate the net benefit (Total Cost Savings) against the baseline and degradation separately, adjusted per vehicle
    df1['Total_Benefit'] = (baseline_cost - (df1['Electricity_Cost'] + df1['Degradation_Cost'])) / num_vehicles
    df1['Degradation_Cost'] = df1['Degradation_Cost'] / num_vehicles

    # Define the speeds and locations for plotting
    speeds = ['6.6', '12', '19']  # Speeds to consider
    locations = ['Home', 'Home_Work']  # Locations for V2G scenarios
    location_labels = {'Home': '', 'Home_Work': ''}

    bar_height = 0.5  # Height for each grouped bar
    smart_colors = ['#fdad1a', '#fdad1a', '#fdad1a']  # Color for Smart Charging bars
    v2g_colors = {
        'Home': '#219f71',  # Color for V2G Home
        'Home_Work': '#004a6d',  # Color for V2G Home_Work
        'Degradation': '#c4383a'  # Red for degradation
    }

    y_positions = []  # Store y positions for each group
    scenario_labels = []  # Store labels for y positions
    group_positions = []  # To store the center of each group for annotation
    current_y = 2  # Current y position

    # Smart Charging Section: Add each speed separately with labels
    smart_bar_offsets = np.linspace(-bar_height / 20, bar_height / 20, len(speeds)) / 2

    for idx, speed in enumerate(speeds):
        smart_data = df1[
            (df1['Charge_Type'] == 'Smart') &
            (df1['Charging_Speed'] == speed) &
            (df1['V2G_Location'] == 'Home_Work')
            ]
        if not smart_data.empty:
            degradation_cost = -smart_data['Degradation_Cost'].values[0]
            total_benefit = smart_data['Total_Benefit'].values[0]

            # Plot Degradation and Benefit bars directly next to each other for each speed
            ax.barh(current_y + smart_bar_offsets[idx] - bar_height / 4 + 0.25, degradation_cost, height=bar_height, color=v2g_colors['Degradation'])
            ax.barh(current_y + smart_bar_offsets[idx] + bar_height / 4, total_benefit, height=bar_height, color=smart_colors[idx])

            # Add text labels for Degradation and Benefit
            # ax.text(degradation_cost - 0.25, current_y + smart_bar_offsets[idx], f"${degradation_cost:.0f}", ha='right', va='center', fontsize=axis_text_size)
            # ax.text(total_benefit + 0.25, current_y + smart_bar_offsets[idx], f"${total_benefit:.0f}", ha='left', va='center', fontsize=axis_text_size)

            # Add individual labels for each speed
            y_positions.append(current_y + smart_bar_offsets[idx])
            scenario_labels.append(f"{speed} kW")  # Only show speed for Smart Charging
            current_y += 0.6
    # Calculate center of Smart Charging section for annotation
    group_positions.append(np.mean([2.5 + offset for offset in smart_bar_offsets]))
    first_line_position = current_y - 0.15
    second_line_position = current_y + 4.25
    ax.axhline(first_line_position, color='black', linestyle='--', linewidth=1)
    ax.axhline(second_line_position, color='black', linestyle='--', linewidth=1)
    ax.set_xlim(-lb, ub)

    # Helper function to plot side-by-side bars for V2G with locations
    def plot_v2g_section(data, section_label):
        nonlocal current_y
        centers = []

        for speed in speeds:
            bar_offsets = np.linspace(-bar_height * 1.15, bar_height * 1.15, len(locations)) / 2
            for i, loc in enumerate(locations):
                loc_data = data[(data['Charging_Speed'] == speed) & (data['V2G_Location'] == loc)]
                if not loc_data.empty:
                    degradation_cost = -loc_data['Degradation_Cost'].values[0]
                    total_benefit = loc_data['Total_Benefit'].values[0]

                    # Plot bars for degradation and benefit
                    ax.barh(current_y + bar_offsets[i] + 0.5, degradation_cost, height=bar_height, color=v2g_colors['Degradation'])
                    ax.barh(current_y + bar_offsets[i] + bar_height, total_benefit, height=bar_height, color=v2g_colors[loc])

                    # Add text labels
                    # ax.text(degradation_cost - 0.2, current_y + bar_offsets[i] + 0.5, f"${degradation_cost:.0f}", ha='right', va='center', fontsize=axis_text_size)
                    # ax.text(total_benefit + 0.2, current_y + bar_offsets[i] + 0.5, f"${total_benefit:.0f}", ha='left', va='center', fontsize=axis_text_size)

                    # Add labels
                    y_positions.append(current_y + bar_offsets[i] + 0.5)
                    scenario_labels.append(f"{speed} kW {location_labels[loc]}")
                    centers.append(current_y + bar_offsets[i])

            current_y += 1.5  # Update the y position
        return np.mean(centers)  # Return the center position

    # Plot V2G Actual section by all speeds and store its center
    v2g_actual_data = df1[(df1['Charge_Type'] == 'Bidirectional') & (df1['Scenario'] == 'No_change')]
    if not v2g_actual_data.empty:
        center = plot_v2g_section(v2g_actual_data, "V2G / No Change in Plugging Behavior")
        group_positions.append(center)

    # Plot V2G Potential section by all speeds and store its center
    v2g_potential_data = df1[(df1['Charge_Type'] == 'Bidirectional') & (df1['Scenario'] == 'With_change')]
    if not v2g_potential_data.empty:
        center = plot_v2g_section(v2g_potential_data, "V2G / Plugging-in When Parked")
        group_positions.append(center)

    # Customize y-axis labels with all speeds retained
    ax.set_yticks(y_positions)
    ax.set_yticklabels(scenario_labels, fontsize=axis_text_size - 1)
    ax.axvline(0, color='grey', linewidth=1)
    ax.set_xlim(-lb, ub)
    ax.grid(True, axis='x', linestyle='--', alpha=0.7)

    # Add panel-specific title
    ax.set_title(title, fontsize=title_size)

    # Customize x-axis ticks and labels
    x_ticks = ax.get_xticks()

    # Replace the 0 value with the baseline value
    x_tick_labels = []
    baseline_found = False  # Flag to track when baseline is encountered

    for tick in x_ticks:
        if tick == 0:
            x_tick_labels.append(f'Baseline\n(${round((baseline_cost / num_vehicles), 2)})')
            baseline_found = True
        elif baseline_found and tick > 0:  # First tick on the right side of baseline
            x_tick_labels.append(f'')  # Replace with your desired text
            baseline_found = False  # Reset the flag so only the first positive tick is changed
        elif tick == x_ticks[0]:  # First tick (Loss)
            x_tick_labels.append("Loss")
        elif tick == x_ticks[-1]:  # Last tick (Savings)
            x_tick_labels.append("Savings")
        elif tick > 0:  # Last tick (Savings)
            x_tick_labels.append(f'$+{int(tick)}')
        else:  # Numbers in between
            x_tick_labels.append(f'${int(tick)}')

    # Update the x-axis ticks and labels
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_tick_labels)  # Ensure the labels are applied here

    # Apply rotation only to the numerical tick labels
    for tick, label in zip(ax.get_xticks(), ax.get_xticklabels()):
        label_text = label.get_text()
        if "$" in label_text and "Baseline" not in label_text:  # Rotate only numerical labels without "Baseline"
            label.set_rotation(30)  # Rotate numbers
            label.set_horizontalalignment('right')  # Align rotated text
            label.set_fontsize(axis_text_size - 2)  # Increase font size (adjust as needed)
        else:
            label.set_rotation(0)  # Keep "Loss," "Baseline," and "Savings" unrotated
            label.set_fontsize(axis_text_size)  # Increase font size (adjust as needed)

    # Add x-axis title only if not unified
    if not unified_x_title:
        ax.set_xlabel("", fontsize=title_size)

    # Add secondary y-axis for the last graph in the row
    if last_in_row:
        x_annotation_position = ub * 1.20
        group_labels = ['V1G', 'V2G\nNo Change in\nPlugging Behavior', 'V2G\nPlugging-in\nWhen Parked']
        group_offsets = [0, 0.5, 1.0]  # Offset for each label, in y-axis units

        for pos, label, offset in zip(group_positions, group_labels, group_offsets):
            ax.text(
                x_annotation_position,
                pos + offset,  # Adjust the y-position with the offset
                label,
                ha='center',
                va='center',
                fontsize=title_size - 2,
                weight='bold',
                rotation=90
            )
        # Add legend for the last graph in the row
        smart_patches = [mpatches.Patch(color=smart_colors[i], label=f"Smart Charging") for i in range(len(speeds))]
        v2g_patches = [
            mpatches.Patch(color=v2g_colors['Home'], label="Bidirectional Charger at Home"),
            mpatches.Patch(color=v2g_colors['Home_Work'], label="Bidirectional Charger at Home + Work")
        ]
        degradation_patch = mpatches.Patch(color=v2g_colors['Degradation'], label="Battery Degradation")

        # Place the legend outside the plot at the bottom in one line
        fig = ax.get_figure()
        fig.legend(
            handles=[smart_patches[0]] + v2g_patches + [degradation_patch],
            loc='lower center',
            fontsize=axis_text_size - 1,
            ncol=len(smart_patches) + len(v2g_patches) + 1,  # Number of columns in the legend
            bbox_to_anchor=(0.5, -0.01)  # Center the legend below the plot
        )

    return ax


def plot_benefit_vs_degradation_panel(data_list, num_vehicles, utilities, titles, lbs, ubs, y_title, x_title, figsize=(20, 8), title_size=18, axis_text_size=14):
    num_plots = len(data_list)  # Number of panels
    fig, axes = plt.subplots(1, num_plots, figsize=figsize, sharey=True)  # Create a single row of plots

    for i, ax in enumerate(axes):
        # Call the existing plot function, passing the specific axis
        plot_benefit_vs_degradation1(
            data_list[i],
            num_vehicles=num_vehicles,
            Utility=utilities[i],
            title=titles[i],
            lb=lbs[i],
            ub=ubs[i],
            title_size=title_size - 1,
            axis_text_size=axis_text_size,
            ax=ax,  # Pass the specific axis
            last_in_row=(i == len(axes) - 1),  # Add legend and secondary y-axis only for the last plot
            unified_x_title=(i != len(axes) - 1)  # Unified x-axis title only for the last plot
        )

    # Add shared y-axis and x-axis labels
    fig.text(0.01, 0.55, y_title, va='center', rotation='vertical', fontsize=title_size + 1)  # Y-axis
    fig.text(0.5, 0.04, x_title, ha='center', fontsize=title_size)  # X-axis

    # Adjust layout
    plt.tight_layout(rect=[0.04, 0.09, 1, 1])
    plt.show()


def plot_benefit_vs_degradation2(df, num_vehicles, Utility="PGE", title='title', lb=0, ub=1000, title_size=18, axis_text_size=18, ax=None, last_in_row=False, unified_x_title=False):
    global smart_data
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))  # Create a new plot if no axis is provided

    df1 = df.copy()
    df1 = df1[df1["Utility"] == Utility]
    # Convert Charging Speed to string and remove any trailing '.0'
    df1['Charging_Speed'] = df1['Charging_Speed'].astype(str).str.rstrip('.0')
    df1['Scenarios'] = df1['Charge_Type'] + ' - ' + df1['Charging_Speed'] + ' - ' + df1['Scenario'] + ' - ' + df1['V2G_Location']
    baseline_row = df1.iloc[-1]

    # Calculate baseline cost
    baseline_cost = baseline_row['Electricity_Cost'] + baseline_row['Degradation_Cost']

    # Calculate the net benefit (Total Cost Savings) against the baseline and degradation separately, adjusted per vehicle
    df1['Total_Benefit'] = (baseline_cost - (df1['Electricity_Cost'] + df1['Degradation_Cost'])) / num_vehicles
    df1['Degradation_Cost'] = df1['Degradation_Cost'] / num_vehicles

    # Define the speeds and locations for plotting
    speeds = ['6.6', '12', '19']  # Speeds to consider
    locations = ['Home', 'Home_Work']  # Locations for V2G scenarios
    location_labels = {'Home': '', 'Home_Work': ''}

    bar_height = 0.9  # Height for each grouped bar
    v2g_colors = {
        "Smart": '#FFD61F',
        'Home': '#3370FF',
        'Home_Work': '#000c29',
        'Degradation Smart': '#E69B0F',
        'Degradation Home': '#E00025',
        'Degradation Home_Work': '#250804'
    }

    y_positions = []  # Store y positions for each group
    scenario_labels = []  # Store labels for y positions
    group_positions = []  # To store the center of each group for annotation
    current_y = 2  # Current y position

    # Smart Charging Section: Add each speed separately with labels
    smart_bar_offsets = np.linspace(-bar_height / 20, bar_height / 20, len(speeds)) / 2

    # Calculate center of Smart Charging section for annotation
    group_positions.append(np.mean([2.5 + offset for offset in smart_bar_offsets]))
    first_line_position = current_y - 0.15
    second_line_position = current_y + 3.75
    # Assuming second_line_position is the y-coordinate of the horizontal line
    # Draw the horizontal line
    ax.axhline(second_line_position, color='black', linestyle='--', linewidth=2)
    ax.set_xlim(-lb, ub)
    # Add background colors

    smart_info = df1[(df1['Charge_Type'] == 'Smart') & (df1['V2G_Location'] == 'Home_Work')]

    # Helper function to plot side-by-side bars for V2G with locations
    def plot_v2g_section(data, smart, section_label):
        nonlocal current_y
        centers = []

        for speed in speeds:
            # Filter data for the current speed
            speed_data = data[data['Charging_Speed'] == speed]
            smart_data = smart[smart["Charging_Speed"] == speed]
            if not speed_data.empty:
                # Extract values for Home and Home_Work
                smart_value = smart_data[smart_data['V2G_Location'] == 'Home_Work']['Total_Benefit']
                home_value = speed_data[speed_data['V2G_Location'] == 'Home']['Total_Benefit']
                home_work_value = speed_data[speed_data['V2G_Location'] == 'Home_Work']['Total_Benefit']

                # Plot both components on the same stack (not cumulative)
                ax.barh(current_y, home_work_value, height=bar_height, color=v2g_colors['Home_Work'], hatch="*", label="Home_Work", edgecolor='white')
                ax.barh(current_y, home_value, height=bar_height, color=v2g_colors['Home'], hatch="X", label="Home")
                ax.barh(current_y, smart_value, height=bar_height, color=v2g_colors['Smart'], hatch="o", label="Smart")

                # Degradation stacking (Red and Pink, separate but aligned)
                degradation_home_work = -speed_data[speed_data['V2G_Location'] == 'Home_Work']['Degradation_Cost']
                degradation_home = -speed_data[speed_data['V2G_Location'] == 'Home']['Degradation_Cost']
                degradation_smart = -smart_data[smart_data['V2G_Location'] == 'Home_Work']['Degradation_Cost']

                ax.barh(current_y, degradation_home_work, height=bar_height, color=v2g_colors['Degradation Home_Work'], hatch="*", label="Degradation (Home_Work)", edgecolor='white')
                ax.barh(current_y, degradation_home, height=bar_height, color=v2g_colors['Degradation Home'], hatch="X", label="Degradation (Home)")
                ax.barh(current_y, degradation_smart, height=bar_height, color=v2g_colors['Degradation Smart'], hatch="o", label="Degradation (Smart)")


                # Add labels for speed
                y_positions.append(current_y)
                ax.set_ylim(ax.get_ylim()[0], current_y + 1)  # Add extra space at the top
                scenario_labels.append(f"{speed} kW")
                centers.append(current_y)

            current_y += 1.5  # Update the y position for the next speed
        return np.mean(centers)  # Return the center position

    # Plot V2G Actual section by all speeds and store its center
    v2g_actual_data = df1[(df1['Charge_Type'] == 'Bidirectional') & (df1['Scenario'] == 'No_change')]
    smart_info = df1[(df1['Charge_Type'] == 'Smart') & (df1['V2G_Location'] == 'Home_Work') & (df1['Scenario'] == 'No_change')]
    if not v2g_actual_data.empty:
        center = plot_v2g_section(v2g_actual_data, smart_info, "V2G / No Change in Plugging Behavior")
        group_positions.append(center)

    # Plot V2G Potential section by all speeds and store its center
    v2g_potential_data = df1[(df1['Charge_Type'] == 'Bidirectional') & (df1['Scenario'] == 'With_change')]
    smart_info = df1[(df1['Charge_Type'] == 'Smart') & (df1['V2G_Location'] == 'Home_Work') & (df1['Scenario'] == 'With_change')]
    if not v2g_potential_data.empty:
        center = plot_v2g_section(v2g_potential_data, smart_info, "V2G / Plugging-in When Parked")
        group_positions.append(center)

    ax.axhspan(ymin=ax.get_ylim()[0], ymax=second_line_position, xmin=0, xmax=1, color='#42be65', alpha=0.1, zorder=-1)
    ax.axhspan(ymin=second_line_position, ymax=ax.get_ylim()[1], xmin=0, xmax=1, color='#525252', alpha=0.1, zorder=-1)

    # Customize y-axis labels with all speeds retained
    ax.set_yticks(y_positions)
    ax.set_yticklabels(scenario_labels, fontsize=axis_text_size - 1)
    ax.axvline(0, color='grey', linewidth=1)
    ax.set_xlim(-lb, ub)
    ax.grid(True, axis='x', linestyle='--', alpha=0.7)

    # Add panel-specific title
    ax.set_title(title, fontsize=title_size)

    # Customize x-axis ticks and labels
    x_ticks = ax.get_xticks()

    # Replace the 0 value with the baseline value
    x_tick_labels = []
    baseline_found = False  # Flag to track when baseline is encountered

    for tick in x_ticks:
        if tick == 0:
            x_tick_labels.append(f'Baseline\n(${round((baseline_cost / num_vehicles), 2)})')
            baseline_found = True
        elif baseline_found and tick > 0:  # First tick on the right side of baseline
            x_tick_labels.append(f'')  # Replace with your desired text
            baseline_found = False  # Reset the flag so only the first positive tick is changed
        elif tick == x_ticks[0]:  # First tick (Loss)
            x_tick_labels.append("Loss")
        elif tick == x_ticks[-1]:  # Last tick (Savings)
            x_tick_labels.append("Savings")
        elif tick > 0:  # Last tick (Savings)
            x_tick_labels.append(f'$+{int(tick)}')
        else:  # Numbers in between
            x_tick_labels.append(f'${int(tick)}')

    # Update the x-axis ticks and labels
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_tick_labels)  # Ensure the labels are applied here

    # Apply rotation only to the numerical tick labels
    for tick, label in zip(ax.get_xticks(), ax.get_xticklabels()):
        label_text = label.get_text()
        if "$" in label_text and "Baseline" not in label_text:  # Rotate only numerical labels without "Baseline"
            label.set_rotation(90)  # Rotate numbers
            label.set_horizontalalignment('right')  # Align rotated text
            label.set_fontsize(axis_text_size - 2)  # Increase font size (adjust as needed)
        else:
            label.set_rotation(90)  # Keep "Loss," "Baseline," and "Savings" unrotated
            label.set_fontsize(axis_text_size)  # Increase font size (adjust as needed)

    # Add x-axis title only if not unified
    if not unified_x_title:
        ax.set_xlabel("", fontsize=title_size)

    # Add secondary y-axis for the last graph in the row
    if last_in_row:
        x_annotation_position = ub * 1.30
        group_labels = ['V2G\nNo Change in\nPlugging Behavior', 'V2G\nPlugging-in\nWhen Parked']
        group_offsets = [0.75, 4.75]  # Offset for each label, in y-axis units

        for pos, label, offset in zip(group_positions, group_labels, group_offsets):
            ax.text(
                x_annotation_position,
                pos + offset,  # Adjust the y-position with the offset
                label,
                ha='center',
                va='center',
                fontsize=title_size - 2,
                weight='bold',
                rotation=90
            )

    return ax


def plot_benefit_vs_degradation_panel2(data_list, num_vehicles, utilities, titles, lbs, ubs, y_title, x_title, figsize=(20, 8), title_size=18, axis_text_size=14):
    num_plots = len(data_list)  # Number of panels
    fig, axes = plt.subplots(1, num_plots, figsize=figsize, sharey=True)  # Create a single row of plots

    for i, ax in enumerate(axes):
        # Call the existing plot function, passing the specific axis
        plot_benefit_vs_degradation2(
            data_list[i],
            num_vehicles=num_vehicles,
            Utility=utilities[i],
            title=titles[i],
            lb=lbs[i],
            ub=ubs[i],
            title_size=title_size - 1,
            axis_text_size=axis_text_size,
            ax=ax,  # Pass the specific axis
            last_in_row=(i == len(axes) - 1),  # Add legend and secondary y-axis only for the last plot
            unified_x_title=(i != len(axes) - 1)  # Unified x-axis title only for the last plot
        )

    # Add shared y-axis and x-axis labels
    fig.text(0.1, 0.55, y_title, va='center', rotation='vertical', fontsize=title_size + 1)  # Y-axis
    fig.text(0.5, 0.04, x_title, ha='center', fontsize=title_size)  # X-axis

    # Adjust layout
    plt.tight_layout(rect=[0.04, 0.09, 1, 1])
    plt.savefig("PGE_all.png",  dpi=300, bbox_inches='tight')
    plt.show()


def plot_benefit_vs_degradation_panel_combined_fixed( data_list_sets, num_vehicles_list, utilities_list, titles_list, lbs_list, ubs_list,
    y_titles, x_title, figsize=(42, 50), title_size=38, axis_text_size=35, row_spacing=0.15, col_spacing=0.05):
    num_rows = len(data_list_sets)  # Number of rows (sets of data_list)
    num_cols = len(data_list_sets[0])  # Number of columns in each row

    # Create the figure
    fig = plt.figure(figsize=figsize)

    axes = []  # Track all axes
    for i, (data_list, num_vehicles, utilities, titles, lbs, ubs) in enumerate(
        zip(data_list_sets, num_vehicles_list, utilities_list, titles_list, lbs_list, ubs_list)
    ):
        for j in range(num_cols):
            # Calculate subplot positions
            left = j * (1 / num_cols) + col_spacing / 2  # Add column spacing
            bottom = 1 - (i + 1) * (1 / num_rows) - i * row_spacing  # Add spacing between rows
            width = 1 / num_cols - col_spacing  # Reduce width for column spacing
            height = 1 / num_rows - row_spacing  # Reduce height for row spacing

            # Add subplot manually
            ax = fig.add_axes([left, bottom, width, height])
            axes.append(ax)

            # Plot the data in the subplot
            plot_benefit_vs_degradation2(
                data_list[j],
                num_vehicles=num_vehicles,
                Utility=utilities[j],
                title=titles[j],
                lb=lbs[j],
                ub=ubs[j],
                title_size=title_size - 1,
                axis_text_size=axis_text_size,
                ax=ax,
                last_in_row=(j == num_cols - 1),  # Add secondary y-axis for last column
                unified_x_title=False
            )

            # Ensure y-axis ticks only for the first column
            if j == 0:
                ax.set_yticks(ax.get_yticks())  # Retain y-ticks for the first column
                ax.set_yticklabels(ax.get_yticklabels(), fontsize=axis_text_size - 1)
            else:
                ax.set_yticks([])  # Remove y-ticks for all other columns
                ax.set_yticklabels([])

    # Add y-axis titles with proper alignment
    for i, y_title in enumerate(y_titles):
        # Calculate y-center of the row
        row_bottom = 1 - (i + 1) * (1 / num_rows) - i * row_spacing
        row_top = row_bottom + (1 / num_rows) - row_spacing
        y_center = (row_bottom + row_top) / 2

        # Place the y-title dynamically aligned to the row
        fig.text(-0.07, y_center, y_title, va="center", rotation="vertical", fontsize=title_size + 1)

    # Add shared x-axis title for the entire figure
    fig.text(0.5, -0.19, x_title, ha="center", fontsize=title_size)

    # Add a single legend only for the last row (SMUD)
    legend_patches = [
        mpatches.Patch(facecolor="#FFD61F", hatch="o", label="V1G Profit"),
        mpatches.Patch(facecolor="#3370FF", hatch="X", label="V2G Profit (Home)"),
        mpatches.Patch(facecolor="#000c29", hatch="*", label="V2G Profit (Home & Work)", edgecolor="white"),
        mpatches.Patch(facecolor="#E69B0F", hatch="o", label="Degradation Cost (Smart)"),
        mpatches.Patch(facecolor="#E00025", hatch="X", label="Degradation Cost (Home)"),
        mpatches.Patch(facecolor="#250804", hatch="*", label="Degradation Cost (Home & Work)", edgecolor='white'),
    ]
    if i == num_rows - 1:  # Add legend only for the last row
        fig.legend(
            handles=legend_patches,
            loc="lower center",
            fontsize=axis_text_size - 1,
            ncol=3,
            bbox_to_anchor=(0.5, -0.25),
            frameon=False,
        )

    # Adjust layout for proper spacing
    plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)  # Add overall padding
    plt.savefig("plot_output_fixed_with_y_ticks_first_col.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_utility_panels(utility_data, utility_names, ylim_pairs, title_size=20, axis_text_size=18, figsize=(24, 28),
        y_title="Savings Compared to the Base Scenario ($/year)", x_title="Distance Driven (km)"):

    # Validate `ylim_pairs` and ensure they are tuples of two floats
    validated_ylim_pairs = [(float(pair[0]), float(pair[1])) for pair in ylim_pairs]

    num_rows = len(utility_data)
    fig, axes = plt.subplots(num_rows, 2, figsize=figsize, constrained_layout=False)
    color_map = {6.6: '#E00025', 12: '#3370FF', 19: '#FFD61F'}  # Define consistent colors

    for i, ((summary_actual, summary_potential), utility_name, (ylim_min, ylim_max)) in enumerate(
            zip(utility_data, utility_names, validated_ylim_pairs)):
        # Calculate common Y-axis ticks
        ylim = (ylim_min, ylim_max)
        y_ticks = list(range(int(ylim[0]), int(ylim[1]) + 1, int((ylim[1] - ylim[0]) / 5)))

        # Plot "No change in plugging behavior"
        ax_actual = axes[i, 0]
        plot_saving_ev_vs_distance(
            summary_actual,
            ax=ax_actual,
            add_actual_lines=False,
            add_potential_lines=False,
            ylim=ylim,
            text_size=axis_text_size,
            color_maps=color_map,
            title="No Change in Plugging Behavior" if i == 0 else "",
        )
        ax_actual.set_yticks(y_ticks)  # Explicitly set Y-axis ticks
        ax_actual.set_ylabel(utility_name, fontsize=axis_text_size, labelpad=15)

        # Plot "Plugging when parked"
        ax_potential = axes[i, 1]
        plot_saving_ev_vs_distance(
            summary_potential,
            ax=ax_potential,
            add_actual_lines=False,
            add_potential_lines=False,
            ylim=ylim,
            text_size=axis_text_size,
            color_maps=color_map,
            title="Plugging When Parked" if i == 0 else "",
        )
        ax_potential.set_yticks(y_ticks)  # Explicitly set the same Y-axis ticks
        ax_potential.set_yticklabels([])  # Remove tick labels for the right column
        ax_potential.set_ylabel("")  # Remove Y-axis label for the right column

        # Set X-axis labels only for the bottom row
        if i == num_rows - 1:
            ax_actual.set_xlabel(x_title, fontsize=axis_text_size)
            ax_potential.set_xlabel(x_title, fontsize=axis_text_size)
        else:
            ax_actual.set_xlabel("")
            ax_potential.set_xlabel("")

        # Remove X-axis tick labels for all but the last row
        if i < num_rows - 1:
            ax_actual.set_xticklabels([])
            ax_potential.set_xticklabels([])

    # Add a general Y-axis title
    fig.text(-0.02, 0.6, y_title, va='center', rotation='vertical', fontsize=title_size)

    # Add Legends (same as before)
    marker_map = {'Bidirectional': 'o', 'Smart': 's'}
    size_map = {66: 150, 70: 200, 75: 250, 80: 300, 85: 350, 100: 400}
    type_legend_elements = [
        Line2D([0], [0], marker=marker_map['Bidirectional'], color='w', label='V2G', markerfacecolor='gray', markersize=20),
        Line2D([0], [0], marker=marker_map['Smart'], color='w', label='V1G', markerfacecolor='gray', markersize=20)
    ]
    speed_legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='6.6 kW', markerfacecolor=color_map[6.6], markersize=20),
        Line2D([0], [0], marker='o', color='w', label='12 kW', markerfacecolor=color_map[12], markersize=20),
        Line2D([0], [0], marker='o', color='w', label='19 kW', markerfacecolor=color_map[19], markersize=20),
    ]
    size_legend_elements = [
        Line2D([0], [0], marker='o', color='w', label=f'{key} kWh', markerfacecolor='gray', markersize=value / 15)
        for key, value in size_map.items()
    ]
    legend1 = fig.legend(
        handles=type_legend_elements,
        title="Charging Type",
        loc='lower center',
        bbox_to_anchor=(0.2, 0.12),
        fontsize=axis_text_size,
        title_fontsize=axis_text_size,
        frameon=True,
        ncol=2
    )
    legend2 = fig.legend(
        handles=speed_legend_elements,
        title="Charging Speed",
        loc='lower center',
        bbox_to_anchor=(0.4, 0.08),
        fontsize=axis_text_size,
        title_fontsize=axis_text_size,
        frameon=True,
        ncol=1
    )
    legend3 = fig.legend(
        handles=size_legend_elements,
        title="Battery Size",
        loc='lower center',
        bbox_to_anchor=(0.7, 0.1),
        fontsize=axis_text_size,
        title_fontsize=axis_text_size,
        frameon=True,
        ncol=3
    )
    # Adjust layout for better spacing
    plt.subplots_adjust(wspace=0.1, hspace=0.4, left=0.1, right=0.9, top=0.9, bottom=0.25)
    plt.savefig("utility_panels_aligned_y_ticks.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_2x2_utility_panels_with_adjustments(grouped_data, utilities, figsize=(16, 14), text_size=14):
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    # Flatten axes for easier indexing
    axes = axes.flatten()
    # Labels for the subplots
    subplot_labels = ['a-', 'b-', 'c-', 'd-']

    for i, (data, utility, ax) in enumerate(zip(grouped_data, utilities, axes)):
        # Check if the utility is PGE or SDGE to disable the secondary y-axis
        enable_secondary_yaxis = utility not in ["PGE", "SDGE"]

        # Draw the box plot for each subplot
        draw_box_plot(data, text_size=text_size, ax=ax, enable_secondary_yaxis=enable_secondary_yaxis)

        # Move the title to below the x-axis title
        ax.set_title("", fontsize=text_size)  # Remove the top title
        ax.annotate(
            f"{subplot_labels[i]} {utility}",
            xy=(0.5, -0.25),  # Positioned below the x-axis title
            xycoords="axes fraction",
            ha="center",
            fontsize=text_size + 4,
        )

        # Adjust y-axis tick labels
        if i % 2 == 0:  # First column
            sec_ax = ax.secondary_yaxis('right')
            sec_ax.remove()  # Completely remove the secondary y-axis
        else:  # Second column
            ax.set_yticklabels([])  # Remove primary y-axis tick labels

    # Add a single legend for the entire figure
    legend_patches = [
        mpatches.Patch(facecolor="#FFD61F", label="19 kW"),
        mpatches.Patch(facecolor="#3370FF", label="12 kW"),
        mpatches.Patch(facecolor="#E00025", label="6.6 kW"),
    ]
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        fontsize=text_size + 4,
        title="Charging Speed",  # Add the legend title
        title_fontsize=text_size + 5,  # Set the title font size
        ncol=3,  # Arrange legend items in a single row
        bbox_to_anchor=(0.5, 0.05),  # Position legend below all subplots
        frameon=False,
        markerscale=4,  # Scale the size of legend markers
    )

    # Adjust layout for proper spacing
    plt.tight_layout(h_pad=3)  # Increase spacing between rows
    plt.subplots_adjust(bottom=0.2, top=0.99)  # Ensure enough space for the legend
    # Save the updated figure
    plt.savefig("2x2_panel_corrected_single_legend.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_box_by_tariff_panel(data_list, utility_names, figtitle="Annual CO$_2$ Emissions Reduction per Vehicle", Rate_type="RT_Rate", fz=16, figsize=(14, 10)):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import matplotlib.patches as mpatches

    # Define consistent colors for the charging speeds
    charging_speed_colors = {6.6: '#FFD61F', 12: '#3370FF', 19: '#E00025'}

    # Create the 2x2 panel
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()  # Flatten for easy indexing
    scenario_order = [
        "V2G | Plugging in when parked",
        "V2G | No change in plugging behavior",
        "V1G | Plugging in when parked",
        "V1G | No change in plugging behavior"
    ]

    # Subplot labels
    subplot_labels = ['a-', 'b-', 'c-', 'd-']

    # Loop through each pair of DataFrames and plot
    for i, ((df1, df2), utility_name) in enumerate(zip(data_list, utility_names)):
        ax = axes[i]

        # Combine and filter the DataFrames
        df1["Charging_Behavior"] = "No change in plugging behavior"
        df2["Charging_Behavior"] = "Plugging in when parked"
        df_combined = pd.concat([df1, df2], axis=0, ignore_index=True)
        df_filtered = df_combined[df_combined["Rate_Type"] == Rate_type].copy()
        df_filtered["Charge_Type"] = df_filtered["Charge_Type"].replace({
            "Smart": "V1G",
            "Bidirectional": "V2G"
        })
        df_filtered['Scenario'] = df_filtered["Charge_Type"] + ' | ' + df_filtered["Charging_Behavior"]

        # Create the box plot
        sns.boxplot(
            data=df_filtered,
            y="Scenario",
            x="diff",
            hue="Charging_Speed",
            palette=charging_speed_colors,
            dodge=True,
            order=scenario_order,
            ax=ax,
            hue_order=[19, 12, 6.6],
        )

        # Remove individual x-axis labels
        ax.set_xlabel("")  # Set x-axis label to an empty string

        # Add custom labels for the primary y-axis (left side)
        if utility_name in ["PGE", "SDGE"]:  # Only adjust for PGE and SDGE
            y_labels_left = ["Plugging in\nwhen\nparked", "No change in\nplugging\nbehavior"] * 2
            ax.set_yticks([0, 1, 2, 3])
            ax.set_yticklabels(y_labels_left, ha='left', fontsize=fz)
            ax.tick_params(axis='y', pad=120)  # Increase padding to push labels further left
            ax.set_ylabel("", fontsize=fz)
        else:  # Remove primary y-axis tick labels for SCE and SMUD
            ax.set_yticklabels([])
            ax.set_ylabel("")

        # Add secondary y-axis for the right side with custom labels
        if utility_name in ["SCE", "SMUD"]:
            ax2 = ax.twinx()  # Create a secondary y-axis
            y_labels_right = ["V2G", "V2G", "V1G", "V1G"]
            ax2.set_ylim(ax.get_ylim())  # Sync limits
            ax2.set_yticks([0, 1, 2, 3])
            ax2.set_yticklabels(y_labels_right, rotation=90, va='center', ha="left", fontsize=fz + 2)
        else:  # Remove secondary y-axis for PGE and SDGE
            ax2 = ax.twinx()
            ax2.set_yticklabels([])  # Remove secondary y-axis labels
            ax2.set_ylabel("")

        # Add dashed lines to separate groups
        ax.axhline(1.5, color='grey', linestyle='--', linewidth=1)
        ax.axhline(2.5, color='grey', linestyle='--', linewidth=1)

        # Add subplot label and utility name below the plot
        ax.text(
            0.5, -0.15, f"{subplot_labels[i]} {utility_name}",
            transform=ax.transAxes,
            ha="center", va="center", fontsize=fz + 2, fontweight="bold"
        )

        # Remove individual legends
        ax.legend_.remove()

    # Add shared x-axis and y-axis labels
    fig.text(0.5, 0.02, "Annual CO$_2$ Emissions Reduction per Vehicle (tonnes)", ha='center', fontsize=fz + 2)

    # Add a shared legend
    legend_labels = [f"{speed} kW" for speed in charging_speed_colors.keys()]
    handles = [mpatches.Patch(color=color, label=label) for color, label in zip(charging_speed_colors.values(), legend_labels)]
    fig.legend(
        handles=handles,
        title="Charging Speed",
        loc="lower center",
        bbox_to_anchor=(0.5, -0.08),
        fontsize=fz,
        title_fontsize=fz + 2,
        ncol=3,
    )
    ax.set_xlabel("")  # Set x-axis label to an empty string
    # Adjust layout for spacing
    plt.tight_layout(rect=[0, 0.05, 1, 1])  # Leave space for the legend
    plt.savefig(f'{figtitle}.png', bbox_inches='tight', dpi=300)
    plt.show()


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

    return all_hourly_charging_data_grouped, summary, summary_actual, summary_potential

