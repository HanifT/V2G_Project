# %%
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
# %%
# Directory containing the Excel files
directory = "/Users/haniftayarani/V2G_Project/Travel_data"
# List of Excel file names
csv_files1 = ["bev_trips_full.csv"]
csv_files2 = ["survey_trip.csv"]
csv_files2_1 = ["survey_location.csv"]
csv_files3 = ["tripv2pub.csv"]


# Function to read CSV files and concatenate them into a single DataFrame
def read_and_concat_csv(directory, csv_files, **kwargs):
    data_frames = []
    for file in csv_files:
        file_path = os.path.join(directory, file)
        data_frame = pd.read_csv(file_path, **kwargs)
        data_frames.append(data_frame)
    concatenated_chts = pd.concat(data_frames, ignore_index=True)
    return concatenated_chts

# Specify dtype for column 3 if known, otherwise leave as None
dtype_spec = {3: str}

# Read the CSV files into single DataFrames
evmt_data = read_and_concat_csv(directory, csv_files1, dtype=dtype_spec, low_memory=False)
chts = read_and_concat_csv(directory, csv_files2, dtype=dtype_spec, low_memory=False)
chts_loc = read_and_concat_csv(directory, csv_files2_1, dtype=dtype_spec, low_memory=False)
nhts = read_and_concat_csv(directory, csv_files3, dtype=dtype_spec, low_memory=False)
chts = pd.merge(chts,chts_loc[["sampno", "loctype","locno"]], how="left", on=("sampno", "locno"))
chts = chts[~chts["loctype"].isna()]
chts["loctype"] = chts["loctype"].astype(int)
chts.loc[chts["loctype"] == 1, "location"] = "Home"
chts.loc[chts["loctype"] == 2, "location"] = "Work"
chts.loc[chts["loctype"] == 3, "location"] = "School"
chts.loc[chts["loctype"] == 4, "location"] = "Other"


def convert_and_trim(chts, column_name, new_column_name):
    # Convert the column to integers
    chts[column_name] = pd.to_numeric(chts[column_name], errors='coerce').fillna(0).astype(int)

    # Remove the first two digits from the right
    chts[new_column_name] = chts[column_name] // 100

    return chts


# Extract the hour and store it in new columns
evmt_data['Hour_start'] = pd.to_datetime(evmt_data['start_time_ (local)']).dt.hour
evmt_data['Hour_end'] = pd.to_datetime(evmt_data['end_time_ (local)']).dt.hour

# Convert and trim the 'original_column' and store the result in 'new_column'
chts = convert_and_trim(chts, 'strttime', 'Hour_start')
nhts = convert_and_trim(nhts, 'STRTTIME', 'Hour_start')

chts = convert_and_trim(chts, 'endtime', 'Hour_end')
nhts = convert_and_trim(nhts, 'ENDTIME', 'Hour_end')

evmt_data['distance_mile'] = evmt_data['distance']
chts['distance_mile'] = chts['distance_mi']
chts = chts[(chts['distance_mile'] > 1) & (chts['distance_mile'] < 500)]
nhts['distance_mile'] = nhts['TRPMILES']
nhts = nhts[(nhts['distance_mile'] > 1) & (nhts['distance_mile'] < 500)]
chts_short = chts[chts["distance_mile"] < 200]
nhts_short = nhts[nhts["distance_mile"] < 200]


def plot_distribution(dataframe1, dataframe2, dataframe3, dataframe4, dataframe5, column_name, chts1_name='DataFrame 1', chts2_name='DataFrame 2', chts3_name='DataFrame 3',
                      chts4_name='DataFrame 4', chts5_name='DataFrame 5', xlim=None, log_transform=False):

    dataframes = [dataframe1, dataframe2, dataframe3, dataframe4, dataframe5]
    chts_names = [chts1_name, chts2_name, chts3_name, chts4_name, chts5_name]
    colors = ['skyblue', 'salmon', 'red', 'orange', 'lightgreen']
    linestyles = ['-', '--', '-.', ':', '-']

    for chts, name, color, ls in zip(dataframes, chts_names, colors, linestyles):
        data = chts[chts[column_name] >= 1][column_name]  # Filter out values below 1

        if log_transform:
            data = np.log1p(data)  # Optional log transformation for skewed data

        sns.kdeplot(data=data, label=name, color=color, linestyle=ls, clip=(1, None))  # Clip at 1
    # Add labels and title
    plt.xlabel(column_name)
    plt.ylabel('Density')
    plt.title('Distribution of ' + column_name)

    # Set x-axis limits if provided
    # Set x-axis limits if provided
    if xlim is not None:
        plt.xlim(xlim)
    else:
        plt.xlim(0.5, None)  # Set x-axis to start at 0

    # Add legend
    plt.legend()
    plt.grid()
    # Show plot
    plt.show()


# Example usage with log transformation for 'distance_mile'
plot_distribution(nhts, chts, nhts_short, chts_short, evmt_data, 'distance_mile', chts1_name='NHTS', chts2_name='CHTS', chts3_name="NHTS_Short", chts4_name="CHTS_Short", chts5_name='EVM Data', xlim=(0.1, 80))
plot_distribution(nhts, chts, nhts_short, chts_short, evmt_data, 'distance_mile', chts1_name='NHTS', chts2_name='CHTS', chts3_name="NHTS_Short", chts4_name="CHTS_Short", chts5_name='EVM Data', xlim=(0.1, 10))

plot_distribution(nhts, chts, nhts_short, chts_short, evmt_data, 'Hour_start', chts1_name='NHTS', chts2_name='CHTS', chts3_name="NHTS_Short", chts4_name="CHTS_Short", chts5_name='EVM Data', xlim=(0, 24))
plot_distribution(nhts, chts, nhts_short, chts_short, evmt_data, 'Hour_end', chts1_name='NHTS', chts2_name='CHTS', chts3_name="NHTS_Short", chts4_name="CHTS_Short", chts5_name='EVM Data', xlim=(0, 24))
# %% check

trip_data = pd.read_csv("data.csv")
vehicle_list = ["P_1087", "P_1091", "P_1092", "P_1093", "P_1094", "P_1098", "P_1100", 'P_1109', 'P_1111', "P_1112", "P_1123", "P_1125", "P_1125a", "P_1127",
                'P_1131', 'P_1132', 'P_1135', 'P_1137', "P_1141", "P_1143", 'P_1217', 'P_1253', 'P_1257', 'P_1260', 'P_1267', 'P_1271', 'P_1272', 'P_1279',
                'P_1280', 'P_1281', 'P_1285', 'P_1288', 'P_1294', 'P_1295', 'P_1296', 'P_1304', 'P_1307', "P_1357", "P_1367", 'P_1375', 'P_1353',
                'P_1368', 'P_1371', "P_1376", 'P_1393', "P_1414", 'P_1419', 'P_1421', 'P_1422', 'P_1423', 'P_1424', 'P_1427', "P_1435"]

trip_data_check = trip_data[trip_data["vehicle_name"].isin(vehicle_list)]
trip_data_check = trip_data_check.groupby("vehicle_name")["distance"].sum().reset_index(drop=False)

# %%
import pandas as pd


trips_df = chts_short
def locations_mapping(trips_df):
    # Create unique time series
    time_series_data = []
    for _, row in trips_df.iterrows():
        time_series_index = pd.RangeIndex(row['Hour_start'], row['Hour_end'] + 1, name='hour')
        time_series = pd.DataFrame({'household_id': row['sampno'],
                                   'person_id': row['perno'],
                                   'location': row['location']},
                                  index=time_series_index)
        time_series_data.append(time_series)

    time_series_df = pd.concat(time_series_data)

    # Aggregate data based on time, household and person
    time_series_df = time_series_df.reset_index().groupby(['hour', 'household_id', 'person_id'])['location'].last().reset_index()

    # Fill missing hours
    all_hours = pd.DataFrame({'hour': range(time_series_df['hour'].min(), time_series_df['hour'].max() + 1)})
    time_series_df = time_series_df.merge(all_hours, on='hour', how='outer')
    time_series_df = time_series_df.sort_values(['household_id', 'person_id', 'hour'])

    # Fill forward missing values
    time_series_df['location'] = time_series_df.groupby(['household_id', 'person_id'])['location'].fillna(method='ffill')

    # Fill remaining missing values with "home"
    time_series_df['location'] = time_series_df['location'].fillna('home')
    return time_series_df


def locations_mappingn(trips_df):
    # Create unique time series
    time_series_data = []
    for _, row in trips_df.iterrows():
        time_series_index = pd.RangeIndex(row['Hour_start'], row['Hour_end'] + 1, name='hour')
        time_series = pd.DataFrame({'household_id': row['Household'],
                                    'location': row['destination_label']},
                                   index=time_series_index)
        time_series_data.append(time_series)

    time_series_df = pd.concat(time_series_data)

    # Aggregate data based on time, household and person
    time_series_df = time_series_df.reset_index().groupby(['hour', 'household_id'])['location'].last().reset_index()

    # Fill missing hours
    all_hours = pd.DataFrame({'hour': range(time_series_df['hour'].min(), time_series_df['hour'].max() + 1)})
    time_series_df = time_series_df.merge(all_hours, on='hour', how='outer')
    time_series_df = time_series_df.sort_values(['household_id', 'hour'])

    # Fill forward missing values
    time_series_df['location'] = time_series_df.groupby(['household_id'])['location'].fillna(method='ffill')

    # Fill remaining missing values with "home"
    time_series_df['location'] = time_series_df['location'].fillna('home')
    return time_series_df

time_series_chts_short = locations_mapping(chts_short)
time_series_chts = locations_mapping(chts)
time_series_evmt_data = locations_mappingn(evmt_data)


def draw_loc(df1,bar_width):
    # Group data by hour and location, then calculate the frequency of each location at each hour
    location_distribution = df1.groupby(['hour', 'location']).size().unstack(fill_value=0)

    # Normalize counts to get probability distribution
    location_distribution_prob = location_distribution.div(location_distribution.sum(axis=1), axis=0)

    # Group data by hour and location, then calculate the frequency of each location at each hour
    location_distribution = df1.groupby(['hour', 'location']).size().unstack(fill_value=0)

    # Plotting
    location_distribution.plot(kind='bar', stacked=True, figsize=(10, 6), width=bar_width)

    # Add labels and title
    plt.xlabel('Hour')
    plt.ylabel('Frequency')
    plt.title('Distribution of Locations Over Hours')

    # Show plot
    plt.legend(title='Location')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def draw_loc1(df1,bar_width):
    # Group data by hour and location, then calculate the frequency of each location at each hour
    location_distribution = df1.groupby(['hour', 'location']).size().unstack(fill_value=0)

    # Normalize counts to get probability distribution
    location_distribution_prob = location_distribution.div(location_distribution.sum(axis=1), axis=0)

    # Plotting
    location_distribution_prob.plot(kind='bar', stacked=True, figsize=(10, 6), width=bar_width)

    # Add labels and title
    plt.xlabel('Hour')
    plt.ylabel('Probability')
    plt.title('Probability Distribution of Locations Over Hours')

    # Show plot
    plt.legend(title='Location')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def draw_loc1(df, bar_width=0.8):
    # Define colors for each location
    location_colors = {
        'Home': 'skyblue',
        'Work': 'salmon',
        'Other': 'lightgreen',
        'School': 'orange'
    }

    # Group data by hour and location, then calculate the frequency of each location at each hour
    location_distribution = df.groupby(['hour', 'location']).size().unstack(fill_value=0)

    # Normalize counts to get probability distribution
    location_distribution_prob = location_distribution.div(location_distribution.sum(axis=1), axis=0)

    # Plotting
    ax = location_distribution_prob.plot(kind='bar', stacked=True, figsize=(10, 6), width=bar_width, color=[location_colors.get(col, 'grey') for col in location_distribution_prob.columns])

    # Add labels and title
    ax.set_xlabel('Hour')
    ax.set_ylabel('Probability')
    ax.set_title('Probability Distribution of Locations Over Hours')

    # Show plot
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels, title='Location')
    ax.set_xticklabels(location_distribution_prob.index, rotation=45)
    plt.tight_layout()
    plt.show()


# Example usage for dataframe1
draw_loc1(time_series_chts_short, bar_width=1)

# Example usage for dataframe2
draw_loc1(time_series_evmt_data, bar_width=1)

draw_loc(time_series_chts_short,1)
draw_loc(time_series_evmt_data,1)


draw_loc1(time_series_chts_short, 1)
draw_loc1(time_series_evmt_data,1)
