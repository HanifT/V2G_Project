import os
import zipfile
import pandas as pd
import shutil
import matplotlib.pyplot as plt
import numpy as np


def extract_csv_from_zip(zip_folder, output_folder):
    for zip_filename in os.listdir(zip_folder):
        if zip_filename.endswith('.zip'):
            with zipfile.ZipFile(os.path.join(zip_folder, zip_filename), 'r') as zip_ref:
                zip_ref.extractall(output_folder)


zip_folder = 'G:\\My Drive\\Chapter2\\Load Data 2022'  # Replace with your actual folder containing zip files
temp_folder = 'G:\\My Drive\\Chapter2\\Load Data 2022\\temp'  # Create a temporary folder to extract CSV files
os.makedirs(temp_folder, exist_ok=True)
dataframes = []
extract_csv_from_zip(zip_folder, temp_folder)
for csv_filename in os.listdir(temp_folder):
    if csv_filename.endswith('.csv'):
        df = pd.read_csv(os.path.join(temp_folder, csv_filename))
        dataframes.append(df)

# Concatenate the DataFrames
combined_df = pd.concat(dataframes, axis=0)

# Sort based on the first row
combined_df = combined_df.sort_values(by=combined_df.columns[0])

# Cleanup: Remove the temporary folder and its contents
shutil.rmtree(temp_folder)

combined_df = combined_df.loc[combined_df["TAC_AREA_NAME"] == "CA ISO-TAC"]

combined_df['INTERVALSTARTTIME_GMT'] = pd.to_datetime(combined_df['INTERVALSTARTTIME_GMT'], utc=True)
combined_df['INTERVALENDTIME_GMT'] = pd.to_datetime(combined_df['INTERVALENDTIME_GMT'], utc=True)

# Set the timezone to 'America/Los_Angeles' (Pacific Time Zone)
combined_df['INTERVALSTARTTIME_GMT'] = combined_df['INTERVALSTARTTIME_GMT'].dt.tz_convert('America/Los_Angeles')
combined_df['INTERVALENDTIME_GMT'] = combined_df['INTERVALENDTIME_GMT'].dt.tz_convert('America/Los_Angeles')

# Split the 'INTERVALSTARTTIME_GMT' column into separate columns for year, month, day, and hour
combined_df['Year'] = combined_df['INTERVALSTARTTIME_GMT'].dt.year
combined_df['Month'] = combined_df['INTERVALSTARTTIME_GMT'].dt.month
combined_df['Day'] = combined_df['INTERVALSTARTTIME_GMT'].dt.day
combined_df['Hour'] = combined_df['INTERVALSTARTTIME_GMT'].dt.hour
combined_df['DayOfYear'] = combined_df['INTERVALSTARTTIME_GMT'].dt.dayofyear
combined_df['HourOfYear'] = (combined_df['INTERVALSTARTTIME_GMT'].dt.dayofyear * 24-23) + combined_df['Hour']


combined_df = combined_df.reset_index(drop=True)

# Pivot the DataFrame to have hours as columns and days as rows

pivoted_df = combined_df.pivot_table(index='DayOfYear', columns='Hour', values='MW')


pivoted_df_winter =pivoted_df[1:90].T
pivoted_df_spring =pivoted_df[90:180].T
pivoted_df_summer =pivoted_df[180:270].T
pivoted_df_fall = pivoted_df[270:360].T


df_fall = pivoted_df_fall
df_winter = pivoted_df_winter
df_spring = pivoted_df_spring
df_summer = pivoted_df_summer
# Calculate the mean and standard deviation for each hour
hourly_mean_fall = df_fall.mean(axis=1)
hourly_std_fall = df_fall.std(axis=1)
hourly_mean_winter = df_winter.mean(axis=1)
hourly_std_winter = df_winter.std(axis=1)
hourly_mean_spring = df_spring.mean(axis=1)
hourly_std_spring = df_spring.std(axis=1)
hourly_mean_summer = df_summer.mean(axis=1)
hourly_std_summer = df_summer.std(axis=1)
# Create an x-axis for the hours of the day
hours_of_day = np.arange(24)

# Create the figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the hourly means with error bars
ax.errorbar(hours_of_day, hourly_mean_fall, yerr=hourly_std_fall, fmt='o-', capsize=5, label="Oct-Dec")
ax.errorbar(hours_of_day, hourly_mean_winter, yerr=hourly_std_winter, fmt='o-', capsize=5, label="Jan-Mar")
ax.errorbar(hours_of_day, hourly_mean_spring, yerr=hourly_std_spring, fmt='o-', capsize=5, label="Apr-Jun")
ax.errorbar(hours_of_day, hourly_mean_summer, yerr=hourly_std_summer, fmt='o-', capsize=5, label="Jul-Sep")
# Set labels and title
ax.set_xlabel('Hour of the Day', fontsize=16)
ax.set_ylabel('Average Hourly Power Consumption (MW)', fontsize=16)
ax.set_title('Average Hourly Values with Error Bars', fontsize=16)
ax.legend()
ax.set_xticks(np.arange(0, 24, 1))
plt.tick_params(labelsize=14)
# Show the plot
plt.grid(True)
plt.show()


df_fall.columns = range(len(df_fall.columns))
df_winter.columns = range(len(df_winter.columns))
df_spring.columns = range(len(df_spring.columns))
df_summer.columns = range(len(df_summer.columns))

df_year = pd.concat([df_winter, df_spring, df_summer, df_fall], axis=1)
df_year.columns = range(len(df_year.columns))
df_year = df_year.T
daily_mean_year = df_year.sum(axis=1)
hourly_std_year = df_year.std(axis=1)
days_of_year = np.arange(359)
fig, ax = plt.subplots(figsize=(10, 6))
ax.errorbar(days_of_year, daily_mean_year, yerr=hourly_std_year, fmt='o-', capsize=5)
ax.set_xlabel('Day of the Year', fontsize=16)
ax.set_ylabel('Average Daily Power Consumption (MW)', fontsize=16)
ax.set_title('Average Daily Values with Error Bars', fontsize=16)
ax.set_xticks(np.arange(0, 361, 30))
plt.tick_params(labelsize=14)
# Show the plot
plt.grid(True)
plt.show()

#
# # Create a figure and axis for the plot
# fig, ax = plt.subplots(figsize=(10, 6))
#
# # Create a scatter plot with automatic legend generation
# for destination in charging_sessions['destination_label'].unique():
#     for charging_rate in charging_sessions['energy[charge_type][type]'].unique():
#         subset = charging_sessions[(charging_sessions['destination_label'] == destination) & (charging_sessions['energy[charge_type][type]'] == charging_rate)]
#         if not subset.empty:
#             label = f'{destination}, {charging_rate}'
#             ax.scatter(subset['duration_charging']/(60*24), subset['battery[soc][start][charging]'], label=label, alpha=0.7)
#
# # Add labels and legend
# ax.set_xlabel('Charging Duration (hour)', fontsize=16)
# ax.set_ylabel('SOC at the Beginning of Charging Session (%)', fontsize=16)
#
# # Set a grid for better readability
# ax.grid(True, linestyle='--', alpha=0.6)
#
# # Use Seaborn's style to improve aesthetics
# sns.set_style("whitegrid")
#
# # Automatically generate the legend based on existing labels
# ax.legend(loc='upper right')
#
# # Set the title for the plot
# plt.title("Charging Sessions Analysis")
# plt.tick_params(labelsize=14)
# # Ensure a tight layout
# plt.tight_layout()
#
# # Show the plot
# plt.show()
#
#
# # Create a figure and axis for the plot
# fig, ax = plt.subplots(figsize=(10, 6))
#
# # Create a scatter plot with automatic legend generation
# for destination in charging_sessions['destination_label'].unique():
#     for charging_rate in charging_sessions['energy[charge_type][type]'].unique():
#         subset = charging_sessions[(charging_sessions['destination_label'] == destination) & (charging_sessions['energy[charge_type][type]'] == charging_rate)]
#         if not subset.empty:
#             label = f'{destination}, {charging_rate}'
#             ax.scatter(subset['duration_charging']/(60*24), subset['battery[soc][end][charging]'], label=label, alpha=0.7)
#
# # Add labels and legend
# ax.set_xlabel('Charging Duration (hour)', fontsize=16)
# ax.set_ylabel('SOC at the End of Charging Session (%)', fontsize=16)
#
#
# # Set a grid for better readability
# ax.grid(True, linestyle='--', alpha=0.6)
#
# # Use Seaborn's style to improve aesthetics
# sns.set_style("whitegrid")
#
# # Automatically generate the legend based on existing labels
# ax.legend(loc='upper right')
#
# # Set the title for the plot
# plt.title("Charging Sessions Analysis")
# plt.tick_params(labelsize=14)
# # Ensure a tight layout
# plt.tight_layout()
#
# # Show the plot
# plt.show()
#
#
# charging_sessions = full_events.loc[(full_events["energy[charge_type][type]"] != "NA") & ((full_events["Make"] == "Tesla") | (full_events["Make"] == "Chevrolet"))]
# charging_sessions = charging_sessions.loc[charging_sessions["total_energy"] > 2]
# charging_sessions = charging_sessions.loc[~((charging_sessions["total_energy"] < 10) & (charging_sessions["duration_charging"] > 10*3600))]
# charging_sessions = charging_sessions.loc[charging_sessions["duration_charging"] < 2*3600]
#
# # Set the timezone to 'America/Los_Angeles' (Pacific Time Zone)
# charging_sessions['start_time_local'] = pd.to_datetime(charging_sessions['start_time_local'], utc=True)
# charging_sessions['start_time_local'] = charging_sessions['start_time_local'].dt.tz_convert('America/Los_Angeles')
#
#
# # Split the 'INTERVALSTARTTIME_GMT' column into separate columns for year, month, day, and hour
# charging_sessions['start_hour_charging'] = charging_sessions['start_time_local'].dt.hour
#
#
# # Create a figure and axis for the plot
# fig, ax = plt.subplots(figsize=(10, 6))
#
# # Create a scatter plot with automatic legend generation
# for destination in charging_sessions['destination_label'].unique():
#     for charging_rate in charging_sessions['energy[charge_type][type]'].unique():
#         subset = charging_sessions[(charging_sessions['destination_label'] == destination) & (charging_sessions['energy[charge_type][type]'] == charging_rate)]
#         if not subset.empty:
#             label = f'{destination}, {charging_rate}'
#             ax.scatter(subset['duration_charging']/3600, subset['start_hour_charging'], label=label, alpha=0.7)
#
# # Add labels and legend
# ax.set_xlabel('Charging Duration (hour)', fontsize=16)
# ax.set_ylabel('Start Time of Charging (hour)', fontsize=16)
#
# # Set a grid for better readability
# ax.grid(True, linestyle='--', alpha=0.6)
#
# # Use Seaborn's style to improve aesthetics
# sns.set_style("whitegrid")
#
# # Automatically generate the legend based on existing labels
# ax.legend(loc='upper right')
#
# # Set the title for the plot
# plt.title("Charging Sessions Analysis")
# plt.tick_params(labelsize=14)
# # Ensure a tight layout
# plt.tight_layout()
#
# # Show the plot
# plt.show()
#
#
#
# # Create a figure and axis for the plot
# fig, ax = plt.subplots(figsize=(10, 6))
# # charging_sessions = charging_sessions.loc[(charging_sessions["parking_duration"] > 0) & (charging_sessions["parking_duration"] <  )]
# charging_sessions = charging_sessions.loc[(charging_sessions["parking_duration"] > 0) & (charging_sessions["parking_duration"] < 14400)]
#
# # Create a scatter plot with automatic legend generation
# for destination in charging_sessions['destination_label'].unique():
#     for charging_rate in charging_sessions['energy[charge_type][type]'].unique():
#         subset = charging_sessions[(charging_sessions['destination_label'] == destination) & (charging_sessions['energy[charge_type][type]'] == charging_rate)]
#         if not subset.empty:
#             label = f'{destination}, {charging_rate}'
#             ax.scatter(subset['duration_charging']/(60*24), subset['parking_duration']/(60*24), label=label, alpha=0.7)
#
# # Add labels and legend
# ax.set_xlabel('Charging Duration (Day)', fontsize=16)
# ax.set_ylabel('Parking Duration (Day)', fontsize=16)
#
# # Set a grid for better readability
# ax.grid(True, linestyle='--', alpha=0.6)
#
# # Use Seaborn's style to improve aesthetics
# sns.set_style("whitegrid")
#
# # Automatically generate the legend based on existing labels
# ax.legend(loc='upper right')
#
# # Set the title for the plot
# plt.title("Charging Sessions Analysis")
# plt.tick_params(labelsize=14)
# # Ensure a tight layout
# plt.tight_layout()
#
# # Show the plot
# plt.show()
#
