# %%
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np

# %% Reading Files

# Directory containing the Excel files
directory = "/Users/haniftayarani/V2G_Project/Batt_cap"

# List of Excel file names
excel_files1 = ["Batt_cat.xlsx"]
excel_files2 = ["Batt_size.xlsx"]
excel_files3 = ["TeslaBatteryDatabase.xlsx"]


def read_excel_files(directory, excel_files):
    all_data = pd.DataFrame()  # Create an empty DataFrame to store all data

    for excel_file_name in excel_files:
        excel_file_path = os.path.join(directory, excel_file_name)
        print(f"Reading Excel file '{excel_file_path}'...")

        with pd.ExcelFile(excel_file_path) as xls:
            for sheet_name in xls.sheet_names:
                df = pd.read_excel(xls, sheet_name=sheet_name)

                # Handle single-column DataFrames
                if df.shape[1] == 1:
                    df = df.set_index(df.columns[0])  # Use the column as index

                # Concatenate with the existing data
                all_data = pd.concat([all_data, df], axis=0, ignore_index=True)

    return all_data  # Return the combined DataFrame


bd_dataframes = read_excel_files(directory, excel_files1)
bs_dataframes = read_excel_files(directory, excel_files2)

Battery_data = pd.merge(bd_dataframes, bs_dataframes, how="left", on="Vehicle_Model")
Battery_data["Remaining_battery_capacity_reported"] = Battery_data["ActualRange"] / Battery_data["OriginalRange"]
Battery_data["Remaining_battery_capacity_calculated"] = Battery_data["ActualRange"] / Battery_data["Range"]
Battery_data["Remaining_battery_capacity"] = Battery_data["Remaining_battery_capacity"] .clip(upper=1)
Battery_data["Remaining_battery_capacity_reported"] = Battery_data["Remaining_battery_capacity_reported"] .clip(upper=1)
Battery_data["Remaining_battery_capacity_calculated"] = Battery_data["Remaining_battery_capacity_calculated"] .clip(upper=1)

# Battery_data.loc[Battery_data["Mileage_mile"] < 5000, "Remaining_battery_capacity"] = 1
# Battery_data.loc[Battery_data["Mileage_mile"] < 5000, "Remaining_battery_capacity_reported"] = 1
# Battery_data.loc[Battery_data["Mileage_mile"] < 5000, "Remaining_battery_capacity_calculated"] = 1
Battery_data["Battery_price"] = Battery_data["Battery_Capacity_kWh"] * 149
Battery_data = Battery_data[Battery_data["Remaining_battery_capacity_calculated"] > 0.90]


# %%
Tesla_dataframes = pd.read_csv("/Users/haniftayarani/V2G_Project/Batt_cap/TeslaBatteryDatabase.csv")


def convert_to_float_kwh(df, column_name):
    df[column_name] = df[column_name].str.replace(' kWh', '', regex=False)
    df[column_name] = df[column_name].str.replace(',', '.')
    df[column_name] = df[column_name].astype(float)
    return df


def convert_to_float_km(df, column_name):
    new_column_name = f"{column_name}_km"
    df[new_column_name] = df[column_name].str.replace(' km', '', regex=False)
    df[new_column_name] = df[new_column_name].str.replace(',', '.')
    df[new_column_name] = df[new_column_name].astype(float)
    return df


Tesla_dataframes = convert_to_float_kwh(Tesla_dataframes, "CapacityNetNew")
Tesla_dataframes = convert_to_float_kwh(Tesla_dataframes, "CapacityNetNow")
Tesla_dataframes = convert_to_float_km(Tesla_dataframes, "Odometer")
Tesla_dataframes = convert_to_float_km(Tesla_dataframes, "RatedRange")

Tesla_dataframes["Degradation_calculated"] = (1 - (Tesla_dataframes["CapacityNetNow"] / Tesla_dataframes["CapacityNetNew"]))*100
Tesla_dataframes.loc[Tesla_dataframes["Degradation_calculated"] < 0, "Degradation_calculated"] = 0
Tesla_dataframes["Battery_price"] = Tesla_dataframes["CapacityNetNew"] * 149
Tesla_dataframes["Charging_cycles"] = (Tesla_dataframes["Odometer_km"] / Tesla_dataframes["RatedRange_km"]).astype(int)
Tesla_dataframes["kWh_charged"] = Tesla_dataframes["Charging_cycles"] * Tesla_dataframes["CapacityNetNew"]
Tesla_dataframes = Tesla_dataframes[Tesla_dataframes["kWh_charged"] < 50000]
# %%


# Function to calculate degradation per kWh charged
def calculate_degradation_per_kwh(row):
    # Calculate the number of full cycles needed
    num_cycles = row['Mileage_mile'] / row['OriginalRange']

    # Total kWh used
    total_kwh_used = num_cycles * row['Battery_Capacity_kWh']

    # Remaining capacity in kWh
    # remaining_capacity_kwh = row['Remaining_battery_capacity'] * row['Battery_Capacity_kWh']
    remaining_capacity_kwh = row['Remaining_battery_capacity'] * row['Battery_Capacity_kWh']
    # Degraded capacity in kWh
    degraded_capacity_kwh = row['Battery_Capacity_kWh'] - remaining_capacity_kwh

    # Degradation per kWh charged
    degradation_per_kwh = degraded_capacity_kwh / total_kwh_used

    return degradation_per_kwh, num_cycles


# Apply the function to each row to calculate degradation per kWh charged and number of cycles
Battery_data[['Degradation_per_kWh', 'Number_of_Cycles']] = Battery_data.apply(
    lambda row: pd.Series(calculate_degradation_per_kwh(row)), axis=1)
Battery_data = Battery_data[~Battery_data['Degradation_per_kWh'].isna()]
Battery_data["Number_of_Cycles"] = Battery_data["Number_of_Cycles"].round()
Battery_data["kWh_charged"] = Battery_data["Number_of_Cycles"]*Battery_data["Battery_Capacity_kWh"]

# %%


# Function to categorize battery capacity into groups
# def categorize_battery_capacity(capacity):
#
#     if capacity > 0 and capacity <= 79:
#         return '60  to 79'
#     # elif capacity < 80 and capacity>= 70:
#     #     return '70  to 80'
#     else:
#         return 'Above 80'


def categorize_battery_capacity(capacity):

    if capacity < 0:
        return "Battery"
    else:
        return ''

def deg_plot(df, col_names, y_title='Depreciation Amount', equation_location=(0.5, 0.1), font_size=20):
    # Set global font size for the plot
    plt.rcParams.update({'font.size': font_size})

    # Create FacetGrid
    g = sns.FacetGrid(df, col='Battery_capacity_group', col_wrap=1, height=8, aspect=1.5)
    g.map_dataframe(sns.scatterplot, x='kWh_charged', y=col_names, color='b', label='Data Points')

    for ax_name, ax in g.axes_dict.items():
        data_subset = df[df['Battery_capacity_group'] == ax_name]

        # Perform linear regression through the origin
        x = data_subset['kWh_charged']
        y = data_subset[col_names]
        slope = np.sum(x * y) / np.sum(x ** 2)  # Slope calculation for regression through origin

        # Plot the best-fit line
        ax.plot(x, slope * x, color='r', label='Best Fit Line (Through Origin)')

        # Add equation of the best-fit line to the plot with larger font size
        equation = f'Depreciation Amount = {slope:.2e} * kWh_charged'
        ax.text(equation_location[0], equation_location[1], equation, fontsize=font_size, ha='center', transform=ax.transAxes, color='green')
        ax.set_xlim(0, 80000)
        ax.set_ylim(0, 3000)
        ax.grid(True)

    # Set labels and titles with larger font size
    g.set_xlabels('kWh Charged', fontsize=font_size)
    g.set_ylabels(y_title, fontsize=font_size)
    g.set_titles("", fontsize=font_size)

    # Adjust legend font size
    for ax in g.axes.flat:
        ax.legend(fontsize=font_size)

    plt.tight_layout()
    plt.show()



def health_plot(df, col_names, y_title='Remaining Battery Health', equation_location=(0.5, 0.1)):
    g = sns.FacetGrid(df, col='Battery_capacity_group', col_wrap=1, height=4, aspect=1.5)
    g.map_dataframe(sns.scatterplot, x='kWh_charged', y=col_names, color='b', label='Data Points')
    # Add best-fit lines for each plot
    for ax_name, ax in g.axes_dict.items():
        data_subset = df[df['Battery_capacity_group'] == ax_name]
        slope, intercept, _, _, _ = stats.linregress(data_subset['kWh_charged'], data_subset[col_names])
        ax.plot(data_subset['kWh_charged'], intercept + slope * data_subset['kWh_charged'], color='r', label='Best Fit Line')
        # Add equation of the best-fit line to the plot
        equation = f'Remaining Battery Health = {slope:.2e} * kWh_charged + {intercept:.2f}'
        ax.text(equation_location[0], equation_location[1], equation, fontsize=10, ha='center', transform=ax.transAxes, color='green')
        ax.set_xlim(0, 80000)
        ax.set_ylim(0.7, 1)
        ax.grid(True)
    # Set titles and labels
    # g.set_titles('Battery Capacity Group: {col_name} kWh')
    g.set_titles("")  # Set to empty string to remove titles

    # Set larger font size for x and y labels
    g.set_xlabels('kWh Charged', fontsize=16)
    g.set_ylabels(y_title, fontsize=16)

    # Increase the size of x-ticks and y-ticks
    g.set_xticklabels(fontsize=10)
    g.set_yticklabels(fontsize=14)
    # Add legend to each plot
    for ax in g.axes.flat:
        ax.legend()
    plt.tight_layout()
    plt.show()


def health_plot1(df, col_names, y_title='Remaining Battery Health', equation_location=(0.5, 0.1)):
    g = sns.FacetGrid(df, col='Battery_capacity_group', col_wrap=2, height=4, aspect=1.5)
    g.map_dataframe(sns.scatterplot, x='Mileage_mile', y=col_names, color='b', label='Data Points')
    # Add best-fit lines for each plot
    for ax_name, ax in g.axes_dict.items():
        data_subset = df[df['Battery_capacity_group'] == ax_name]
        slope, intercept, _, _, _ = stats.linregress(data_subset['Mileage_mile'], data_subset[col_names])
        ax.plot(data_subset['Mileage_mile'], intercept + slope * data_subset['Mileage_mile'], color='r', label='Best Fit Line')
        # Add equation of the best-fit line to the plot
        equation = f'Remaining Battery Health = {slope:.2e} * Mileage_mile + {intercept:.2f}'
        ax.text(equation_location[0], equation_location[1], equation, fontsize=10, ha='center', transform=ax.transAxes, color='green')
        ax.set_xlim(0, 80000)
        ax.set_ylim(0.7, 1)
        ax.grid(True)
    # Set titles and labels
    # g.set_titles('Battery Capacity Group: {col_name} kWh')
    g.set_xlabels('Mileage_mile')
    g.set_ylabels(y_title)
    # Add legend to each plot
    for ax in g.axes.flat:
        ax.legend()
    plt.tight_layout()
    plt.show()

def deg_plot_new(df, col_name, y_title='Remaining Battery Health', equation_location=(0.5, 0.9)):
    # Define a consistent color palette for chemistries
    unique_chemistries = df['Chemistry'].unique() if 'Chemistry' in df.columns else []
    palette = sns.color_palette("deep", len(unique_chemistries))
    color_dict = dict(zip(unique_chemistries, palette))

    g = sns.FacetGrid(df, col='Battery_capacity_group', col_wrap=2, height=4, aspect=1.5)

    if 'Chemistry' in df.columns:
        g.map_dataframe(sns.scatterplot, x='kWh_charged', y=col_name, hue='Chemistry', palette=color_dict)
    else:
        g.map_dataframe(sns.scatterplot, x='kWh_charged', y=col_name, color='b')

    for ax_name, ax in g.axes_dict.items():
        data_subset = df[df['Battery_capacity_group'] == ax_name]

        if 'Chemistry' in df.columns:
            for idx, chem in enumerate(data_subset['Chemistry'].unique()):
                chem_subset = data_subset[data_subset['Chemistry'] == chem]
                x = chem_subset['kWh_charged']
                y = chem_subset[col_name]
                slope = np.sum(x * y) / np.sum(x ** 2)
                ax.plot(x, slope * x, label=f'Best Fit Line ({chem})', color=color_dict[chem])
                equation = f'{chem}: {slope:.2e} * kWh_charged'
                ax.text(equation_location[0], equation_location[1] - idx * 0.05, equation, fontsize=10, ha='center', transform=ax.transAxes, color=color_dict[chem])
        else:
            x = data_subset['kWh_charged']
            y = data_subset[col_name]
            slope = np.sum(x * y) / np.sum(x ** 2)
            ax.plot(x, slope * x, color='r', label='Best Fit Line (Through Origin)')
            equation = f'Remaining Battery Health = {slope:.2e} * kWh_charged'
            ax.text(equation_location[0], equation_location[1], equation, fontsize=10, ha='center', transform=ax.transAxes, color='green')
        ax.set_xlim(0, 80000)
        ax.set_ylim(0, 3000)
        ax.grid(True)
    g.set_titles('Battery Capacity Group: {col_name} kWh')
    g.set_xlabels('kWh Charged')
    g.set_ylabels(y_title)

    for ax in g.axes.flat:
        ax.legend()

    plt.tight_layout()
    plt.show()


def health_plot_new(df, col_name, y_title='Remaining Battery Health', equation_location=(0.5, 0.9)):
    # Define a consistent color palette for chemistries
    unique_chemistries = df['Chemistry'].unique() if 'Chemistry' in df.columns else []
    palette = sns.color_palette("deep", len(unique_chemistries))
    color_dict = dict(zip(unique_chemistries, palette))

    g = sns.FacetGrid(df, col='Battery_capacity_group', col_wrap=2, height=4, aspect=1.5)

    if 'Chemistry' in df.columns:
        g.map_dataframe(sns.scatterplot, x='kWh_charged', y=col_name, hue='Chemistry', palette=color_dict)
    else:
        g.map_dataframe(sns.scatterplot, x='kWh_charged', y=col_name, color='b')

    for ax_name, ax in g.axes_dict.items():
        data_subset = df[df['Battery_capacity_group'] == ax_name]

        if 'Chemistry' in df.columns:
            for idx, chem in enumerate(data_subset['Chemistry'].unique()):
                chem_subset = data_subset[data_subset['Chemistry'] == chem]
                slope, intercept, _, _, _ = stats.linregress(chem_subset['kWh_charged'], chem_subset[col_name])
                ax.plot(chem_subset['kWh_charged'], intercept + slope * chem_subset['kWh_charged'], label=f'Best Fit Line ({chem})', color=color_dict[chem])
                equation = f'{chem}: {slope:.2e} * kWh_charged + {intercept:.2f}'
                ax.text(equation_location[0], equation_location[1] - idx * 0.05, equation, fontsize=10, ha='center', transform=ax.transAxes, color=color_dict[chem])
        else:
            slope, intercept, _, _, _ = stats.linregress(data_subset['kWh_charged'], data_subset[col_name])
            ax.plot(data_subset['kWh_charged'], intercept + slope * data_subset['kWh_charged'], color='r', label='Best Fit Line')
            equation = f'Remaining Battery Health = {slope:.2e} * kWh_charged + {intercept:.2f}'
            ax.text(equation_location[0], equation_location[1], equation, fontsize=10, ha='center', transform=ax.transAxes, color='green')
        ax.set_xlim(0, 80000)
        ax.set_ylim(0.7, 1)
        ax.grid(True)
    g.set_titles('Battery Capacity Group: {col_name} kWh')
    g.set_xlabels('kWh Charged')
    g.set_ylabels(y_title)

    for ax in g.axes.flat:
        ax.legend()

    plt.tight_layout()

    plt.show()

# %%


# Example usage:
# Add a new column for battery capacity groups
Battery_data['Battery_capacity_group'] = Battery_data['Battery_Capacity_kWh'].apply(categorize_battery_capacity)
Tesla_dataframes['Battery_capacity_group'] = Tesla_dataframes['CapacityNetNew'].apply(categorize_battery_capacity)

# Calculate depreciation amount
Battery_data["depreciation_amount"] = Battery_data["Battery_price"] * (1.0 - Battery_data["Remaining_battery_capacity_calculated"])
Tesla_dataframes["depreciation_amount"] = Tesla_dataframes["Battery_price"] * (1.0 - Tesla_dataframes["CapacityNetNow"]/Tesla_dataframes["CapacityNetNew"])
Tesla_dataframes.loc[Tesla_dataframes["depreciation_amount"] < 0, "depreciation_amount"] = 0
Tesla_dataframes.loc[Tesla_dataframes["Chemistry"] == "NCA MIC", "Chemistry"] = "NCA"
Tesla_dataframes["Remaining_battery_capacity_calculated"] = Tesla_dataframes["CapacityNetNow"]/Tesla_dataframes["CapacityNetNew"]
Tesla_dataframes = Tesla_dataframes[Tesla_dataframes["depreciation_amount"] < 3000]


deg_plot_new(Tesla_dataframes, 'depreciation_amount', y_title='Depreciation Amount $', equation_location=(0.5, 0.6))
deg_plot(Battery_data, 'depreciation_amount', y_title='Depreciation Amount $', equation_location=(0.5, 0.6))
health_plot_new(Tesla_dataframes, 'Remaining_battery_capacity_calculated', y_title='Remaining Battery Capacity', equation_location=(0.5, 0.3))
health_plot(Battery_data, 'Remaining_battery_capacity', y_title='Remaining Battery Capacity', equation_location=(0.5, 0.3))
health_plot1(Battery_data, 'Remaining_battery_capacity', y_title='Remaining Battery Capacity', equation_location=(0.5, 0.3))

# %%
def batt_mile(data, y_title):
    df = pd.DataFrame(data)
    df = df[df["Remaining_battery_capacity"] > 0.5]

    # Function to keep only part before the second space
    def trim_vehicle_model(name):
        parts = name.split()
        if len(parts) > 2:
            return ' '.join(parts[:2])
        return name

    # Apply the function to the 'Vehicle_Model' column
    df['Vehicle_Model'] = df['Vehicle_Model'].apply(trim_vehicle_model)

    # Plot
    plt.figure(figsize=(10, 6))
    scatter_plot = sns.scatterplot(
        data=df,
        x='Mileage',
        y='Remaining_battery_capacity',
        hue='Vehicle_Model',
        size='Battery_Capacity_kWh',
        sizes=(50, 200),
        alpha=0.7
    )

    # Set titles and labels
    # scatter_plot.set_title('Battery Health vs. Mileage', fontsize=15)
    scatter_plot.set_xlabel('Mileage', fontsize=20)
    scatter_plot.set_ylabel(y_title, fontsize=20)

    # Customize legend
    handles, labels = scatter_plot.get_legend_handles_labels()
    vehicle_model_handles = handles[1:len(df['Vehicle_Model'].unique()) + 1]
    vehicle_model_labels = labels[1:len(df['Vehicle_Model'].unique()) + 1]
    battery_capacity_handles = handles[len(df['Vehicle_Model'].unique()) + 2:]
    battery_capacity_labels = labels[len(df['Vehicle_Model'].unique()) + 2:]

    # Create a new legend with custom titles
    legend1 = scatter_plot.legend(vehicle_model_handles, vehicle_model_labels, title='Vehicle Model', loc='lower center', fontsize=16, title_fontsize=14)
    scatter_plot.add_artist(legend1)
    scatter_plot.legend(battery_capacity_handles, battery_capacity_labels, title='Battery Capacity (kWh)', loc='lower right', fontsize=16, title_fontsize=14)

    # Customize ticks
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.ylim(0.7, 1)
    # Add grid
    plt.grid(True)

    # Show plot
    plt.tight_layout()
    plt.show()


# Calling the function
batt_mile(Battery_data, 'Remaining Battery Capacity')
