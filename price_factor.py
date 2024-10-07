# %%
import pandas as pd
import json
import os
import pandas as pd
import zipfile
import json
import matplotlib as plt
from parking import draw_RT
# %%
# # %% Define the directory where your CSV files are located
# folder_path = '/Users/haniftayarani/V2G_data/real_time_price'
#
# # Initialize an empty list to store the dataframes
# dataframes_list = []
#
# # Loop through all the files in the directory
# for file_name in os.listdir(folder_path):
#     # Check if the file is a CSV
#     if file_name.endswith('.csv'):
#         # Construct full file path
#         file_path = os.path.join(folder_path, file_name)
#         # Read the CSV file and store it in the list
#         df = pd.read_csv(file_path)
#         dataframes_list.append(df)
#
# # Concatenate all the dataframes in the list
# combined_price = pd.concat(dataframes_list, ignore_index=True)
# combined_price = combined_price[combined_price["LMP_TYPE"] == "LMP"]
# combined_price = combined_price.drop(columns=["NODE_ID_XML", "NODE", "PNODE_RESMRID", "POS", "OPR_INTERVAL", "MARKET_RUN_ID", "XML_DATA_ITEM", "GRP_TYPE", "OPR_DT"])
#
# # Assuming your dataframe is named 'df'
# pge_values = ['PGCC', 'PGEB', 'PGF1', 'PGFG', 'PGHB', 'PGKN', 'PGLP', 'PGNB', 'PGNC', 'PGNP', 'PGNV', 'PGP2', 'PGSA', 'PGSB', 'PGSF', 'PGSI', 'PGSN', 'PGST', 'PGZP']
# sce_values = ['SCEC', 'SCEN', 'SCEW', 'SCHD', 'SCLD', 'SCNW']
# sdge_values = ['SDG1']
#
# # Filter rows where the first four letters of 'NODE_ID' values are in the list
# combined_price_PGE = combined_price[combined_price['NODE_ID'].str[:4].isin(pge_values)]
# combined_price_SCE = combined_price[combined_price['NODE_ID'].str[:4].isin(sce_values)]
# combined_price_SDGE = combined_price[combined_price['NODE_ID'].str[:4].isin(sdge_values)]
#
# lengths_dict = {
#     'PGE': len(combined_price_PGE),
#     'SCE': len(combined_price_SCE),
#     'SDGE': len(combined_price_SDGE)
# }
# rt_dict = combined_price_PGE.to_dict()
#
# # %%
# # Path to the folder containing the zip file
# folder_path = '/Users/haniftayarani/V2G_Project/demand'
#
# # Iterate over all files in the folder
# for file_name in os.listdir(folder_path):
#     # Check if the file is a zip file
#     if file_name.endswith('.zip'):
#         # Unzip the file
#         with zipfile.ZipFile(os.path.join(folder_path, file_name), 'r') as zip_ref:
#             zip_ref.extractall(folder_path)
#
# # Get a list of all CSV files in the extracted folder
# csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
#
# # Read each CSV file and concatenate them into one DataFrame
# dfs = []
# for csv_file in csv_files:
#     df = pd.read_csv(os.path.join(folder_path, csv_file))
#     dfs.append(df)
#
# combined_demand = pd.concat(dfs, ignore_index=True)
# combined_demand = combined_demand.drop(columns=["LOAD_TYPE", "OPR_DT", "OPR_INTERVAL", "MARKET_RUN_ID", "LABEL", "MARKET_RUN_ID", "POS"])
#
# combined_demand_PGE = combined_demand[combined_demand["TAC_AREA_NAME"] == "PGE-TAC"].reset_index(drop=True)
# combined_demand_PGE = combined_demand_PGE.drop_duplicates(subset=["INTERVALSTARTTIME_GMT"])
#
# combined_demand_SCE = combined_demand[combined_demand["TAC_AREA_NAME"] == "SCE-TAC"].reset_index(drop=True)
# combined_demand_SCE = combined_demand_SCE.drop_duplicates(subset=["INTERVALSTARTTIME_GMT"])
#
# combined_demand_SDGE = combined_demand[combined_demand["TAC_AREA_NAME"] == "SDGE-TAC"].reset_index(drop=True)
# combined_demand_SDGE = combined_demand_SDGE.drop_duplicates(subset=["INTERVALSTARTTIME_GMT"])
#
# # %%
# class RTPricer:
#     def __init__(self, df1, df2, price_tou_low, price_tou_high, range_start, range_end, label):
#         self.df1 = df1
#         self.df2 = df2
#         self.price_tou_low = price_tou_low
#         self.price_tou_high = price_tou_high
#         self.range_start = range_start
#         self.range_end = range_end
#         self.label = label
#
#     def calculate_rt_price(self):
#         # Calculate average load
#         self.df1["average_load"] = self.df1["MW"] / len(self.df2["NODE_ID"].unique())
#
#         # Merge demand data with price data
#         self.df2 = pd.merge(self.df2, self.df1[["INTERVALSTARTTIME_GMT", "average_load"]], on="INTERVALSTARTTIME_GMT", how="left")
#
#         # Calculate revenue
#         self.df2["revenue"] = self.df2["average_load"] * self.df2["MW"]
#         total_rt = self.df2["revenue"].sum()
#
#         # Calculate total TOU revenue
#         pge_price_tou = {key: self.price_tou_low if (key in range(self.range_start)) or (key in range(self.range_end, 24)) else self.price_tou_high for key in range(24)}
#         pge_load = self.df1.groupby("OPR_HR")["MW"].sum().to_dict()
#         total_tou = sum(pge_price_tou[key] * pge_load[key] for key in pge_price_tou if key in pge_load)
#
#         # Adjust factor
#         adj_factor = total_tou / total_rt
#
#         # Calculate real-time price
#         self.df2["rt_price"] = self.df2["MW"] * adj_factor
#         self.df2["rt_price_generation"] = self.df2["MW"]
#
#         return self.df2, adj_factor
#
#     # def plot_histogram(self):
#     #     plt.hist(self.df2["rt_price"]/1000, bins=20, color='skyblue', edgecolor='black')
#     #
#     #     # Add labels and title with the provided label
#     #     plt.xlabel('Real-time Price ($/kWh)')
#     #     plt.ylabel('Frequency')
#     #     plt.title(f'Histogram of Real-time Price for {self.label}')  # Include the provided label in the title
#     #
#     #     # Show plot
#     #     plt.show()
#
# # %%
#
#
# rt_pricer = RTPricer(combined_demand_PGE, combined_price_PGE, 470, 535, 16, 21, "PGE")
# combined_price_PGE_new, adj_factor_PGE = rt_pricer.calculate_rt_price()
# # rt_pricer.plot_histogram()
# combined_price_PGE_new["INTERVALSTARTTIME_PST"] = pd.to_datetime(combined_price_PGE_new["INTERVALSTARTTIME_GMT"]).dt.tz_convert('America/Los_Angeles')
# combined_price_PGE_new['hour_of_year_start'] = combined_price_PGE_new['INTERVALSTARTTIME_PST'].apply(lambda x: ((x.dayofyear - 1) * 24 + x.hour))
# combined_price_PGE_new = combined_price_PGE_new.sort_values(by="INTERVALSTARTTIME_PST").reset_index(drop=True)
# combined_price_PGE_average = combined_price_PGE_new[["hour_of_year_start", "rt_price"]]
# combined_price_PGE_average = combined_price_PGE_average.groupby("hour_of_year_start")["rt_price"].mean()
# combined_price_PGE_average = pd.concat([combined_price_PGE_average, combined_price_PGE_average, combined_price_PGE_average], axis=0).reset_index(drop=True).to_dict()
#
# combined_price_PGE_old = combined_price_PGE_new[["hour_of_year_start", "rt_price_generation"]]
# combined_price_PGE_old = combined_price_PGE_old.groupby("hour_of_year_start")["rt_price_generation"].mean()
# combined_price_PGE_old = pd.concat([combined_price_PGE_old,combined_price_PGE_old], axis=0).reset_index(drop=True).to_dict()
# draw_RT(combined_price_PGE_old)
# draw_RT(combined_price_PGE_average)
#
# rt_pricer = RTPricer(combined_demand_SCE, combined_price_SCE, 275, 475, 16, 21, "SCE")
# combined_price_SCE_new, adj_factor_SCE = rt_pricer.calculate_rt_price()
# # rt_pricer.plot_histogram()
# combined_price_SCE_new["INTERVALSTARTTIME_PST"] = pd.to_datetime(combined_price_SCE_new["INTERVALSTARTTIME_GMT"]).dt.tz_convert('America/Los_Angeles')
# combined_price_SCE_new['hour_of_year_start'] = combined_price_SCE_new['INTERVALSTARTTIME_PST'].apply(lambda x: ((x.dayofyear - 1) * 24 + x.hour))
# combined_price_SCE_new = combined_price_SCE_new.sort_values(by="INTERVALSTARTTIME_PST").reset_index(drop=True)
# combined_price_SCE_average = combined_price_SCE_new[["hour_of_year_start", "rt_price"]]
# combined_price_SCE_average = combined_price_SCE_average.groupby("hour_of_year_start")["rt_price"].mean()
# combined_price_SCE_average = pd.concat([combined_price_SCE_average, combined_price_SCE_average, combined_price_SCE_average], axis=0).reset_index(drop=True).to_dict()
#
# rt_pricer = RTPricer(combined_demand_SDGE, combined_price_SDGE, 312, 417, 17, 21, "PGE")
# combined_price_SDGE_new, adj_factor_SDGE = rt_pricer.calculate_rt_price()
# # rt_pricer.plot_histogram()
# combined_price_SDGE_new["INTERVALSTARTTIME_PST"] = pd.to_datetime(combined_price_SDGE_new["INTERVALSTARTTIME_GMT"]).dt.tz_convert('America/Los_Angeles')
# combined_price_SDGE_new['hour_of_year_start'] = combined_price_SDGE_new['INTERVALSTARTTIME_PST'].apply(lambda x: ((x.dayofyear - 1) * 24 + x.hour))
# combined_price_SDGE_new = combined_price_SDGE_new.sort_values(by="INTERVALSTARTTIME_PST").reset_index(drop=True)
# combined_price_SDGE_new_average = combined_price_SDGE_new[["hour_of_year_start", "rt_price"]]
# combined_price_SDGE_new_average = combined_price_SDGE_new_average.groupby("hour_of_year_start")["rt_price"].mean()
# combined_price_SDGE_new_average = pd.concat([combined_price_SDGE_new_average, combined_price_SDGE_new_average, combined_price_SDGE_new_average], axis=0).reset_index(drop=True).to_dict()
#
# # %%
# with open("combined_price_PGE_average.json", "w") as json_file:
#     json.dump(combined_price_PGE_average, json_file)
#
# with open("combined_price_SCE_average.json", "w") as json_file:
#     json.dump(combined_price_SCE_average, json_file)
#
# with open("combined_price_SDGE_new_average.json", "w") as json_file:
#     json.dump(combined_price_SDGE_new_average, json_file)


# %% TOU signal
def tou_price(pp_summer, opp_summer, pp_winter, opp_winter):
    # Define the peak and off-peak prices for summer and winter
    summer_peak_price = pp_summer
    summer_off_peak_price = opp_summer
    winter_peak_price = pp_winter
    winter_off_peak_price = opp_winter

    # Initialize the dictionary to hold TOU prices for each hour of the year
    tou_prices = {}

    # Generate TOU prices for each hour of the year and store as strings
    for hour in range(8760):
        hour_str = str(hour)
        month = (hour // 720) % 12 + 1  # Calculate month (1-12)
        hour_of_day = hour % 24

        # Determine summer and winter periods
        if 6 <= month <= 9:  # June to September (summer)
            if 16 <= hour_of_day < 21:
                tou_prices[hour_str] = summer_peak_price
            else:
                tou_prices[hour_str] = summer_off_peak_price
        else:  # October to May (winter)
            if 16 <= hour_of_day < 21:
                tou_prices[hour_str] = winter_peak_price
            else:
                tou_prices[hour_str] = winter_off_peak_price

    # Create a list to hold the combined data for four years
    combined_tou_prices = {}

    # Concatenate the data three times to represent four years
    for year in range(3):
        for hour in range(8760):
            combined_hour_str = str(year * 8760 + hour)
            combined_tou_prices[combined_hour_str] = tou_prices[str(hour)]

    return combined_tou_prices


def ev_rate_price(so, sm, sp, wo, wm, wp):
    # Define EV rate prices for summer and winter
    ev_summer_prices = [so, sm, sp, sm]  # Prices for summer EV rate (12 am - 3 pm, 3 - 4 pm, 4 - 9 pm, 9 pm - 12 am)
    ev_winter_prices = [wo, wm, wp, wm]  # Prices for winter EV rate (12 am - 3 pm, 3 - 4 pm, 4 - 9 pm, 9 pm - 12 am)

    # Initialize the dictionary to hold EV rate prices for each hour of the year
    ev_prices = {}

    # Generate EV rate prices for each hour of the year and store as strings
    for hour in range(8760):
        hour_str = str(hour)
        month = (hour // 720) % 12 + 1  # Calculate month (1-12)
        hour_of_day = hour % 24

        # Determine summer and winter periods for EV rates
        if 6 <= month <= 9:  # Summer EV rate
            if hour_of_day < 15:
                ev_prices[hour_str] = ev_summer_prices[0]
            elif 15 <= hour_of_day < 16:
                ev_prices[hour_str] = ev_summer_prices[1]
            elif 16 <= hour_of_day < 21:
                ev_prices[hour_str] = ev_summer_prices[2]
            elif 21 <= hour_of_day < 24:
                ev_prices[hour_str] = ev_summer_prices[3]
        else:  # Winter EV rate
            if hour_of_day < 15:
                ev_prices[hour_str] = ev_winter_prices[0]
            elif 15 <= hour_of_day < 16:
                ev_prices[hour_str] = ev_winter_prices[1]
            elif 16 <= hour_of_day < 21:
                ev_prices[hour_str] = ev_winter_prices[2]
            elif 21 <= hour_of_day < 24:
                ev_prices[hour_str] = ev_winter_prices[0]  # Same as early morning price

    # Create a list to hold the combined data for four years
    combined_ev_prices = {}

    # Concatenate the data three times to represent four years
    for year in range(3):
        for hour in range(8760):
            combined_hour_str = str(year * 8760 + hour)
            combined_ev_prices[combined_hour_str] = ev_prices[str(hour)]

    return combined_ev_prices

# %%
#
# combined_price_PGE_gen = combined_price_PGE_new[["hour_of_year_start", "rt_price_generation"]]
# combined_price_PGE_gen = combined_price_PGE_gen.groupby("hour_of_year_start")["rt_price_generation"].mean()
# combined_price_PGE_gen = pd.concat([combined_price_PGE_gen, combined_price_PGE_gen, combined_price_PGE_gen], axis=0).reset_index(drop=True).to_dict()