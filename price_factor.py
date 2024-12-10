# %%
import os
import pandas as pd
import zipfile
import json
import matplotlib as plt
from parking import draw_RT
# %% Price input
utility_data = {
    "PGE": {
        "TOU": {
            "weekday_prices": {'summer': {'peak': 610, 'mid_peak': 500, 'off_peak': 500}, 'winter': {'peak': 490, 'mid_peak': 460, 'off_peak': 460}},
            "weekend_prices": {'summer': {'peak': 610, 'mid_peak': 500, 'off_peak': 500}, 'winter': {'peak': 490, 'mid_peak': 460, 'off_peak': 460}},
            "weekday_times": {'summer': {'peak': [(16, 21)], 'mid_peak': [(26, 26)]}, 'winter': {'peak': [(16, 21)], 'mid_peak': [(26, 26)]}},
            "weekend_times": {'summer': {'peak': [(16, 21)], 'mid_peak': [(26, 26)]}, 'winter': {'peak': [(16, 21)], 'mid_peak': [(26, 26)]}},
            "month_ranges": {'summer': (6, 9), 'winter': (10, 5)}
        },
        "EVR": {
            "weekday_prices": {'summer': {'peak': 640, 'mid_peak': 530, 'off_peak': 320}, 'winter': {'peak': 510, 'mid_peak': 490, 'off_peak': 320}},
            "weekend_prices": {'summer': {'peak': 640, 'mid_peak': 530, 'off_peak': 320}, 'winter': {'peak': 510, 'mid_peak': 490, 'off_peak': 320}},
            "weekday_times": {'summer': {'peak': [(16, 21)], 'mid_peak': [(15, 16), (21, 24)]}, 'winter': {'peak': [(16, 21)], 'mid_peak': [(26, 26), (21, 24)]}},
            "weekend_times": {'summer': {'peak': [(16, 21)], 'mid_peak': [(15, 16), (21, 24)]}, 'winter': {'peak': [(16, 21)], 'mid_peak': [(26, 26), (21, 24)]}},
            "month_ranges": {'summer': (6, 9), 'winter': (10, 5)}
        },
        "Commercial": {
            "weekday_prices": {'summer': {'peak': 298, 'mid_peak': 78, 'off_peak': 102}, 'winter': {'peak': 298, 'mid_peak': 78, 'off_peak': 102}},
            "weekend_prices": {'summer': {'peak': 298, 'mid_peak': 78, 'off_peak': 102}, 'winter': {'peak': 298, 'mid_peak': 78, 'off_peak': 102}},
            "weekday_times": {'summer': {'peak': [(16, 21)], 'mid_peak': [(9, 14)]}, 'winter': {'peak': [(16, 21)], 'mid_peak': [(9, 14)]}},
            "weekend_times": {'summer': {'peak': [(16, 21)], 'mid_peak': [(9, 14)]}, 'winter': {'peak': [(16, 21)], 'mid_peak': [(9, 14)]}},
            "month_ranges": {'summer': (6, 9), 'winter': (10, 5)}
        }
    },
    "SCE": {
        "TOU": {
            "weekday_prices": {'summer': {'peak': 580, 'mid_peak': 360, 'off_peak': 360}, 'winter': {'peak': 510, 'mid_peak': 350, 'off_peak': 390}},
            "weekend_prices": {'summer': {'peak': 470, 'mid_peak': 360, 'off_peak': 360}, 'winter': {'peak': 510, 'mid_peak': 350, 'off_peak': 390}},
            "weekday_times": {'summer': {'peak': [(16, 21)], 'mid_peak': [(26, 26)]}, 'winter': {'peak': [(16, 21)], 'mid_peak': [(8, 16)]}},
            "weekend_times": {'summer': {'peak': [(16, 21)], 'mid_peak': [(26, 26)]}, 'winter': {'peak': [(16, 21)], 'mid_peak': [(8, 16)]}},
            "month_ranges": {'summer': (6, 9), 'winter': (10, 5)}
        },
        "EVR": {
            "weekday_prices": {'summer': {'peak': 570, 'mid_peak': 250, 'off_peak': 250}, 'winter': {'peak': 550, 'mid_peak': 230, 'off_peak': 230}},
            "weekend_prices": {'summer': {'peak': 570, 'mid_peak': 250, 'off_peak': 250}, 'winter': {'peak': 550, 'mid_peak': 230, 'off_peak': 230}},
            "weekday_times": {'summer': {'peak': [(16, 21)], 'mid_peak': [(26, 26)]}, 'winter': {'peak': [(16, 21)], 'mid_peak': [(8, 16)]}},
            "weekend_times": {'summer': {'peak': [(16, 21)], 'mid_peak': [(26, 26)]}, 'winter': {'peak': [(16, 21)], 'mid_peak': [(8, 16)]}},
            "month_ranges": {'summer': (6, 9), 'winter': (10, 5)}
        },
        "Commercial": {
            "weekday_prices": {'summer': {'peak': 332, 'mid_peak': 117, 'off_peak': 96}, 'winter': {'peak': 179, 'mid_peak': 114, 'off_peak': 52}},
            "weekend_prices": {'summer': {'peak': 332, 'mid_peak': 117, 'off_peak': 96}, 'winter': {'peak': 179, 'mid_peak': 114, 'off_peak': 52}},
            "weekday_times": {'summer': {'peak': [(16, 21)], 'mid_peak': [(26, 26)]}, 'winter': {'peak': [(16, 21)], 'mid_peak': [(21, 24), (0, 8)]}},
            "weekend_times": {'summer': {'peak': [(26, 26)], 'mid_peak': [(16, 21)]}, 'winter': {'peak': [(16, 21)], 'mid_peak': [(21, 24), (0, 8)]}},
            "month_ranges": {'summer': (6, 9), 'winter': (10, 5)}
        }
    },
    "SDGE": {
        "TOU": {
            "weekday_prices": {'summer': {'peak': 561, 'mid_peak': 499, 'off_peak': 481}, 'winter': {'peak': 561, 'mid_peak': 499, 'off_peak': 481}},
            "weekend_prices": {'summer': {'peak': 561, 'mid_peak': 499, 'off_peak': 481}, 'winter': {'peak': 561, 'mid_peak': 499, 'off_peak': 481}},
            "weekday_times": {'summer': {'peak': [(16, 21)], 'mid_peak': [(6, 16), (21, 24)]}, 'winter': {'peak': [(16, 21)], 'mid_peak': [(6, 10), (21, 24)]}},
            "weekend_times": {'summer': {'peak': [(16, 21)], 'mid_peak': [(14, 16), (21, 24)]}, 'winter': {'peak': [(16, 21)], 'mid_peak': [(14, 16), (21, 24)]}},
            "month_ranges": {'summer': (6, 9), 'winter': (10, 5)}
        },
        "EVR": {
            "weekday_prices": {'summer': {'peak': 455, 'mid_peak': 408, 'off_peak': 235}, 'winter': {'peak': 455, 'mid_peak': 408, 'off_peak': 124}},
            "weekend_prices": {'summer': {'peak': 455, 'mid_peak': 408, 'off_peak': 124}, 'winter': {'peak': 455, 'mid_peak': 408, 'off_peak': 124}},
            "weekday_times": {'summer': {'peak': [(16, 21)], 'mid_peak': [(6, 16)]}, 'winter': {'peak': [(16, 21)], 'mid_peak': [(6, 16)]}},
            "weekend_times": {'summer': {'peak': [(16, 21)], 'mid_peak': [(14, 16)]}, 'winter': {'peak': [(16, 21)], 'mid_peak': [(14, 16)]}},
            "month_ranges": {'summer': (6, 9), 'winter': (10, 5)}
        },
        "Commercial": {
            "weekday_prices": {'summer': {'peak': 640, 'mid_peak': 530, 'off_peak': 320}, 'winter': {'peak': 510, 'mid_peak': 490, 'off_peak': 320}},
            "weekend_prices": {'summer': {'peak': 640, 'mid_peak': 530, 'off_peak': 320}, 'winter': {'peak': 510, 'mid_peak': 490, 'off_peak': 320}},
            "weekday_times": {'summer': {'peak': [(16, 21)], 'mid_peak': [(6, 16), (21, 24)]}, 'winter': {'peak': [(16, 21)], 'mid_peak': [(6, 16), (21, 24)]}},
            "weekend_times": {'summer': {'peak': [(16, 21)], 'mid_peak': [(14, 16), (21, 24)]}, 'winter': {'peak': [(16, 21)], 'mid_peak': [(14, 16), (21, 24)]}},
            "month_ranges": {'summer': (6, 9), 'winter': (10, 5)}
        },

    },
    "SMUD": {
        "TOU": {
            "weekday_prices": {'summer': {'peak': 346, 'mid_peak': 196, 'off_peak': 142}, 'winter': {'peak': 163, 'mid_peak': 118, 'off_peak': 118}},
            "weekend_prices": {'summer': {'peak': 346, 'mid_peak': 196, 'off_peak': 142}, 'winter': {'peak': 163, 'mid_peak': 118, 'off_peak': 118}},
            "weekday_times": {'summer': {'peak': [(17, 20)], 'mid_peak': [(12, 17), (20, 24)]}, 'winter': {'peak': [(17, 20)], 'mid_peak': [(26, 26)]}},
            "weekend_times": {'summer': {'peak': [(17, 20)], 'mid_peak': [(12, 17), (20, 24)]}, 'winter': {'peak': [(17, 20)], 'mid_peak': [(26, 26)]}},
            "month_ranges": {'summer': (6, 9), 'winter': (10, 5)}
        },
        "EVR": {
            "weekday_prices": {'summer': {'peak': 346, 'mid_peak': 196, 'off_peak': 142}, 'winter': {'peak': 163, 'mid_peak': 118, 'off_peak': 118}},
            "weekend_prices": {'summer': {'peak': 346, 'mid_peak': 196, 'off_peak': 142}, 'winter': {'peak': 163, 'mid_peak': 118, 'off_peak': 118}},
            "weekday_times": {'summer': {'peak': [(17, 20)], 'mid_peak': [(12, 17), (20, 24)]}, 'winter': {'peak': [(17, 20)], 'mid_peak': [(26, 26)]}},
            "weekend_times": {'summer': {'peak': [(17, 20)], 'mid_peak': [(12, 17), (20, 24)]}, 'winter': {'peak': [(17, 20)], 'mid_peak': [(26, 26)]}},
            "month_ranges": {'summer': (6, 9), 'winter': (10, 5)}
        },
        "Commercial": {
            "weekday_prices": {'summer': {'peak': 210, 'mid_peak': 107, 'off_peak': 107}, 'winter': {'peak': 136, 'mid_peak': 111, 'off_peak': 71}},
            "weekend_prices": {'summer': {'peak': 210, 'mid_peak': 107, 'off_peak': 107}, 'winter': {'peak': 136, 'mid_peak': 111, 'off_peak': 71}},
            "weekday_times": {'summer': {'peak': [(16, 21)], 'mid_peak': [(26, 26)]}, 'winter': {'peak': [(16, 21)], 'mid_peak': [(9, 16)]}},
            "weekend_times": {'summer': {'peak': [(26, 26)], 'mid_peak': [(26, 26)]}, 'winter': {'peak': [(26, 26)], 'mid_peak': [(9, 16)]}},
            "month_ranges": {'summer': (6, 9), 'winter': (10, 5)}
        }
    }
}
#
#
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
# combined_price = combined_price.drop(columns=["NODE_ID_XML", "NODE", "PNODE_RESMRID", "POS",
#                                               "OPR_INTERVAL", "MARKET_RUN_ID", "XML_DATA_ITEM",
#                                               "GRP_TYPE", "OPR_DT"])
#
# # Assuming your dataframe is named 'df'
# pge_values = ['PGCC', 'PGEB', 'PGF1', 'PGFG', 'PGHB', 'PGKN', 'PGLP', 'PGNB', 'PGNC',
#               'PGNP', 'PGNV', 'PGP2', 'PGSA', 'PGSB', 'PGSF', 'PGSI', 'PGSN', 'PGST', 'PGZP']
# sce_values = ['SCEC', 'SCEN', 'SCEW', 'SCHD', 'SCLD', 'SCNW']
#
# sdge_values = ['SDG1']
#
# smud_values = ["SMD"]
#
# # Filter rows where the first four letters of 'NODE_ID' values are in the list
# combined_price_PGE = combined_price[combined_price['NODE_ID'].str[:4].isin(pge_values)]
# combined_price_SCE = combined_price[combined_price['NODE_ID'].str[:4].isin(sce_values)]
# combined_price_SDGE = combined_price[combined_price['NODE_ID'].str[:4].isin(sdge_values)]
# combined_price_SMUD = combined_price[combined_price['NODE_ID'].str[:3].isin(smud_values)]
#
#
# lengths_dict = {
#     'PGE': len(combined_price_PGE),
#     'SCE': len(combined_price_SCE),
#     'SDGE': len(combined_price_SDGE),
#     'SMUD': len(combined_price_SMUD)
#
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
# combined_demand_SMUD = combined_demand[combined_demand["TAC_AREA_NAME"] == "BANCSMUD"].reset_index(drop=True)
# combined_demand_SMUD = combined_demand_SMUD.drop_duplicates(subset=["INTERVALSTARTTIME_GMT"])
#
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
# rt_pricer = RTPricer(combined_demand_PGE, combined_price_PGE, 480, 500, 16, 21, "PGE")
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
# rt_pricer = RTPricer(combined_demand_SCE, combined_price_SCE, 375, 545, 16, 21, "SCE")
# combined_price_SCE_new, adj_factor_SCE = rt_pricer.calculate_rt_price()
# # rt_pricer.plot_histogram()
# combined_price_SCE_new["INTERVALSTARTTIME_PST"] = pd.to_datetime(combined_price_SCE_new["INTERVALSTARTTIME_GMT"]).dt.tz_convert('America/Los_Angeles')
# combined_price_SCE_new['hour_of_year_start'] = combined_price_SCE_new['INTERVALSTARTTIME_PST'].apply(lambda x: ((x.dayofyear - 1) * 24 + x.hour))
# combined_price_SCE_new = combined_price_SCE_new.sort_values(by="INTERVALSTARTTIME_PST").reset_index(drop=True)
# combined_price_SCE_average = combined_price_SCE_new[["hour_of_year_start", "rt_price"]]
# combined_price_SCE_average = combined_price_SCE_average.groupby("hour_of_year_start")["rt_price"].mean()
# combined_price_SCE_average = pd.concat([combined_price_SCE_average, combined_price_SCE_average, combined_price_SCE_average], axis=0).reset_index(drop=True).to_dict()
#
#
# rt_pricer = RTPricer(combined_demand_SDGE, combined_price_SDGE, 481, 561, 16, 21, "SDGE")
# combined_price_SDGE_new, adj_factor_SDGE = rt_pricer.calculate_rt_price()
# # rt_pricer.plot_histogram()
# combined_price_SDGE_new["INTERVALSTARTTIME_PST"] = pd.to_datetime(combined_price_SDGE_new["INTERVALSTARTTIME_GMT"]).dt.tz_convert('America/Los_Angeles')
# combined_price_SDGE_new['hour_of_year_start'] = combined_price_SDGE_new['INTERVALSTARTTIME_PST'].apply(lambda x: ((x.dayofyear - 1) * 24 + x.hour))
# combined_price_SDGE_new = combined_price_SDGE_new.sort_values(by="INTERVALSTARTTIME_PST").reset_index(drop=True)
# combined_price_SDGE_average = combined_price_SDGE_new[["hour_of_year_start", "rt_price"]]
# combined_price_SDGE_average = combined_price_SDGE_average.groupby("hour_of_year_start")["rt_price"].mean()
# combined_price_SDGE_average = pd.concat([combined_price_SDGE_average, combined_price_SDGE_average, combined_price_SDGE_average], axis=0).reset_index(drop=True).to_dict()
#
#
# rt_pricer = RTPricer(combined_demand_SMUD, combined_price_SMUD, 212, 270, 17, 20, "SMUD")
# combined_price_SMUD_new, adj_factor_SMUD = rt_pricer.calculate_rt_price()
# # rt_pricer.plot_histogram()
# combined_price_SMUD_new["INTERVALSTARTTIME_PST"] = pd.to_datetime(combined_price_SMUD_new["INTERVALSTARTTIME_GMT"]).dt.tz_convert('America/Los_Angeles')
# combined_price_SMUD_new['hour_of_year_start'] = combined_price_SMUD_new['INTERVALSTARTTIME_PST'].apply(lambda x: ((x.dayofyear - 1) * 24 + x.hour))
# combined_price_SMUD_new = combined_price_SMUD_new.sort_values(by="INTERVALSTARTTIME_PST").reset_index(drop=True)
# combined_price_SMUD_average = combined_price_SMUD_new[["hour_of_year_start", "rt_price"]]
# combined_price_SMUD_average = combined_price_SMUD_average.groupby("hour_of_year_start")["rt_price"].mean()
# combined_price_SMUD_average = pd.concat([combined_price_SMUD_average, combined_price_SMUD_average, combined_price_SMUD_average], axis=0).reset_index(drop=True).to_dict()
#
# draw_RT(combined_price_PGE_average)
# draw_RT(combined_price_SCE_average)
# draw_RT(combined_price_SDGE_average)
# draw_RT(combined_price_SMUD_average)
#
# # %%
# with open("combined_price_PGE_average.json", "w") as json_file:
#     json.dump(combined_price_PGE_average, json_file)
#
# with open("combined_price_SCE_average.json", "w") as json_file:
#     json.dump(combined_price_SCE_average, json_file)
#
# with open("combined_price_SDGE_average.json", "w") as json_file:
#     json.dump(combined_price_SDGE_average, json_file)
#
# with open("combined_price_SMUD_average.json", "w") as json_file:
#     json.dump(combined_price_SMUD_average, json_file)


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

def price(weekday_prices, weekend_prices, weekday_times, weekend_times, month_ranges):
    """
    Adjust TOU pricing to include multiple disjoint time periods for peak and mid-peak,
    with separate prices for weekdays and weekends.

    Arguments:
    - weekday_prices: Dictionary with 'summer' and 'winter' keys each containing 'peak', 'mid_peak', 'off_peak' prices.
    - weekend_prices: Same structure as weekday_prices, but for weekends.
    - weekday_times: Dictionary with 'summer' and 'winter' keys each containing lists of tuples for 'peak', 'mid_peak'.
    - weekend_times: Same structure as weekday_times, but for weekends.
    - month_ranges: Dictionary with 'summer' and 'winter' keys indicating month ranges.
    """
    # Initialize the dictionary to hold TOU prices for each hour of the year
    tou_prices = {}

    # Generate TOU prices for each hour of the year
    for hour in range(26280):
        hour_str = str(hour)
        month = (hour // 720) % 12 + 1  # Calculate month (1-12)
        hour_of_day = hour % 24
        day_of_year = hour // 24
        day_of_week = (day_of_year + 5) % 7  # Adding 5 because 1st January 1900 was a Monday (day 0)

        # Determine if it's summer or winter
        is_summer = month_ranges['summer'][0] <= month <= month_ranges['summer'][1]
        season = 'summer' if is_summer else 'winter'

        # Determine if it's a weekend or a weekday
        is_weekend = day_of_week >= 5

        # Select appropriate prices and times based on day type
        prices = weekend_prices[season] if is_weekend else weekday_prices[season]
        times = weekend_times[season] if is_weekend else weekday_times[season]

        # Helper function to check if the current hour is within any given time ranges
        def in_time_ranges(ranges):
            return any(start <= hour_of_day < end for start, end in ranges)

        # Determine price based on time of day
        if in_time_ranges(times['peak']):
            tou_prices[hour_str] = prices['peak']
        elif in_time_ranges(times['mid_peak']):
            tou_prices[hour_str] = prices['mid_peak']
        else:
            tou_prices[hour_str] = prices['off_peak']

    return tou_prices



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


def get_utility_prices(region):
    # Load RT rate data from the JSON file
    with open(f"combined_price_{region}_average.json", "r") as json_file:
        rt_rate_data = json.load(json_file)

    # Access TOU and EV rate data from the utility_data dictionary
    tou_data = utility_data[region]['TOU']
    evr_data = utility_data[region]['EVR']

    # Function to convert dictionary keys to integers
    def convert_keys(data):
        return {int(key): value for key, value in data.items()}

    # Calculate TOU prices
    tou_prices = price(
        tou_data['weekday_prices'], tou_data['weekend_prices'],
        tou_data['weekday_times'], tou_data['weekend_times'],
        tou_data['month_ranges']
    )

    # Calculate EV rate prices
    ev_rate_prices = price(
        evr_data['weekday_prices'], evr_data['weekend_prices'],
        evr_data['weekday_times'], evr_data['weekend_times'],
        evr_data['month_ranges']
    )

    # Calculate EV rate prices
    commercial_prices = price(
        evr_data['weekday_prices'], evr_data['weekend_prices'],
        evr_data['weekday_times'], evr_data['weekend_times'],
        evr_data['month_ranges']
    )

    # Convert keys for RT rate, TOU prices, and EV rate prices
    rt_rate = convert_keys(rt_rate_data)
    tou_prices = {int(key): value for key, value in tou_prices.items()}
    ev_rate_prices = {int(key): value for key, value in ev_rate_prices.items()}
    commercial_prices = {int(key): value for key, value in commercial_prices.items()}

    return rt_rate, tou_prices, ev_rate_prices, commercial_prices
