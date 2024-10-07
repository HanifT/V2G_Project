import os
import pandas as pd
import zipfile
import json
# %% Define the directory where your CSV files are located
folder_path = '/Users/haniftayarani/V2G_Project/real_time_price'

# Initialize an empty list to store the dataframes
dataframes_list = []

# Loop through all the files in the directory
for file_name in os.listdir(folder_path):
    # Check if the file is a CSV
    if file_name.endswith('.csv'):
        # Construct full file path
        file_path = os.path.join(folder_path, file_name)
        # Read the CSV file and store it in the list
        df = pd.read_csv(file_path)
        dataframes_list.append(df)

# Concatenate all the dataframes in the list
combined_price = pd.concat(dataframes_list, ignore_index=True)
combined_price = combined_price[combined_price["LMP_TYPE"] == "LMP"]
combined_price = combined_price.drop(columns=["NODE_ID_XML", "NODE", "PNODE_RESMRID", "POS", "OPR_INTERVAL", "MARKET_RUN_ID", "XML_DATA_ITEM", "GRP_TYPE", "OPR_DT"])

# Assuming your dataframe is named 'df'
pge_values = ['PGCC', 'PGEB', 'PGF1', 'PGFG', 'PGHB', 'PGKN', 'PGLP', 'PGNB', 'PGNC', 'PGNP', 'PGNV', 'PGP2', 'PGSA', 'PGSB', 'PGSF', 'PGSI', 'PGSN', 'PGST', 'PGZP']
sce_values = ['SCEC', 'SCEN', 'SCEW', 'SCHD', 'SCLD', 'SCNW']
sdge_values = ['SDG1']

# Filter rows where the first four letters of 'NODE_ID' values are in the list
combined_price_PGE = combined_price[combined_price['NODE_ID'].str[:4].isin(pge_values)]
combined_price_SCE = combined_price[combined_price['NODE_ID'].str[:4].isin(sce_values)]
combined_price_SDGE = combined_price[combined_price['NODE_ID'].str[:4].isin(sdge_values)]

lengths_dict = {
    'PGE': len(combined_price_PGE),
    'SCE': len(combined_price_SCE),
    'SDGE': len(combined_price_SDGE)
}

# %%
# Path to the folder containing the zip file
folder_path = '/Users/haniftayarani/V2G_Project/demand'

# Iterate over all files in the folder
for file_name in os.listdir(folder_path):
    # Check if the file is a zip file
    if file_name.endswith('.zip'):
        # Unzip the file
        with zipfile.ZipFile(os.path.join(folder_path, file_name), 'r') as zip_ref:
            zip_ref.extractall(folder_path)

# Get a list of all CSV files in the extracted folder
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

# Read each CSV file and concatenate them into one DataFrame
dfs = []
for csv_file in csv_files:
    df = pd.read_csv(os.path.join(folder_path, csv_file))
    dfs.append(df)

combined_demand = pd.concat(dfs, ignore_index=True)
combined_demand = combined_demand.drop(columns=["LOAD_TYPE", "OPR_DT", "OPR_INTERVAL", "MARKET_RUN_ID", "LABEL", "MARKET_RUN_ID", "POS"])

combined_demand_PGE = combined_demand[combined_demand["TAC_AREA_NAME"] == "PGE-TAC"].reset_index(drop=True)
combined_demand_PGE = combined_demand_PGE.drop_duplicates(subset=["INTERVALSTARTTIME_GMT"])

combined_demand_SCE = combined_demand[combined_demand["TAC_AREA_NAME"] == "SCE-TAC"].reset_index(drop=True)
combined_demand_SCE = combined_demand_SCE.drop_duplicates(subset=["INTERVALSTARTTIME_GMT"])

combined_demand_SDGE = combined_demand[combined_demand["TAC_AREA_NAME"] == "SDGE-TAC"].reset_index(drop=True)
combined_demand_SDGE = combined_demand_SDGE.drop_duplicates(subset=["INTERVALSTARTTIME_GMT"])

