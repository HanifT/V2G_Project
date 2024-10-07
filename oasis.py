import requests
import pandas as pd
import zipfile
import io
# %%
# Define the base URL
base_url = "http://oasis.caiso.com/oasisapi/SingleZip"

# Define parameters for the API request
params = {
    "resultformat": "6",
    "queryname": "PRC_LMP",
    "version": "12",
    "market_run_id": "DAM",
    "grp_type": "ALL_APNODES"
}

# Define the start and end dates
start_date = pd.Timestamp("2023-01-01")
end_date = pd.Timestamp("2023-12-31")

# Loop over each day and download the data
for date in pd.date_range(start_date, end_date):
    # Construct the datetime string for the API request
    datetime_str = date.strftime("%Y%m%dT08:00-0000")

    # Add the startdatetime and enddatetime parameters to the request
    params["startdatetime"] = datetime_str
    params["enddatetime"] = (date + pd.Timedelta(days=1)).strftime("%Y%m%dT08:00-0000")

    # Make the API request
    response = requests.get(base_url, params=params)

    # Save the response content as a ZIP file
    with open(f"price_data_{date.strftime('%Y%m%d')}.zip", "wb") as f:
        f.write(response.content)

    # Extract the CSV file from the ZIP file
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        # Find the CSV file in the ZIP file
        csv_filename = [name for name in z.namelist() if name.endswith('.csv')][0]

        # Extract the CSV file
        with z.open(csv_filename) as csv_file:
            # Save the extracted CSV file
            with open(f"price_data_{date.strftime('%Y%m%d')}.csv", "wb") as csv_out:
                csv_out.write(csv_file.read())
