from read_clean_functions import (read_clean, read_time_series, data_departure)
import glob
import pandas as pd
import numpy as np
from math import sqrt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.utils import resample
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
vehicle_names = ["P_1092"]


for vehicle_name in vehicle_names:

    all_files = glob.glob(f'D:\\LSTM\\Data\\{vehicle_name}\\*.csv')

    df_timeseries1, df_timeseries2 = read_time_series(all_files)
    df_timeseries_clean, df_trips, df_full_trips = read_clean(df_timeseries2, vehicle_name)
    departure_input = data_departure(df_timeseries_clean)


# Start saving on Git
def input_data(df):

    df_v1 = df[['start_time_ (local)', 'end_time_ (local)', "month", "day", "day_name", "day_type", 'hour', "distance", "speed[average]", "duration_trip", "Destination_label", "Origin_label", "new_Destination",
                'energy[net]', 'energy[consumption]', 'battery[soc][start][trip]', 'battery[soc][end][trip]', 'energy[charge_type][type]', 'Lat', 'Long']].copy()

    df_v1.loc[:, "start_time_hour"] = pd.to_datetime(df_v1["start_time_ (local)"]).dt.hour
    df_v1.loc[:, "start_time_minute"] = pd.to_datetime(df_v1["start_time_ (local)"]).dt.minute
    df_v1.loc[:, "end_time_hour"] = pd.to_datetime(df_v1["end_time_ (local)"]).dt.hour
    df_v1.loc[:, "end_time_minute"] = pd.to_datetime(df_v1["end_time_ (local)"]).dt.minute
    df_v1 = df_v1.sort_values(by=["start_time_ (local)"], ascending=False)
    df_v1["next_departure_time"] = df_v1["start_time_hour"].shift(-1).fillna(0).astype(int)
    df_v1["new_origin"] = df_v1["new_Destination"].shift(-1).fillna(0)
    df_v1["charge_type_count"] = (df_v1["energy[charge_type][type]"] != 'NA').cumsum()
    df_v1["distance_from_charge"] = df_v1.groupby("charge_type_count")["distance"].cumsum()
    df_v1 = df_v1.sort_values(by=["start_time_ (local)"], ascending=False).reset_index(drop=True)


    return df_v1

df_final = input_data(df_full_trips)

input_model_data = df_final[["start_time_hour", "end_time_hour", "distance", "distance_from_charge", "speed[average]",
                             "duration_trip", "battery[soc][start][trip]", "battery[soc][end][trip]", "new_Destination", "new_origin", "day_name", "month",
                             "next_departure_time"]]

# Factorize categorical columns
categorical_columns = ["new_Destination", "new_origin", "day_name", "month"]
for col in categorical_columns:
    input_model_data[col] = pd.factorize(input_model_data[col])[0]

# Split the data into features and target variable
X = input_model_data.drop("next_departure_time", axis=1)
y = input_model_data["next_departure_time"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,  shuffle=False)

# Create and train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X)

# Evaluate the model
mse = mean_squared_error(y, predictions)
mae = mean_absolute_error(y, predictions)
rmse = sqrt(mean_squared_error(y, predictions))
print(f'Mean Squared Error: {mse}')
print(f'Root Mean Squared Error: {rmse}')
print(f'Mean absolute Error: {mae}')


# Create an array of indices for plotting
indices = np.arange(len(y))

# Plot y_test and predictions in a line plot
plt.figure(figsize=(10, 6))
plt.scatter(indices, y, label='Actual Values (y_test)', marker='o')
plt.scatter(indices, predictions, label='Predicted Values', marker='o')
# Additional plot settings
plt.title('Actual vs Predicted Values (Scatter Plot)')
plt.xlabel('Data Points')
plt.ylabel('Values')
plt.legend()

# Add grid
plt.grid(True)

plt.show()