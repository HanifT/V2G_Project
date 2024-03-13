from read_clean_functions import (read_clean, read_time_series, data_departure, r_sq)
import glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from math import sqrt

########################################################################################################################
########################################################################################################################
vehicle_names = ["P_1092"]


for vehicle_name in vehicle_names:

    all_files = glob.glob(f'D:\\LSTM\\Data\\{vehicle_name}\\*.csv')

    df_timeseries1, df_timeseries2 = read_time_series(all_files)
    df_timeseries_clean, df_trips, df_full_trips = read_clean(df_timeseries2, vehicle_name)
    departure_input = data_departure(df_timeseries_clean)

########################################################################################################################
########################################################################################################################
# Start saving on Git
def input_data(df):

    df_v1 = df[['start_time_ (local)', 'end_time_ (local)', "month", "day", "day_name", "day_type", 'hour', "distance", "speed[average]", "duration_trip", "Destination_label", "Origin_label", "new_Destination",
                'energy[net]', 'energy[consumption]', 'battery[soc][start][trip]', 'battery[soc][end][trip]', 'energy[charge_type][type]', 'Lat', 'Long']].copy()
    df_v1.loc[:, "year"] = pd.to_datetime(df_v1["start_time_ (local)"]).dt.year
    df_v1.loc[:, "month"] = pd.to_datetime(df_v1["start_time_ (local)"]).dt.month
    df_v1.loc[:, "day"] = pd.to_datetime(df_v1["start_time_ (local)"]).dt.day
    df_v1.loc[:, "start_time_hour"] = pd.to_datetime(df_v1["start_time_ (local)"]).dt.hour
    df_v1.loc[:, "start_time_minute"] = pd.to_datetime(df_v1["start_time_ (local)"]).dt.minute
    df_v1.loc[:, "end_time_hour"] = pd.to_datetime(df_v1["end_time_ (local)"]).dt.hour
    df_v1.loc[:, "end_time_minute"] = pd.to_datetime(df_v1["end_time_ (local)"]).dt.minute
    df_v1 = df_v1.sort_values(by=["year", "month", "day", "start_time_hour", "start_time_minute"], ascending=True)

    df_v1["next_departure_time"] = df_v1["start_time_hour"].shift(-1).fillna(0).astype(int)
    df_v1["new_origin"] = df_v1["new_Destination"].shift(-1).fillna(0)
    df_v1["charge_type_count"] = (df_v1["energy[charge_type][type]"] != 'NA').cumsum()
    df_v1["distance_from_charge"] = df_v1.groupby("charge_type_count")["distance"].cumsum()
    df_v1 = df_v1.sort_values(by=["year", "month", "day", "start_time_hour", "start_time_minute"], ascending=True)

    df_v1 = df_v1[["year", "month", "day", 'start_time_hour', 'start_time_minute', 'end_time_hour',  "day_name", "day_type", "distance", "speed[average]", "duration_trip", "Destination_label", "Origin_label", "new_Destination",
                'energy[net]', 'energy[consumption]', 'battery[soc][start][trip]', 'battery[soc][end][trip]', 'energy[charge_type][type]', 'Lat', 'Long', "distance_from_charge", "new_origin", "next_departure_time"]].copy()

    return df_v1

df_final = input_data(df_full_trips)

input_model_data = df_final[["start_time_hour", "end_time_hour", "distance", "distance_from_charge", "speed[average]",
                             "duration_trip", "battery[soc][start][trip]", "battery[soc][end][trip]", "new_Destination", "new_origin", "day_name", "month",
                             "next_departure_time"]]

input_model_data = df_final[["start_time_hour", "end_time_hour", "distance", "distance_from_charge", "speed[average]",
                             "duration_trip", "battery[soc][start][trip]", "battery[soc][end][trip]", "Destination_label", "Origin_label", "day_name", "month",
                             "next_departure_time"]]
input_model_data = input_model_data.dropna()
# Identify categorical columns
categorical_columns = ["Destination_label", "Origin_label", "day_name", "month"] # Create a ColumnTransformer to apply one-hot encoding to categorical columns
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first', sparse=False), categorical_columns)
    ],
    remainder='passthrough'  # Include non-categorical columns as they are
)

# Create a pipeline with preprocessing and linear regression steps
pipeline = Pipeline([
    ('preprocessor', preprocessor),
])

# Apply preprocessing
input_model_data_encoded = pipeline.fit_transform(input_model_data)


# Split the data into features and target variable
X_l = input_model_data_encoded[:, :-1]  # Features

y_l = input_model_data_encoded[:, -1]   # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_l, y_l, test_size=0.1,  shuffle=False)

########################################################################################################################
########################################################################################################################
# Create and train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)
final_lr = pd.DataFrame({"test": y_test, "prediction": predictions})
# Evaluate the model
mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
rmse = sqrt(mean_squared_error(y_test, predictions))
print(f'Mean Squared Error: {mse}')
print(f'Root Mean Squared Error: {rmse}')
print(f'Mean absolute Error: {mae}')


import statsmodels.api as sm
import pandas as pd

# Add a constant term to the independent variables matrix
X_train_with_const = sm.add_constant(X_train)

# Create and fit the OLS model
ols_model = sm.OLS(y_train, X_train_with_const).fit()

# Get the summary of the regression
print(ols_model.summary())
########################################################################################################################
########################################################################################################################
# Apply polynomial features transformation
degree = 2  # You can adjust the degree as needed
poly = PolynomialFeatures(degree=degree)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)
X_l_poly = poly.transform(X_l)

# Create and train a polynomial regression model
poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)

# Make predictions on the entire dataset
predictions_poly = poly_model.predict(X_test_poly)

# Evaluate the polynomial regression model
mse_poly = mean_squared_error(y_test, predictions_poly)
mae_poly = mean_absolute_error(y_test, predictions_poly)
rmse_poly = sqrt(mean_squared_error(y_test, predictions_poly))
print(f'Mean Squared Error (Polynomial Regression): {mse_poly}')
print(f'Root Mean Squared Error (Polynomial Regression): {rmse_poly}')
print(f'Mean Absolute Error (Polynomial Regression): {mae_poly}')

# Optionally, you can also visualize the predictions
final_poly = pd.DataFrame({"test": y_test, "prediction": predictions_poly})

########################################################################################################################
########################################################################################################################
# Create and train an SVM regressor
param_grid = {'C': [0.1, 1, 10], 'gamma': [0.01, 0.1, 1]}
svm_model = GridSearchCV(SVR(kernel='rbf'), param_grid, cv=3)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

svm_model.fit(X_train_scaled, y_train)

# Make predictions on the entire dataset
predictions_svm = svm_model.predict(X_test_scaled)

# Evaluate the SVM model
mse_svm = mean_squared_error(y_test, predictions_svm)
mae_svm = mean_absolute_error(y_test, predictions_svm)
rmse_svm = sqrt(mean_squared_error(y_test, predictions_svm))
final_svm = pd.DataFrame({"test": y_test, "prediction": predictions_svm})
print(f'Mean Squared Error (SVM): {mse_svm}')
print(f'Root Mean Squared Error (SVM): {rmse_svm}')
print(f'Mean Absolute Error (SVM): {mae_svm}')
########################################################################################################################
########################################################################################################################
# Create and train a random forest regressor
forest_model = RandomForestRegressor()
forest_model.fit(X_train, y_train)

# Make predictions on the entire dataset
predictions_forest = forest_model.predict(X_test)

# Evaluate the random forest model
mse_forest = mean_squared_error(y_test, predictions_forest)
mae_forest = mean_absolute_error(y_test, predictions_forest)
rmse_forest = sqrt(mean_squared_error(y_test, predictions_forest))
final_rf = pd.DataFrame({"test": y_test, "prediction": predictions_forest})
print(f'Mean Squared Error (Random Forest): {mse_forest}')
print(f'Root Mean Squared Error (Random Forest): {rmse_forest}')
print(f'Mean Absolute Error (Random Forest): {mae_forest}')
########################################################################################################################
########################################################################################################################
# Create and train a decision tree regressor
tree_model = DecisionTreeRegressor(max_depth=5)
tree_model.fit(X_train, y_train)

# Make predictions on the entire dataset
predictions_tree = tree_model.predict(X_test)

# Evaluate the decision tree model
mse_tree = mean_squared_error(y_test, predictions_tree)
mae_tree = mean_absolute_error(y_test, predictions_tree)
rmse_tree = sqrt(mean_squared_error(y_test, predictions_tree))
final_dt = pd.DataFrame({"test": y_test, "prediction": predictions_tree})
print(f'Mean Squared Error (Decision Tree): {mse_tree}')
print(f'Root Mean Squared Error (Decision Tree): {rmse_tree}')
print(f'Mean Absolute Error (Decision Tree): {mae_tree}')
########################################################################################################################
########################################################################################################################
rsq_lr = r_sq(final_lr["test"], final_lr["prediction"])

rsq_poly = r_sq(final_poly["test"], final_poly["prediction"])

rsq_svm = r_sq(final_svm["test"], final_svm["prediction"])

rsq_rf = r_sq(final_rf["test"], final_rf["prediction"])

rsq_dt = r_sq(final_dt["test"], final_dt["prediction"])
########################################################################################################################
########################################################################################################################
# Set a larger figure size
plt.figure(figsize=(25, 25))  # Adjust the figure size as needed

# Create subplots in a 3x3 grid
for i in range(1, 3):
    plt.subplot(1, 2, i)

    # Determine the start and end indices for each subset of 100 samples
    start_index = (i - 1) * 100
    end_index = i * 100
    indices = np.arange(len(y_test[start_index:end_index]))
    # Plot the two lines for the subset
    plt.scatter(indices, y_test[start_index:end_index], label='Actual')
    plt.scatter(indices, predictions_forest[start_index:end_index], label='Prediction')

    # Add labels and title
    plt.xlabel('Trip ID')
    plt.ylabel('Hour')
    plt.title(f'Subset {i}')
    plt.ylim(bottom=0, top=24)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.subplots_adjust(top=0.92, bottom=0.08, hspace=0.4)
    # Add legend
    plt.legend()

# Adjust layout
plt.tight_layout()
# Adjust margin for titles
plt.subplots_adjust(top=0.92, bottom=0.08, hspace=0.4)
# Add overall title
plt.suptitle('Comparison of Actual and Forecasted Departure Time (Subsets of 100 Samples)', fontsize=16, fontweight='bold')

# Show the plot
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns


# Calculate residuals
residuals = y_test - predictions_forest

# Create a scatter plot of actual vs. predicted values
plt.figure(figsize=(10, 6))

# Set aspect ratio to be equal
plt.gca().set_aspect('equal', adjustable='box')

sns.scatterplot(x=y_test, y=predictions_forest, color='blue', alpha=0.7)

# Draw a line of best fit
sns.regplot(x=y_test, y=predictions_forest, scatter=False, color='red')

# Set labels and title with larger font size
plt.xlabel('Actual Departure Time', fontsize=20)
plt.ylabel('RF Predicted Values', fontsize=20)
plt.title('Best Linear Fit: y=0.30*x + 8.75\nR=0.30', fontsize=16)

# Set the same scale for both axes
min_val = min(min(y_test), min(predictions_forest))
max_val = max(max(y_test), max(predictions_forest))
interval = 2  # Set the interval as needed
ticks = range(int(min_val), int(max_val) + interval, interval)
plt.xticks(ticks, fontsize=15)
plt.yticks(ticks, fontsize=15)

# Show the plot
plt.show()


# Create and fit a linear regression model
linear_model = LinearRegression()
linear_model.fit(y_test.reshape(-1, 1), predictions)

# Access the coefficients and intercept
slope = linear_model.coef_[0]
intercept = linear_model.intercept_

print(f'Slope (Coefficient): {slope}')
print(f'Intercept: {intercept}')