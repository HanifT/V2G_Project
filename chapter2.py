from rc import (read_clean, read_time_series, data_departure )
import glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from keras.models import Sequential
from keras.layers import LSTM, RepeatVector, TimeDistributed,Dense, Dropout
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from keras import regularizers
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from keras.metrics import MeanSquaredError, RootMeanSquaredError, MeanAbsoluteError
from sklearn.utils import resample
import matplotlib.pyplot as plt

vehicle_names = ["P_1092"]


for vehicle_name in vehicle_names:

    all_files = glob.glob(f'D:\\LSTM\\Data\\{vehicle_name}\\*.csv')

    df_timeseries1, df_timeseries2 = read_time_series(all_files)
    df_timeseries_clean, df_full_trips = read_clean(df_timeseries2, vehicle_name)
    departure_input = data_departure(df_timeseries_clean)


# Assuming your dataset is stored in a DataFrame named 'departure_input'
# Specify the possible categories for each column
possible_categories = {
    0: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],  # Months
    1: ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],  # Days of the week
    8: ['Home', 'Work', 'Other']  # Destinations
}

# Create a copy of the original DataFrame
departure_input_encoded = departure_input.copy()

# Apply one-hot encoding to each specified column
for col, categories in possible_categories.items():
    if col in departure_input.columns:
        # Create a subset DataFrame with the current column
        subset_df = departure_input[[col]]

        # Explicitly specify the categories for the current column
        encoder = OneHotEncoder(categories=[categories], drop='first', sparse=False)

        # Apply one-hot encoding to the subset
        encoded_columns = encoder.fit_transform(subset_df)
        encoded_df = pd.DataFrame(encoded_columns, columns=encoder.get_feature_names_out([str(col)]))

        # Concatenate the encoded columns with the original DataFrame
        departure_input_encoded = pd.concat([departure_input_encoded, encoded_df], axis=1)

        # Drop the original column that has been one-hot encoded
        departure_input_encoded = departure_input_encoded.drop(columns=[col])

columns_to_encode = [0, 1, 8]
# Create a subset DataFrame with the specified columns
subset_df = departure_input_encoded.iloc[:, columns_to_encode]

# Apply one-hot encoding to the subset
encoder = OneHotEncoder(drop='first', sparse=False)
encoded_columns = encoder.fit_transform(subset_df)
encoded_df = pd.DataFrame(encoded_columns, columns=encoder.get_feature_names_out(subset_df.columns))

# Concatenate the encoded columns with the original DataFrame
df_encoded = pd.concat([departure_input, encoded_df], axis=1)

# Drop the original columns that have been one-hot encoded
df_encoded = df_encoded.drop(columns=subset_df.columns)

# Get the index of the column 'A'
col_a_index = df_encoded.columns.get_loc('next_departure')

# Move the column to the end
df_encoded = df_encoded[[col for col in df_encoded.columns if col != 'next_departure'] + ['next_departure']]

# Split the data into features and target
X = df_encoded.drop(columns=['next_departure'])

############################################################################################################
# Step 1
############################################################################################################
df_encoded = df_encoded.astype(np.float32)
np.random.seed(123)
# set parameters
time_steps = 10
features = (X.shape[1])
batch_size = 300
epochs_per_iteration = 50


# create a new dataframe to store the denormalized values
scaler = RobustScaler()
df_encoded_norm = scaler.fit_transform(df_encoded)

df_encoded_norm = pd.DataFrame(df_encoded_norm)
df_encoded_norm["next_departure"] = df_encoded["next_departure"]
# split the data into 100 subsets
data_splits_dd = np.array_split(df_encoded_norm, 365)

# define and fit the initial model with the first subset
initial_data_dd = data_splits_dd[0].iloc[:len(data_splits_dd[0])]
initial_generator = TimeseriesGenerator(
    data=initial_data_dd.iloc[:, :features].values,
    targets=initial_data_dd.iloc[:, -1:].values,
    length=time_steps,
    batch_size=len(initial_data_dd)
)
initial_x_dd, initial_y_dd = initial_generator[0]

model6 = Sequential()
model6.add((LSTM(50, input_shape=(time_steps, features), kernel_regularizer=regularizers.l2(0.001))))
model6.add(Dropout(0.5))
model6.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model6.add(Dropout(0.5))
model6.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model6.add(Dropout(0.5))
model6.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model6.add(Dropout(0.5))
model6.add(Dense(1, activation='linear'))
model6.compile(optimizer='adam', loss='mse', metrics=[MeanSquaredError(), RootMeanSquaredError(), MeanAbsoluteError()])

model6.fit(initial_x_dd, initial_y_dd, epochs=epochs_per_iteration, batch_size=batch_size, verbose=1)

# save the weights of the initial model6

model6.save_weights('G:/My Drive/PycharmProjects/initial_weights_dd.h5')
results_df_dd = pd.DataFrame(columns=['iteration', 'Test_mse', 'Test_rmse', 'Test_mae'])

test_loss, test_auc, test_pre, test_rec = model6.evaluate(initial_x_dd, initial_y_dd)

results_df_dd = pd.concat([results_df_dd, pd.DataFrame({'iteration': 0,
                                                        'Test_mse': [test_auc],
                                                        'Test_rmse': [test_pre],
                                                        'Test_mae': [test_rec]})], ignore_index=True)

# Perform bootstrapping
n_bootstraps = 1
bootstrapped_predictions = np.zeros((n_bootstraps, initial_x_dd.shape[0]))
for i in range(n_bootstraps):
    # Resample data
    x_resampled, y_resampled = resample(initial_x_dd, initial_y_dd)
    # Make predictions on resampled data
    bootstrapped_predictions[i] = model6.predict(x_resampled).flatten()
    print(i)
# Calculate confidence interval
conf_int_full = pd.DataFrame()
conf_int = np.percentile(bootstrapped_predictions, [30, 70], axis=0)
conf_int_full = pd.concat([conf_int_full, pd.DataFrame(conf_int).T], ignore_index=True)

dd_prediction_full = pd.DataFrame()
dd_prediction = model6.predict(initial_x_dd)
dd_prediction_full = pd.concat([dd_prediction_full, pd.DataFrame(dd_prediction)], ignore_index=True)


model6 = Sequential()
model6.add((LSTM(50, input_shape=(time_steps, features), kernel_regularizer=regularizers.l2(0.001))))
model6.add(Dropout(0.5))
model6.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model6.add(Dropout(0.5))
model6.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model6.add(Dropout(0.5))
model6.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model6.add(Dropout(0.5))
model6.add(Dense(1, activation='linear'))
model6.compile(optimizer='adam', loss='mse', metrics=[MeanSquaredError(), RootMeanSquaredError(), MeanAbsoluteError()])

model6.fit(initial_x_dd, initial_y_dd, epochs=epochs_per_iteration, batch_size=batch_size, verbose=1)

# save the weights of the initial model6

model6.save_weights('G:/My Drive/PycharmProjects/initial_weights_dd.h5')
results_df_dd = pd.DataFrame(columns=['iteration', 'Test_mse', 'Test_rmse', 'Test_mae'])

test_loss, test_auc, test_pre, test_rec = model6.evaluate(initial_x_dd, initial_y_dd)

results_df_dd = pd.concat([results_df_dd, pd.DataFrame({'iteration': 0,
                                                        'Test_mse': [test_auc],
                                                        'Test_rmse': [test_pre],
                                                        'Test_mae': [test_rec]})], ignore_index=True)

# Perform bootstrapping
bootstrapped_predictions = np.zeros((n_bootstraps, initial_x_dd.shape[0]))
for i in range(n_bootstraps):
    # Resample data
    x_resampled, y_resampled = resample(initial_x_dd, initial_y_dd)
    # Make predictions on resampled data
    bootstrapped_predictions[i] = model6.predict(x_resampled).flatten()
    print(i)
# Calculate confidence interval
conf_int_full = pd.DataFrame()
conf_int = np.percentile(bootstrapped_predictions, [30, 70], axis=0)
conf_int_full = pd.concat([conf_int_full, pd.DataFrame(conf_int).T], ignore_index=True)

dd_prediction_full = pd.DataFrame()
dd_prediction = model6.predict(initial_x_dd)
dd_prediction_full = pd.concat([dd_prediction_full, pd.DataFrame(dd_prediction)], ignore_index=True)
############################################################################################################
# Step 2
############################################################################################################
# define and fit the initial model with the first subset
initial_data_dd1 = data_splits_dd[1].iloc[:len(data_splits_dd[1])]
initial_generator_dd = TimeseriesGenerator(
    data=initial_data_dd1.iloc[:, :features].values,
    targets=initial_data_dd1.iloc[:, -1:].values,
    length=time_steps,
    batch_size=len(initial_data_dd1)
)
initial_x_dd1, initial_y_dd1 = initial_generator_dd[0]
model6.load_weights('G:/My Drive/PycharmProjects/initial_weights_dd.h5')
model6.fit(initial_x_dd1, initial_y_dd1, epochs=epochs_per_iteration, batch_size=batch_size, verbose=1)

test_loss, test_auc, test_pre, test_rec = model6.evaluate(initial_x_dd1, initial_y_dd1)

results_df_dd = pd.concat([results_df_dd, pd.DataFrame({'iteration': 1,
                                                        'Test_mse': [test_auc],
                                                        'Test_rmse': [test_pre],
                                                        'Test_mae': [test_rec]})], ignore_index=True)

bootstrapped_predictions = np.zeros((n_bootstraps, initial_x_dd1.shape[0]))
for i in range(n_bootstraps):
    # Resample data
    x_resampled, y_resampled = resample(initial_x_dd1, initial_y_dd1)
    # Make predictions on resampled data
    bootstrapped_predictions[i] = model6.predict(x_resampled).flatten()
    print(i)
# Calculate confidence interval
conf_int = np.percentile(bootstrapped_predictions, [30, 70], axis=0)
conf_int_full = pd.concat([conf_int_full, pd.DataFrame(conf_int).T], ignore_index=True)

dd_prediction = model6.predict(initial_x_dd1)
dd_prediction_full = pd.concat([dd_prediction_full, pd.DataFrame(dd_prediction)], ignore_index=True)
model6.save_weights('G:/My Drive/PycharmProjects/initial_weights_dd.h5')
train_data_dd = initial_data_dd1
############################################################################################################
# final step
############################################################################################################
# loop over the data subsets and train/test the model6 iteratively

for i in range(1, (len(data_splits_dd)-1)):
    print(f"Iteration {i}...")
    train_data_dd = pd.concat([data_splits_dd[i - 1], data_splits_dd[i]])
    test_data_dd = data_splits_dd[i + 1]

    train_generator = TimeseriesGenerator(
        data=train_data_dd.iloc[:, :features].values,
        targets=train_data_dd.iloc[:, -1:].values,
        length=time_steps,
        batch_size=len(train_data_dd)
    )
    train_x_dd, train_y_dd = train_generator[0]
    model6.load_weights('G:/My Drive/PycharmProjects/initial_weights_dd.h5')
    model6.fit(train_x_dd, train_y_dd, epochs=epochs_per_iteration, batch_size=batch_size, verbose=1)

    test_generator = TimeseriesGenerator(
        data=test_data_dd.iloc[:, :features].values,
        targets=test_data_dd.iloc[:, -1:].values,
        length=time_steps,
        batch_size=len(test_data_dd)
    )
    test_x_dd, test_y_dd = test_generator[0]

    Test_loss, Test_mse, Test_rmse, Test_mae = model6.evaluate(test_x_dd, test_y_dd)
    print(f"Test loss: {Test_loss:.4f}, Test MSE: {Test_mse:.4f}, Test RMSE: {Test_rmse:.4f}, Test MAE: {Test_mae:.4f}")
    print(i)
    # append the results to the DataFrame
    results_df_dd = pd.concat([results_df_dd, pd.DataFrame({'iteration': [i + 1],
                                                            'Test_mse': Test_mse,
                                                            'Test_rmse': Test_rmse,
                                                            'Test_mae': Test_mae})], ignore_index=True)

    bootstrapped_predictions = np.zeros((n_bootstraps, test_x_dd.shape[0]))
    for k in range(n_bootstraps):
        # Resample data
        x_resampled, y_resampled = resample(test_x_dd, test_y_dd)
        # Make predictions on resampled data
        bootstrapped_predictions[k] = model6.predict(x_resampled).flatten()
        print(k)
    # Calculate confidence interval
    conf_int = np.percentile(bootstrapped_predictions, [30, 70], axis=0)
    conf_int_full = pd.concat([conf_int_full, pd.DataFrame(conf_int).T], ignore_index=True)

    dd_prediction = model6.predict(test_x_dd)
    dd_prediction_full = pd.concat([dd_prediction_full, pd.DataFrame(dd_prediction)], ignore_index=True)
    model6.save_weights('G:/My Drive/PycharmProjects/initial_weights_dd.h5')


############################################################################################################
# test
############################################################################################################
test_generator = TimeseriesGenerator(
        data=df_encoded_norm.iloc[:, :features].values,
        targets=df_encoded_norm.iloc[:, -1:].values,
        length=time_steps,
        batch_size=len(df_encoded_norm))
test_x, test_y = test_generator[0]

test3_1 = pd.DataFrame(test_y)

subsets = np.array_split(test3_1, 365)

# remove first ten rows of each subset
for i in range(len(subsets)-1):
    subsets[i] = subsets[i].iloc[10:]

# combine subsets into new dataframe
test3_1n = pd.concat(subsets).reset_index(drop=True)

subsets = np.array_split(df_encoded, 365)

# remove first ten rows of each subset
for i in range(len(subsets)-1):
    subsets[i] = subsets[i].iloc[10:]

# combine subsets into new dataframe
df_encoded_new = pd.concat(subsets).reset_index(drop=True)

df_encoded_new = df_encoded_new.iloc[10:]
df_encoded_new = df_encoded_new.reset_index(drop=True)

final = pd.DataFrame({"test": test3_1n[0], "prediction": dd_prediction_full[0]})
# Save DataFrame to CSV file
# final.to_csv('your_file11.csv', index=False)
#
# # Read DataFrame from CSV file
# final = pd.read_csv('your_file.csv')



# Keep rows where the value in the "test" column changes
final['change_index'] = (final['test'] != final['test'].shift(1)).cumsum()

# Drop rows with NA values
final_changed_cleaned = final.dropna()
final_changed_cleaned = final_changed_cleaned.groupby('change_index').tail(n=1)
final_changed_cleaned = final_changed_cleaned.reset_index(drop=True)
# final_changed_cleaned = final_changed_cleaned[0]

# Set a larger figure size
plt.figure(figsize=(25, 25))  # Adjust the figure size as needed

# Create subplots in a 3x3 grid
for i in range(1, 10):
    plt.subplot(3, 3, i)

    # Determine the start and end indices for each subset of 100 samples
    start_index = (i - 1) * 100
    end_index = i * 100

    # Plot the two lines for the subset
    plt.plot(final_changed_cleaned['test'][start_index:end_index], label='Actual')
    plt.plot(final_changed_cleaned['prediction'][start_index:end_index], label='Prediction')

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



