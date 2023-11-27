from rc import (read_clean, read_time_series, split_func, drawing_data, plot_py, plot_autocorrelation1, plot_autocorrelation2,
                plot_autocorrelation3, plot_autocorrelation_month, plot_autocorrelation_season)
import glob

vehicle_names = ["P_1091", "P_1092", "P_1093", "P_1094", "P_1094", "P_1098", "P_1100", "P_1109", "P_1111", "P_1112", "P_1122",
                 "P_1352", "P_1353", "P_1357", "P_1367", "P_1368", "P_1370", "P_1371", "P_1376", "P_1381", "P_1384", "P_1384", "P_1388"]




for vehicle_name in vehicle_names:

    all_files = glob.glob(f'D:\\LSTM\\Data\\{vehicle_name}\\*.csv')

    df_timeseries1, df_timeseries2 = read_time_series(all_files)
    df_timeseries_clean, df_full_trips = read_clean(df_timeseries2, vehicle_name)
    charging_sessions, charging_sessions_tesla, charging_sessions_Bolt = split_func(df_timeseries_clean)

    # lag_with_max_autocorr1 = plot_autocorrelation1(df_full_trips, 21, vehicle_name)

    lag_with_max_autocorr2 = plot_autocorrelation2(df_full_trips, 21, vehicle_name)
    lag_with_max_autocorr4 = plot_autocorrelation_month(df_full_trips, 15, vehicle_name)
    lag_with_max_autocorr5 = plot_autocorrelation_season(df_full_trips, 21, vehicle_name)

    # plot_autocorrelation3(df_full_trips, lag_with_max_autocorr)
    # filtered_data = drawing_data(df_full_trips)
    # plot_py(filtered_data)