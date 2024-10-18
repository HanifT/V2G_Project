# %%
import pandas as pd
from parking import (clean_data, adjust_battery_size, charging_dataframe, charging_c_k, extra_extra_kwh, extra_extra_cycle, total_v2g_cap_graph, total_v2g_failt_graph, total_v2g_failc_graph, extra_extra_kwh_parking)
import warnings
warnings.filterwarnings("ignore")
##################################################################################################################
##################################################################################################################
# %%
# Section 1
# # Reading the raw data and clean it
vehicle_names = ["P_1352", "P_1353", "P_1357", "P_1367", "P_1368", "P_1370", "P_1371", "P_1376",
                 "P_1381", "P_1384", "P_1388", "P_1393", "P_1403", "P_1409", "P_1412", "P_1414",
                 "P_1419", "P_1421", "P_1422", "P_1423", "P_1424", "P_1427", "P_1429", "P_1435",
                 "P_1087", "P_1091", "P_1092", "P_1093", "P_1094", "P_1098", "P_1100", "P_1109",
                 "P_1111", "P_1112", "P_1122", "P_1123", "P_1125", "P_1125a", "P_1127", "P_1131",
                 "P_1132", "P_1135", "P_1137", "P_1140", "P_1141", "P_1143", "P_1144", "P_1217",
                 "P_1253", "P_1257", "P_1260", "P_1267", "P_1271", "P_1272", "P_1279", "P_1280",
                 "P_1281", "P_1285", "P_1288", "P_1294", "P_1295", "P_1296", "P_1304", "P_1307", "P_1375",
                 "P_1088a", "P_1122", "P_1264", "P_1267", "P_1276", "P_1289", "P_1290", "P_1300", "P_1319"]
final_dataframes = clean_data(vehicle_names)
final_dataframes["bat_cap"] = final_dataframes.apply(adjust_battery_size, axis=1)
final_dataframes = final_dataframes.drop(columns=['Model'])
# # # Saving cleaned data as a csv file
final_dataframes.to_csv("data.csv")
##################################################################################################################
##################################################################################################################
# %%
#
# total_v2g_cap = pd.DataFrame({"TOU Cycle - (No Limit)": df_summary_tou.iloc[:, 0], "TOU Cycle - (Next Trip Guarantee)": df_summary_tou_nt.iloc[:, 0], "TOU Cycle - (Next Charging Guarantee)": df_summary_tou_nc.iloc[:, 0], "TOU Cycle - (charging events + last parking session)": df_summary_tou_parking.iloc[:, 0], "Stationary-TOU": df_summary_tou_sta.iloc[:, 0]})
# total_v2g_cap = pd.DataFrame({"No Tier - (No Limit)": df_summary.iloc[:, 0],"TOU Cycle - (No Limit)": df_summary_tou.iloc[:, 0], "Stationary-TOU": df_summary_tou_sta.iloc[:, 0]})
#
#
# total_v2g_cap_tou = pd.DataFrame({"No limit": df_summary_tou.iloc[:, 0]/df_summary_tou_sta.iloc[:, 0], "Buffer %20":  df_summary_tou_20.iloc[:, 0]/df_summary_tou_sta.iloc[:, 0], "Buffer %30":  df_summary_tou_30.iloc[:, 0]/df_summary_tou_sta.iloc[:, 0],
#                                   "Buffer %40":  df_summary_tou_40.iloc[:, 0]/df_summary_tou_sta.iloc[:, 0], "Buffer %50":  df_summary_tou_50.iloc[:, 0]/df_summary_tou_sta.iloc[:, 0], "Buffer next trip":  df_summary_tou_nt.iloc[:, 0]/df_summary_tou_sta.iloc[:, 0],
#                                   "Buffer next charging":  df_summary_tou_nc.iloc[:, 0]/df_summary_tou_sta.iloc[:, 0], "All Events (Next Trip Guarantee)": df_summary_tou_parking_nt.iloc[:, 0]*1.01/df_summary_tou_sta.iloc[:, 0],
#                                   "All Events (Next Charging Guarantee)": df_summary_tou_parking_nc.iloc[:, 0]/df_summary_tou_sta.iloc[:, 0]})
#
# total_v2g_cap_tou = total_v2g_cap_tou*100
#
#
# total_v2g_cap_tou1 = pd.DataFrame({"No limit": df_summary_tou.iloc[:, 0], "Buffer %20":  df_summary_tou_20.iloc[:, 0], "Buffer %30":  df_summary_tou_30.iloc[:, 0],
#                                   "Buffer %40":  df_summary_tou_40.iloc[:, 0], "Buffer %50":  df_summary_tou_50.iloc[:, 0], "Buffer next trip":  df_summary_tou_nt.iloc[:, 0],
#                                   "Buffer next charging":  df_summary_tou_nc.iloc[:, 0], "All Events (Next Trip Guarantee)": df_summary_tou_parking_nt.iloc[:, 0]*1.01,
#                                   "All Events (Next Charging Guarantee)": df_summary_tou_parking_nc.iloc[:, 0]})
#
# total_v2g_cap_tou2 = pd.DataFrame({"No limit": df_summary_tou.iloc[:, 0], "Buffer %20":  df_summary_tou_20.iloc[:, 0], "Buffer %30":  df_summary_tou_30.iloc[:, 0],
#                                   "Buffer %40":  df_summary_tou_40.iloc[:, 0], "Buffer %50":  df_summary_tou_50.iloc[:, 0], "Buffer next trip":  df_summary_tou_nt.iloc[:, 0],
#                                   "Buffer next charging":  df_summary_tou_nc.iloc[:, 0]})
#
#
# total_v2g_cap_tou1 = total_v2g_cap_tou1
#
# total_v2g_fail_trip_tou = pd.DataFrame({"No limit": data_failure.iloc[:, 0], "Buffer %20":  data_failure_20.iloc[:, 0], "Buffer %30":  data_failure_30.iloc[:, 0], "Buffer %40":  data_failure_40.iloc[:, 0],
#                                        "Buffer %50":  data_failure_50.iloc[:, 0], "Buffer next trip":  data_failure_nt.iloc[:, 0], "Buffer next charging":  data_failure_nc.iloc[:, 0]})
#
# total_v2g_fail_trip_tou = 100 - total_v2g_fail_trip_tou
#
# total_v2g_fail_charging_tou = pd.DataFrame({"No limit": data_failure.iloc[:, 1], "Buffer %20":  data_failure_20.iloc[:, 1], "Buffer %30":  data_failure_30.iloc[:, 1], "Buffer %40":  data_failure_40.iloc[:, 1],
#                                             "Buffer %50":  data_failure_50.iloc[:, 1], "Buffer next trip":  data_failure_nt.iloc[:, 1], "Buffer next charging":  data_failure_nc.iloc[:, 1]})
#
# total_v2g_fail_charging_tou = 100 - total_v2g_fail_charging_tou
#
# total_v2g_fail_trip_tou.index = total_v2g_cap_tou.index
# total_v2g_fail_charging_tou.index = total_v2g_cap_tou.index
# total_v2g_failt_graph(total_v2g_fail_trip_tou)
#
# total_v2g_failc_graph(total_v2g_fail_charging_tou)
#
# total_v2g_cap_graph_base(total_v2g_cap)
# total_v2g_cap_graph(total_v2g_cap_tou, total_v2g_cap_tou1)
# total_v2g_cap_graph1(total_v2g_cap_tou1, total_v2g_cap_tou1)
# total_v2g_cap_graph1(total_v2g_cap_tou2, total_v2g_cap_tou1)