import pandas as pd
from parking import (clean_data, adjust_battery_size, charging_dataframe, charging_c_k, extra_extra_kwh, extra_extra_cycle, total_v2g_cap_graph, total_v2g_failt_graph, total_v2g_failc_graph, extra_extra_kwh_parking)
import warnings
warnings.filterwarnings("ignore")
##################################################################################################################
##################################################################################################################
# Section 1
# # Reading the raw data and clean it
final_dataframes = clean_data()
final_dataframes["bat_cap"] = final_dataframes.apply(adjust_battery_size, axis=1)
final_dataframes = final_dataframes.drop(columns=['Model'])
# # # Saving cleaned data as a csv file
final_dataframes.to_csv("data.csv")
##################################################################################################################
##################################################################################################################

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