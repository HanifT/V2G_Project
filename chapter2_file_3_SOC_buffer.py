import pandas as pd
from parking import (charging_dataframe, box_plot_with_stats_for_three, v2g_tou_cap, v2g_tou_cap_20, v2g_tou_cap_30, v2g_tou_cap_40, v2g_tou_cap_50, storage_cap_tou, total_storage_tou, v2g_participate, v2g_fail)
import warnings
warnings.filterwarnings("ignore")
##################################################################################################################
##################################################################################################################
# Section 1
# Load the CSV file without parsing the datetime columns
# V2G_cap_charging_rate = pd.read_csv("V2G_cap_charging_rate.csv")
##################################################################################################################
##################################################################################################################
# TOU V2G Down to Zero
##################################################################################################################
##################################################################################################################
v2g_tou = V2G_cap_charging_rate.copy()
v2g_tou = v2g_tou_cap(v2g_tou)

V2G_hourly_12_tou, V2G_hourly_6_tou, V2G_hourly_19_tou, V2G_hourly_12_tou_sum, V2G_hourly_6_tou_sum, V2G_hourly_19_tou_sum = storage_cap_tou(v2g_tou)

v2g_tou = v2g_participate(v2g_tou)

v2g_tou.to_csv("v2g_tou.csv")

v2g_participate_dataframe = v2g_tou[v2g_tou["V2G_participate"] == True]

df_summary_tou = total_storage_tou(V2G_hourly_6_tou_sum*1.3, V2G_hourly_12_tou_sum*1.3, V2G_hourly_19_tou_sum)

df_box1_tou = [V2G_hourly_12_tou_sum, V2G_hourly_6_tou_sum, V2G_hourly_19_tou_sum]
labels1_tou = ["12 kW", "6.6 kW", "19 kW"]

box_plot_with_stats_for_three(df_box1_tou, labels1_tou, 0, 2000)
data_failure = v2g_fail(v2g_participate_dataframe)

##################################################################################################################
##################################################################################################################
# TOU V2G Down to 20
##################################################################################################################
##################################################################################################################
v2g_tou_20 = V2G_cap_charging_rate.copy()
v2g_tou_20 = v2g_tou_cap_20(v2g_tou_20)


V2G_hourly_12_tou_20, V2G_hourly_6_tou_20, V2G_hourly_19_tou_20, V2G_hourly_12_tou_20_sum, V2G_hourly_6_tou_20_sum, V2G_hourly_19_tou_20_sum = storage_cap_tou(v2g_tou_20)

v2g_tou_20 = v2g_participate(v2g_tou_20)

v2g_participate_dataframe_20 = v2g_tou_20[v2g_tou_20["V2G_participate"] == True]

df_summary_tou_20 = total_storage_tou(V2G_hourly_6_tou_20_sum, V2G_hourly_12_tou_20_sum, V2G_hourly_19_tou_20_sum)

df_box1_tou_20 = [V2G_hourly_6_tou_20_sum, V2G_hourly_12_tou_20_sum, V2G_hourly_19_tou_20_sum]
labels1_tou_20 = ["6.6 kW", "12 kW", "19 kW"]


box_plot_with_stats_for_three(df_box1_tou_20, labels1_tou_20)
data_failure_20 = v2g_fail(v2g_participate_dataframe_20)


##################################################################################################################
##################################################################################################################
# TOU V2G Down to 30
##################################################################################################################
##################################################################################################################
v2g_tou_30 = V2G_cap_charging_rate.copy()
v2g_tou_30 = v2g_tou_cap_30(v2g_tou_30)

V2G_hourly_12_tou_30, V2G_hourly_6_tou_30, V2G_hourly_19_tou_30, V2G_hourly_12_tou_30_sum, V2G_hourly_6_tou_30_sum, V2G_hourly_19_tou_30_sum = storage_cap_tou(v2g_tou_30)

v2g_tou_30 = v2g_participate(v2g_tou_30)

v2g_participate_dataframe_30 = v2g_tou_30[v2g_tou_30["V2G_participate"] == True]

df_summary_tou_30 = total_storage_tou(V2G_hourly_6_tou_30_sum, V2G_hourly_12_tou_30_sum, V2G_hourly_19_tou_30_sum)

df_box1_tou_30 = [V2G_hourly_6_tou_30_sum, V2G_hourly_12_tou_30_sum, V2G_hourly_19_tou_30_sum]
labels1_tou_30 = ["6.6 kW", "12 kW", "19 kW"]


box_plot_with_stats_for_three(df_box1_tou_30, labels1_tou_30)

data_failure_30 = v2g_fail(v2g_participate_dataframe_30)

##################################################################################################################
##################################################################################################################
# TOU V2G Down to 40
##################################################################################################################
##################################################################################################################
v2g_tou_40 = V2G_cap_charging_rate.copy()
v2g_tou_40 = v2g_tou_cap_40(v2g_tou_40)

V2G_hourly_12_tou_40, V2G_hourly_6_tou_40, V2G_hourly_19_tou_40, V2G_hourly_12_tou_40_sum, V2G_hourly_6_tou_40_sum, V2G_hourly_19_tou_40_sum = storage_cap_tou(v2g_tou_40)

v2g_tou_40 = v2g_participate(v2g_tou_40)

v2g_participate_dataframe_40 = v2g_tou_40[v2g_tou_40["V2G_participate"] == True]

df_summary_tou_40 = total_storage_tou(V2G_hourly_6_tou_40_sum, V2G_hourly_12_tou_40_sum, V2G_hourly_19_tou_40_sum)

df_box1_tou_40 = [V2G_hourly_6_tou_40_sum, V2G_hourly_12_tou_40_sum, V2G_hourly_19_tou_40_sum]
labels1_tou_40 = ["6.6 kW", "12 kW", "19 kW"]


box_plot_with_stats_for_three(df_box1_tou_40, labels1_tou_40)

data_failure_40 = v2g_fail(v2g_participate_dataframe_40)

##################################################################################################################
##################################################################################################################
# TOU V2G Down to 50
##################################################################################################################
##################################################################################################################
v2g_tou_50 = V2G_cap_charging_rate.copy()
v2g_tou_50 = v2g_tou_cap_50(v2g_tou_50)

V2G_hourly_12_tou_50, V2G_hourly_6_tou_50, V2G_hourly_19_tou_50, V2G_hourly_12_tou_50_sum, V2G_hourly_6_tou_50_sum, V2G_hourly_19_tou_50_sum = storage_cap_tou(v2g_tou_50)

v2g_tou_50 = v2g_participate(v2g_tou_50)

v2g_participate_dataframe_50 = v2g_tou_50[v2g_tou_50["V2G_participate"] == True]

df_summary_tou_50 = total_storage_tou(V2G_hourly_6_tou_50_sum, V2G_hourly_12_tou_50_sum, V2G_hourly_19_tou_50_sum)

df_box1_tou_50 = [V2G_hourly_6_tou_50_sum, V2G_hourly_12_tou_50_sum, V2G_hourly_19_tou_50_sum]
labels1_tou_50 = ["6.6 kW", "12 kW", "19 kW"]


box_plot_with_stats_for_three(df_box1_tou_50, labels1_tou_50)

data_failure_50 = v2g_fail(v2g_participate_dataframe_50)
