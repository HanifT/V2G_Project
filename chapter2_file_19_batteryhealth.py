# %%
import pandas as pd
from parking import process_and_plot_utility_data, plot_utility_panels
import warnings
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
# %%

all_hourly_charging_data_grouped_pge, summary_pge, summary_actual_pge, summary_potential_pge = (
    process_and_plot_utility_data("PGE", "/Users/haniftayarani/V2G_Project/Hourly_data", 5500, 7000))

all_hourly_charging_data_grouped_sce, summary_pge_sce, summary_actual_sce, summary_potential_sce = (
    process_and_plot_utility_data("SCE", "/Users/haniftayarani/V2G_Project/Hourly_data", 5500, 7000))

all_hourly_charging_data_grouped_sdge, summary_pge_sdge, summary_actual_sdge, summary_potential_sdge = (
    process_and_plot_utility_data("SDGE", "/Users/haniftayarani/V2G_Project/Hourly_data",5500, 7000))

all_hourly_charging_data_grouped_smud, summary_pge_smud, summary_actual_smud, summary_potential_smud= (
    process_and_plot_utility_data("SMUD", "/Users/haniftayarani/V2G_Project/Hourly_data",1500, 1500))
# Final
# %%

# Prepare the data for each utility
utility_data = [
    (summary_actual_pge, summary_potential_pge),
    (summary_actual_sce, summary_potential_sce),
    (summary_actual_sdge, summary_potential_sdge),
    (summary_actual_smud, summary_potential_smud),
]

utility_names = ["PGE", "SCE", "SDGE", "SMUD"]
ylim_pairs = [(0, 6000), (0, 7000), (0, 7000), (0, 2000)]

# Create the 2x4 panel plot
plot_utility_panels(utility_data, utility_names, ylim_pairs, title_size=18, axis_text_size=16, figsize=(12, 16))



