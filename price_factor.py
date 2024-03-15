import pandas as pd
import matplotlib.pyplot as plt

# %%


class RTPricer:
    def __init__(self, df1, df2, price_tou_low, price_tou_high, range_start, range_end, label):
        self.df1 = df1
        self.df2 = df2
        self.price_tou_low = price_tou_low
        self.price_tou_high = price_tou_high
        self.range_start = range_start
        self.range_end = range_end
        self.label = label

    def calculate_rt_price(self):
        # Calculate average load
        self.df1["average_load"] = self.df1["MW"] / len(self.df2["NODE_ID"].unique())

        # Merge demand data with price data
        self.df2 = pd.merge(self.df2, self.df1[["INTERVALSTARTTIME_GMT", "average_load"]], on="INTERVALSTARTTIME_GMT", how="left")

        # Calculate revenue
        self.df2["revenue"] = self.df2["average_load"] * self.df2["MW"]
        total_rt = self.df2["revenue"].sum()

        # Calculate total TOU revenue
        pge_price_tou = {key: self.price_tou_low if (key in range(self.range_start)) or (key in range(self.range_end, 24)) else self.price_tou_high for key in range(24)}
        pge_load = self.df1.groupby("OPR_HR")["MW"].sum().to_dict()
        total_tou = sum(pge_price_tou[key] * pge_load[key] for key in pge_price_tou if key in pge_load)

        # Adjust factor
        adj_factor = total_tou / total_rt

        # Calculate real-time price
        self.df2["rt_price"] = self.df2["MW"] * adj_factor

        return self.df2, adj_factor

    def plot_histogram(self):
        plt.hist(self.df2["rt_price"]/1000, bins=20, color='skyblue', edgecolor='black')

        # Add labels and title with the provided label
        plt.xlabel('Real-time Price ($/kWh)')
        plt.ylabel('Frequency')
        plt.title(f'Histogram of Real-time Price for {self.label}')  # Include the provided label in the title

        # Show plot
        plt.show()

# %%


rt_pricer = RTPricer(combined_demand_PGE, combined_price_PGE, 405, 460, 16, 21, "PGE")
combined_price_PGE_new, adj_factor_PGE = rt_pricer.calculate_rt_price()
rt_pricer.plot_histogram()

rt_pricer = RTPricer(combined_demand_SCE, combined_price_SCE, 275, 475, 16, 21, "SCE")
combined_price_SCE_new, adj_factor_SCE = rt_pricer.calculate_rt_price()
rt_pricer.plot_histogram()

rt_pricer = RTPricer(combined_demand_SDGE, combined_price_SDGE, 312, 417, 17, 21, "PGE")
combined_price_SDGE_new, adj_factor_SDGE = rt_pricer.calculate_rt_price()
rt_pricer.plot_histogram()


