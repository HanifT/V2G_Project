import pandas as pd
import json
from pyomo.environ import *
# %% reading json file
with open("charging_dict.json", "r") as json_file:
    charging_dict = json.load(json_file)

with open("combined_price_PGE_average.json", "r") as json_file:
    combined_price_PGE_average = json.load(json_file)

with open("combined_price_SCE_average.json", "r") as json_file:
    combined_price_SCE_average = json.load(json_file)

with open("combined_price_SDGE_new_average.json", "r") as json_file:
    combined_price_SDGE_new_average = json.load(json_file)

# %%


class V2GModel:
    def __init__(self, price_data_input, ch_factor_input, bat_cap_input, efficiency_input, charging_rate_input, soc_req_dict_input, soc_buffer_threshold_input, initial_soc_dict_input, end_time_dict_input):
        self.model = ConcreteModel()
        self.price_data = price_data_input
        self.ch_factor = ch_factor_input
        self.bat_cap = bat_cap_input
        self.efficiency = efficiency_input
        self.min_charging_rate = charging_rate_input * (-1)
        self.max_charging_rate = charging_rate_input
        self.soc_req_dict = soc_req_dict_input
        self.soc_buffer_threshold = soc_buffer_threshold_input
        self.initial_soc_dict = initial_soc_dict_input
        self.end_time_dict = end_time_dict_input

    def build_model(self):
        # Determine the maximum time period among all vehicles
        max_time_period = max(max(self.end_time_dict.values(), default=[]), default=0)
        # Set the range of time periods dynamically
        self.model.T = RangeSet(0, max_time_period)
        self.model.c = Param(self.model.T, initialize=self.price_data)  # Electricity prices
        self.model.ch_factor = Param(self.model.T, initialize=self.ch_factor)  # Charging factor
        self.model.bat_cap = Param(self.model.T, initialize=self.bat_cap)  # Battery capacity
        self.model.eff = Param(self.model.T, initialize=self.efficiency)  # Efficiency
        self.model.x_min_chr = Param(self.model.T, initialize=self.min_charging_rate)  # Minimum charging rate
        self.model.x_max_chr = Param(self.model.T, initialize=self.max_charging_rate)  # Maximum charging rate
        self.model.c_soc_req = Param(self.model.T, initialize=self.soc_req_dict)  # Minimum SOC required at departure time
        self.model.c_threshold = Param(initialize=self.soc_buffer_threshold)  # SOC buffer threshold
        self.model.c_initial_soc = Param(self.model.T, initialize=self.initial_soc_dict)  # Initial SOC for each charging session

        # Decision variables
        self.model.x_chr = Var(self.model.T, bounds=(self.min_charging_rate, self.max_charging_rate))  # Charging/discharging energy
        self.model.x_soc = Var(self.model.T, bounds=(0, 100))  # SOC of the BEV

        # Objective function
        self.model.cost = Objective(expr=sum(self.model.c[t] * self.model.x_chr[t] for t in self.model.T), sense=minimize)

        # Constraints
        self.model.energy_balance_constraint = Constraint(self.model.T, rule=self.energy_balance_rule)
        self.model.soc_min_at_departure_constraint = Constraint(self.model.T, rule=self.soc_min_at_departure_rule)
        self.model.soc_buffer_constraint = Constraint(self.model.T, rule=self.soc_buffer_rule)

        print("ch_factor:", self.ch_factor)
        print("bat_cap:", self.bat_cap)
        print("efficiency:", self.efficiency)
        print("min_charging_rate:", self.min_charging_rate)
        print("max_charging_rate:", self.max_charging_rate)
        print("soc_req_dict:", self.soc_req_dict)
        print("soc_buffer_threshold:", self.soc_buffer_threshold)
        print("initial_soc_dict:", self.initial_soc_dict)


    def energy_balance_rule(self, t):
        if t == self.model.T.first():
            return self.model.x_soc[t] == self.model.c_initial_soc[t]  # Initial SOC
        else:
            return self.model.x_soc[t] == self.model.x_soc[t - 1] + (self.model.eff * self.model.ch_factor[t] * self.model.x_chr[t] / self.model.bat_cap[t]) * 100

    def soc_min_at_departure_rule(self, t_dep):
        return self.model.x_soc[t_dep] >= self.model.c_soc_req[t_dep]

    def soc_buffer_rule(self, t):
        return self.model.x_soc[t] >= self.model.c_threshold

    def solve_model(self):
        solver = SolverFactory('cbc')
        results = solver.solve(self.model)
        return results

    def get_solution(self, results):
        if results.solver.termination_condition == TerminationCondition.optimal:
            optimal_cost = self.model.cost()
            optimal_x_chr = {t: self.model.x_chr[t].value for t in self.model.T}
            optimal_x_soc = {t: self.model.x_soc[t].value for t in self.model.T}
            return optimal_cost, optimal_x_chr, optimal_x_soc
        else:
            print("No optimal solution found.")


# %%

# Assuming your dictionary is named vehicle_dict
price_data = combined_price_PGE_average  # Assuming this is already defined
efficiency = {t: 0.9 for t in range(0, 16001)}  # Initialize efficiency for each time period
soc_buffer_threshold = 20
charging_rate = 6.6
# Initialize other parameters...
# Extracting data from the dictionary
soc_req_dict = {}  # Dictionary to store minimum SOC required at departure time
soc_init = {}
end_time_dict = {}  # Dictionary to store end time of charging sessions
ch_factor = {}
bat_cap = {}

# Assuming your dictionary is named vehicle_dict
for vehicle_name, hours_data in charging_dict.items():
    for hour, data in hours_data.items():
        soc_req_dict[hour] = data['soc_end']
        end_time_dict[hour] = data['end_time']
        ch_factor[hour] = data['charging_indicator']
        bat_cap[hour] = data["bat_cap"]
        soc_init[hour] = data["soc_init"]

# Create the V2GModel instance
v2g_model = V2GModel(price_data, ch_factor, bat_cap, efficiency, charging_rate, soc_req_dict, soc_buffer_threshold, soc_init, end_time_dict)




v2g_model.build_model()
results = v2g_model.solve_model()