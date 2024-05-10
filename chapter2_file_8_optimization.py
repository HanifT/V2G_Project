import pandas as pd
import json
from pyomo.environ import *
from chapter2_file_7_realtime import real_time_data
import os
import numpy as np
from pyomo.environ import ConcreteModel, Set, Param, Var, Objective, Constraint, SolverFactory, Reals
from chapter2_file_9_lineariziation import BD_deg
from pyomo.core.kernel.piecewise_library.transforms import piecewise
import logging
from pyomo.environ import value
from math import isclose
logging.getLogger('pyomo.core').setLevel(logging.ERROR)
# %% reading json file
GHG_data = pd.read_csv("CISO.csv")
GHG_dict = dict(enumerate(GHG_data.iloc[:, 0]))

# vehicle_list = ["P_1087", "P_1091", "P_1092", "P_1093", "P_1094", "P_1098", "P_1100",  'P_1109', 'P_1111', "P_1112", "P_1123", "P_1125", "P_1125a", "P_1127",
#                 'P_1131', 'P_1132', 'P_1135', 'P_1137', 'P_1140', 'P_1141', 'P_1143', 'P_1144', 'P_1217', 'P_1253', 'P_1257', 'P_1260', 'P_1271', 'P_1272', 'P_1279',
#                 'P_1280', 'P_1281', 'P_1285', 'P_1288', 'P_1294', 'P_1295', 'P_1296', 'P_1304', 'P_1307', 'P_1375'] # Tesla that works


# vehicle_list = ['P_1352', 'P_1353', 'P_1357', 'P_1367', 'P_1368', 'P_1370', 'P_1371', "P_1376", 'P_1384', 'P_1388', 'P_1393', "P_1409", "P_1414", 'P_1419', 'P_1421', 'P_1422',
#                 'P_1423', 'P_1424', 'P_1427', 'P_1429', 'P_1435']  # Bolt works

# vehicle_list = ['P_1381', "P_1403", "P_1412"]  # Bolt doesn't work
# vehicle_list = ["P_1122", "P_1267"] # tesla that doesn't work


vehicle_list = ["P_1087", "P_1091", "P_1092", "P_1093", "P_1094", "P_1098", "P_1100", 'P_1109', 'P_1111', "P_1112", "P_1123", "P_1125", "P_1125a", "P_1127",
                'P_1131', 'P_1132', 'P_1135', 'P_1137', 'P_1140', 'P_1141', 'P_1143', 'P_1144', 'P_1217', 'P_1253', 'P_1257', 'P_1260', 'P_1271', 'P_1272', 'P_1279',
                'P_1280', 'P_1281', 'P_1285', 'P_1288', 'P_1294', 'P_1295', 'P_1296', 'P_1304', 'P_1307', 'P_1375', 'P_1352', 'P_1353', 'P_1357', 'P_1367', 'P_1368', 'P_1370', 'P_1371', "P_1376", 'P_1384', 'P_1388', 'P_1393', "P_1409", "P_1414", 'P_1419', 'P_1421', 'P_1422',
                'P_1423', 'P_1424', 'P_1427', 'P_1429', 'P_1435']

vehicle_list = ["P_1087"]

real_time_data(vehicle_list)

with open("charging_dict.json", "r") as json_file:
    charging_dict = json.load(json_file)

with open("trip_dict.json", "r") as json_file:
    trip_dict = json.load(json_file)

with open("merged_dict.json", "r") as json_file:
    merged_dict = json.load(json_file)

with open("combined_price_PGE_average.json", "r") as json_file:
    combined_price_PGE_average = json.load(json_file)

with open("combined_price_SCE_average.json", "r") as json_file:
    combined_price_SCE_average = json.load(json_file)

with open("combined_price_SDGE_new_average.json", "r") as json_file:
    combined_price_SDGE_new_average = json.load(json_file)

# Load the JSON file containing the points
points, points1, points2 = BD_deg()
points2['slope'] = (points2['BD'].diff()) / (points2['DOD'].diff())

# %%

# Convert keys to integers
combined_price_PGE_average_numerical = {int(key): value for key, value in combined_price_PGE_average.items()}
merged_dict = {outer_key: {int(inner_key): inner_value for inner_key, inner_value in outer_value.items()} for outer_key, outer_value in merged_dict.items()}
max_index = max(max(map(int, inner_dict.keys())) for inner_dict in merged_dict.values())
# %%
# Create a Pyomo model
m = ConcreteModel()
################################################################################################################
################################################################################################################
# Define sets
# Create a new set without the zeros, sorted
m.T = Set(initialize=sorted(key for value in merged_dict.values() for key in value.keys()))  # Set of time periods (excluding zeros), sorted
m.V = Set(initialize=merged_dict.keys())  # Set of vehicles
m.K = Set(initialize=points2.index)  # Set of vehicles
################################################################################################################
################################################################################################################
# Parameters

# Parameters using merged_dict
m.CRTP = Param(m.T, initialize=[combined_price_PGE_average_numerical[t] for t in list(m.T)[:len(combined_price_PGE_average_numerical)]])
m.GHG = Param(m.T, initialize=[GHG_dict[t] for t in list(m.T)[:len(GHG_dict)]])
m.trv_dist = Param(m.V, m.T, initialize=lambda m, v, t: merged_dict.get(v, {}).get(t, {}).get("distance", 0))
m.SOC_REQ = Param(m.V, m.T, initialize=lambda m, v, t: merged_dict.get(v, {}).get(t, {}).get("soc_need", 0))
m.fac_chr = Param(m.V, m.T, initialize=lambda m, v, t: merged_dict.get(v, {}).get(t, {}).get("charging_indicator", 0))
m.lev_chr = Param(m.V, m.T, initialize=lambda m, v, t: merged_dict.get(v, {}).get(t, {}).get("charge_type", "None"))
m.bat_cap = Param(m.V, m.T, initialize=lambda m, v, t: merged_dict.get(v, {}).get(t, {}).get("bat_cap", 100))
# m.batt_deg = Param(points1.index, initialize={index: (float(row['DOD']), float(row['BD'])) for index, row in points1.iterrows()})
m.batt_deg = Param(points2.index, initialize={index: (float(row['DOD']), float(row['BD'])) for index, row in points2.iterrows()})

# Iterate over indices and print values

# Other parameters (unchanged)
m.MAX = Param(m.V, m.T, initialize=6.6)
m.C_THRESHOLD = Param(initialize=20)
m.eff_chr = Param(initialize=0.9)
m.eff_dri = Param(initialize=3)
# m.bat_cap = Param(initialize=100)

# Charge Level Mapping
CHARGE_LEVEL_MAX_POWER = {
    "LEVEL_1": 6.6,
    "LEVEL_2": 6.6,
    "DC_FAST": 150,
    "DC_FAST_REDUCED": 150,  # New entry for reduced speed
    "None": 0
}
################################################################################################################
################################################################################################################
# Decision variables

m.X_CHR = Var(m.V, m.T, domain=Reals)
################################################################################################################
################################################################################################################
# Dependent variable

m.SOC = Var(m.V, m.T, bounds=(0, 100))
m.DOD = Var(m.V, m.T, domain=NonNegativeReals)
m.rho = Var(m.V, m.T, bounds=(0, 0.00079))
m.batt_deg = Var(m.V, m.T)
m.dc_fast_speed_reduced = Var(m.V, m.T, domain=Binary)

################################################################################################################
################################################################################################################
# Constraint: Balance battery state of charge


def soc_balance_rule(m, v, t):
    if t == 0:
        # Set initial state of charge to 100% at t=0
        return m.SOC[v, t] == 100
    else:
        # Calculate change in state of charge
        soc_change = (((m.fac_chr[v, t] * m.X_CHR[v, t] * m.eff_chr) / m.bat_cap[v, t]) * 100 - ((1 - m.fac_chr[v, t]) * m.trv_dist[v, t] / (m.bat_cap[v, t] * m.eff_dri)) * 100)

        # Update the state of charge using the calculated soc_change
        return m.SOC[v, t] == m.SOC[v, t - 1] + soc_change  # Use the numerical index


m.SOC_Balance = Constraint(m.V, m.T, rule=soc_balance_rule)


# Constraint: Minimum charging rate
def x_chr_min_rule(m, v, t):
    return m.X_CHR[v, t] >= -m.MAX[v, t] * m.fac_chr[v, t]


m.X_CHR_Min = Constraint(m.V, m.T, rule=x_chr_min_rule)


# Parameters using merged_dict
def max_parameter_init(m, v, t):
    charge_level = m.lev_chr[v, t]
    # max_charge_rate = CHARGE_LEVEL_MAX_POWER.get(charge_level, 0)  # Get the maximum charge rate based on charge level
    # return m.X_CHR[v, t] <= max_charge_rate * m.fac_chr[v, t]
    if charge_level == "DC_FAST":
        # max_charge_rate = CHARGE_LEVEL_MAX_POWER.get("DC_FAST")
        max_charge_rate = CHARGE_LEVEL_MAX_POWER.get("DC_FAST") * (1 - m.dc_fast_speed_reduced[v, t]) + CHARGE_LEVEL_MAX_POWER.get("DC_FAST_REDUCED") * m.dc_fast_speed_reduced[v, t]
    else:
        max_charge_rate = CHARGE_LEVEL_MAX_POWER.get(charge_level, 0)
    return m.X_CHR[v, t] <= max_charge_rate * m.fac_chr[v, t]


m.X_CHR_Max = Constraint(m.V, m.T, rule=max_parameter_init)


def soc_min_departure_rule(m, v, t):
    if t == 0 or m.fac_chr[v, t - 1] == 1 and m.fac_chr[v, t] == 0:
        # Beginning of a charging session
        return m.SOC[v, t] >= m.SOC_REQ[v, t]
    else:
        return Constraint.Skip


m.SOC_Min_Departure = Constraint(m.V, m.T, rule=soc_min_departure_rule)


#
# Constraint: Minimum SOC buffer during charging/discharging
def soc_buffer_rule(m, v, t):
    if m.fac_chr[v, t] == 1:
        return m.SOC[v, t] >= m.C_THRESHOLD
    else:
        return Constraint.Skip  # No constraint enforced when fac_chr is 0


m.SOC_Buffer = Constraint(m.V, m.T, rule=soc_buffer_rule)


# Constraint: Reduce charging speed after SOC reaches 80% during DC fast charging

def soc_threshold_for_reduced_speed(m, v, t):
    if m.SOC[v, t] >= 80:
        return m.dc_fast_speed_reduced[v, t] == 1  # Indicator constraint
    return Constraint.Skip


################################################################################################################
################################################################################################################

# Battery Degradation

def depth_of_discharge(m, v, t):
    if m.fac_chr[v, t] == 1:
        return m.DOD[v, t] == 100 - m.SOC[v, t]
    else:
        return Constraint.Skip

m.depth_decharge_constrain = Constraint(m.V, m.T, rule=depth_of_discharge)


# Define binary variables to represent each segment of the piecewise function
m.batt_deg = Var(m.V, m.T)
m.dod_initial = Param(m.K, initialize=points2["DOD"].to_dict())
m.dod_final = Param(m.K, initialize=points2["BD"].to_dict())
m.slope = Param(m.K, initialize=points2["slope"].to_dict())
m.delta = Param( m.K, initialize=100/5)
m.dod_var1 = Var(m.V, m.T, domain=NonNegativeReals, bounds=(0, 19.99))
m.dod_var2 = Var(m.V, m.T, domain=NonNegativeReals)
m.segment = Var(m.V, m.T, m.K, domain=Binary)
m.abs1 = Var(m.V, m.T, domain=NonNegativeReals)

def batt_degradation0(m, v, t):
    if m.fac_chr[v, t] == 1:
      return sum(m.segment[v, t, k] for k in m.K) == 1
    else:
        return Constraint.Skip  # Skip the constraint if m.fac_chr[v, t] is not equal to 1


m.batt_degradation_constraint0 = Constraint(m.V, m.T, rule=batt_degradation0)


def batt_degradation1(m, v, t):
    if m.fac_chr[v, t] == 1:
        return sum((m.dod_var1[v, t] + m.dod_initial[k] * m.segment[v, t, k]) for k in m.K) == m.DOD[v, t]
    else:
        return Constraint.Skip  # Skip the constraint if m.fac_chr[v, t] is not equal to 1


m.batt_degradation_constraint1 = Constraint(m.V, m.T, rule=batt_degradation1)

a = list(range(0, 4))


def batt_degradation2(m, v, t):
    if m.fac_chr[v, t] == 1:
        return sum(m.slope[k+1]*(m.dod_var1[v, t]) + m.dod_final[k]*m.segment[v, t, k] for k in a) == m.rho[v, t]
    else:
        return Constraint.Skip  # Skip the constraint if m.fac_chr[v, t] is not equal to 1


m.batt_degradation_constraint2 = Constraint(m.V, m.T, rule=batt_degradation2)



def batt_degradation3(m, v, t):
    if m.fac_chr[v, t] == 1:
       return m.batt_deg[v, t] == (m.rho[v, t] - m.rho[v, t-1])
    else:
       return Constraint.Skip  # Skip the constraint if m.fac_chr[v, t] is not equal to 1


m.batt_degradation_constraint3 = Constraint(m.V, m.T, rule=batt_degradation3)


def batt_degradation4(m, v, t):
    if m.fac_chr[v, t] == 1:
       return m.batt_deg[v, t] <= m.abs1[v, t]
    else:
       return Constraint.Skip  # Skip the constraint if m.fac_chr[v, t] is not equal to 1


m.batt_degradation_constraint4 = Constraint(m.V, m.T, rule=batt_degradation4)

def batt_degradation5(m, v, t):
    if m.fac_chr[v, t] == 1:
       return m.batt_deg[v, t] <= -1*m.abs1[v, t]
    else:
       return Constraint.Skip  # Skip the constraint if m.fac_chr[v, t] is not equal to 1


m.batt_degradation_constraint5 = Constraint(m.V, m.T, rule=batt_degradation5)

################################################################################################################
################################################################################################################
# Objective function (minimize total electricity cost)
m.Objective = Objective(expr=sum((m.CRTP[t] * m.X_CHR[v, t]) / 1000 for v in m.V for t in m.T) +
                             sum(((m.GHG[t] * m.X_CHR[v, t]) / 20) for v in m.V for t in m.T) +
                             sum(((m.abs1[v, t]) * 10000 * m.fac_chr[v, t]) for v in m.V for t in m.T)
                        , sense=minimize)
################################################################################################################
################################################################################################################

# %%

# Open the LP file for writing
m.write("my_model.lp", io_options={"symbolic_solver_labels": True})

# Solve the model
# SolverFactory('GLPK').solve(m)
# solver = SolverFactory('glpk', executable='/opt/homebrew/bin/glpsol')
# solution = solver.solve(m, tee=True)

# Using CBC (ensure you've installed CBC)
solver = SolverFactory('cbc', executable='/opt/homebrew/bin/cbc')

# Optional: Set the number of threads (`threads` option)
solver.options['threads'] = 10  # Adjust based on the number of cores you want to use

solution = solver.solve(m, tee=True)
################################################################################################################
################################################################################################################
# %%
# Print results
# Open a text file for writing
with open('charging_schedule.txt', 'w') as file:
    file.write("Optimal charging schedule:\n")

    # Loop through vehicles and time periods
    for v in m.V:
        for t in m.T:
            charge_value = m.X_CHR[v, t].value if m.X_CHR[v, t].value is not None else 0.0
            soc_value = m.SOC[v, t].value if m.SOC[v, t].value is not None else 0.0
            DOD_value = m.DOD[v, t].value if m.DOD[v, t].value is not None else 0.0
            DOD_var = m.dod_var1[v, t].value if m.dod_var1[v, t].value is not None else 0.0
            rho_value = m.rho[v, t].value if m.rho[v, t].value is not None else 0.0
            deg_value = m.batt_deg[v, t].value if m.batt_deg[v, t].value is not None else 0.0
            charge_level = m.lev_chr[v, t]
            # Write the output to the file
            file.write(f"Vehicle {v} in Time period {t}: Charge = {charge_value:.2f} kWh, SOC = {soc_value:.2f}%,"
                       f" DOD = {DOD_value:.2f}%,"
                       f" DOD = {DOD_var:.2f}%,"
                       f" rho = {rho_value:.9f},"
                       f" deg = {deg_value:.9f},"
                       f" charge_type = {charge_level}\n")

# Print a message indicating that the data has been written to the file
print("Optimal charging schedule written to charging_schedule.txt")

################################################################################################################
################################################################################################################
# %%
# Construct a data structure for your results
# Extract results
results = []
for v in m.V:
    vehicle_results = {'Vehicle': v}
    for t in m.T:
        x_chr_value = m.X_CHR[v, t].value if m.X_CHR[v, t].value is not None else 0
        x_dod_value = m.DOD[v, t].value if m.DOD[v, t].value is not None else 0
        soc_value = m.SOC[v, t].value if m.SOC[v, t].value is not None else 0
        batt_deg_value = m.batt_deg[v, t].value * m.fac_chr[v, t] if m.batt_deg[v, t].value is not None else 0
        fac_chr_value = m.fac_chr[v, t]

        electricity_cost = (m.CRTP[t] * x_chr_value) / 1000 if m.CRTP[t] is not None else 0
        degradation_cost = (batt_deg_value * 10000)
        ghg_emissions = (m.GHG[t] * x_chr_value) / 20 if m.GHG[t] is not None else 0

        vehicle_results[t] = {
            'X_CHR': x_chr_value,
            'DOD': x_dod_value,
            'SOC': soc_value,
            'Electricity_Cost': electricity_cost,
            'Degradation_Cost': degradation_cost,
            'GHG_Emissions': ghg_emissions
        }
    results.append(vehicle_results)

# Create DataFrame
df = pd.DataFrame(results)

# Reorganize DataFrame
df = df.set_index('Vehicle').stack().apply(pd.Series).reset_index()
df = df.round(10)
df["Degradation_Cost"].sum()
df.columns = ['Vehicle', 'Hour', 'X_CHR', 'DOD', 'SOC', 'Electricity_Cost', 'Degradation_Cost', 'GHG_Emissions']
df.to_csv("test.csv")

# Pivot DataFrame
df_pivot = df.pivot(index='Vehicle', columns='Hour', values=['Electricity_Cost', 'GHG_Emissions'])

# Save to CSV
df_pivot.to_csv('ev_optimization_results.csv')

# Split df_pivot into separate DataFrames for electricity cost and GHG emissions
df_electricity_cost = df_pivot['Electricity_Cost']
df_ghg_emissions = df_pivot['GHG_Emissions']

df_electricity_cost.to_csv('ev_optimization_results1.csv')
df_ghg_emissions.to_csv('ev_optimization_results2.csv')

################################################################################################################
################################################################################################################