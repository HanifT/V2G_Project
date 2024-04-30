import pandas as pd
import json
from pyomo.environ import *
from chapter2_file_7_realtime import real_time_data
import os
from pyomo.environ import ConcreteModel, Set, Param, Var, Objective, Constraint, SolverFactory, Reals
# %% reading json file
GHG_data = pd.read_csv("CISO.csv")
GHG_dict = dict(enumerate(GHG_data.iloc[:, 0]))

# vehicle_list = ["P_1087", "P_1091", "P_1092", "P_1093", "P_1094", "P_1098", "P_1100",  'P_1109', 'P_1111', "P_1112", "P_1123", "P_1125", "P_1125a", "P_1127",
#                 'P_1131', 'P_1132', 'P_1135', 'P_1137', 'P_1140', 'P_1141', 'P_1143', 'P_1144', 'P_1217', 'P_1253', 'P_1257', 'P_1260', 'P_1271', 'P_1272', 'P_1279',
#                 'P_1280', 'P_1281', 'P_1285', 'P_1288', 'P_1294', 'P_1295', 'P_1296', 'P_1304', 'P_1307', 'P_1375'] # Tesla that works


vehicle_list = ['P_1352', 'P_1353', 'P_1357', 'P_1367', 'P_1368', 'P_1370', 'P_1371', "P_1376", 'P_1384', 'P_1388', 'P_1393', "P_1409", "P_1414", 'P_1419', 'P_1421', 'P_1422',
                'P_1423', 'P_1424', 'P_1427', 'P_1429', 'P_1435']  # Bolt works

vehicle_list = ['P_1381', "P_1403", "P_1412"]  # Bolt doesn't work
vehicle_list = ["P_1122", "P_1267"] # tesla that doesn't work



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
    "None": 0
}
################################################################################################################
################################################################################################################
# Decision variables

m.X_CHR = Var(m.V, m.T, domain=Reals, bounds=(-6.6, 150))
################################################################################################################
################################################################################################################
# Dependent variable

m.SOC = Var(m.V, m.T, bounds=(0, 100))
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
        # soc_change = (((m.fac_chr[v, t] * m.X_CHR[v, t] * m.eff_chr) / m.bat_cap) * 100 - ((1 - m.fac_chr[v, t]) * m.trv_dist[v, t] / (m.bat_cap * m.eff_dri)) * 100)

        # Update the state of charge using the calculated soc_change
        return m.SOC[v, t] == m.SOC[v, t - 1] + soc_change  # Use the numerical index


m.SOC_Balance = Constraint(m.V, m.T, rule=soc_balance_rule)
#


# Constraint: Minimum charging rate
def x_chr_min_rule(m, v, t):
    return m.X_CHR[v, t] >= -m.MAX[v, t] * m.fac_chr[v, t]


m.X_CHR_Min = Constraint(m.V, m.T, rule=x_chr_min_rule)


# Parameters using merged_dict
def max_parameter_init(m, v, t):
    charge_level = m.lev_chr[v, t]
    max_charge_rate = CHARGE_LEVEL_MAX_POWER.get(charge_level, 0)  # Get the maximum charge rate based on charge level
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

################################################################################################################
################################################################################################################
# Objective function (minimize total electricity cost)
m.Objective = Objective(expr=sum((m.CRTP[t] * m.X_CHR[v, t])/1000 for v in m.V for t in m.T) + sum(((m.GHG[t] * m.X_CHR[v, t])/20) for v in m.V for t in m.T), sense=minimize)  #Done
# m.Objective = Objective(expr=sum((m.CRTP[t] * m.X_CHR[v, t])/1000 for v in m.V for t in m.T), sense=minimize)  #Done

################################################################################################################
################################################################################################################

# %%

# Open the LP file for writing
m.write("my_model.lp", io_options={"symbolic_solver_labels": True})


# Solve the model
# SolverFactory('GLPK').solve(m)
solver = SolverFactory('glpk', executable='/opt/homebrew/bin/glpsol')
solution = solver.solve(m, tee=True)

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
            charge_level = m.lev_chr[v, t]
            # Write the output to the file
            file.write(f"Vehicle {v} in Time period {t}: Charge = {charge_value:.2f} kWh, SOC = {soc_value:.2f}%, charge_type = {charge_level}\n")

# Print a message indicating that the data has been written to the file
print("Optimal charging schedule written to charging_schedule.txt")



