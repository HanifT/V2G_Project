import pandas as pd
import json
from pyomo.environ import *
from price_factor import get_utility_prices
from chapter2_file_7_realtime import real_time_data
from chapter2_file_17_parking_charging import real_time_data_parking
import os
from pyomo.environ import ConcreteModel, Set, Param, Var, Objective, Constraint, SolverFactory, Reals, NonNegativeReals, Binary
import logging
logging.getLogger('pyomo.core').setLevel(logging.ERROR)
# %% reading json file
GHG_data = pd.read_csv("CISO.csv")
GHG_dict = dict(enumerate(GHG_data.iloc[:, 0]))

# vehicle_list = ["P_1087", "P_1091", "P_1092", "P_1093", "P_1094", "P_1098", "P_1100", 'P_1109', 'P_1111', "P_1112", "P_1123", "P_1125",
#                 "P_1125a", "P_1127", 'P_1131', 'P_1132', 'P_1135', 'P_1137', "P_1141", "P_1143", 'P_1217', 'P_1253', 'P_1257', 'P_1260',
#                 'P_1271', 'P_1272', 'P_1279', 'P_1280', 'P_1281', 'P_1285', 'P_1288', 'P_1294', 'P_1295', 'P_1296', 'P_1304', 'P_1307',
#                 "P_1357", "P_1367", 'P_1375', 'P_1353', 'P_1368', 'P_1371', "P_1376", 'P_1393', "P_1414", 'P_1419', 'P_1421', 'P_1422', 'P_1424', 'P_1427']
#
#
# real_time_data(vehicle_list)
# real_time_data_parking(vehicle_list)


with open("merged_dict.json", "r") as json_file:
    merged_dict = json.load(json_file)

# with open("merged_dict_parking.json", "r") as json_file:
#     merged_dict = json.load(json_file)

# %%
rt_rate_pge, tou_prices_pge, ev_rate_prices_pge, commercial_prices_pge = get_utility_prices("PGE")
rt_rate_sce, tou_prices_sce, ev_rate_prices_sce, commercial_prices_sce = get_utility_prices("SCE")
rt_rate_sdge, tou_prices_sdge, ev_rate_prices_sdge, commercial_prices_sdge = get_utility_prices("SDGE")
rt_rate_smud, tou_prices_smud, ev_rate_prices_smud, commercial_prices_smud = get_utility_prices("SMUD")


merged_dict = {outer_key: {int(inner_key): inner_value for inner_key, inner_value in outer_value.items()} for outer_key, outer_value in merged_dict.items()}
max_index = max(max(map(int, inner_dict.keys())) for inner_dict in merged_dict.values())
# %%


def create_model_and_export_excel(charging_speed, ghg_cost_per_tonne, x_chr_domain, locs, price, commercial_prices, price_name):

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
    filtered_price = {t: price[t] for t in m.T if t in price}
    filtered_commercial_prices = {t: commercial_prices[t] for t in m.T if t in commercial_prices}  # Commercial
    m.CRTP = Param(m.T, initialize=filtered_price, default=420)  # Default to 420 for missing values
    m.CRTP_Commercial = Param(m.T, initialize=filtered_commercial_prices, default=420)  # Default to 420

    # Parameters using merged_dict

    m.GHG = Param(m.T, initialize={t: GHG_dict[t] for t in m.T if t in GHG_dict})
    m.trv_dist = Param(m.V, m.T, initialize=lambda m, v, t: merged_dict.get(v, {}).get(t, {}).get("distance", 0))
    m.soc_cons = Param(m.V, m.T, initialize=lambda m, v, t: float(merged_dict.get(v, {}).get(t, {}).get("soc_diff", 0)))
    m.SOC_REQ = Param(m.V, m.T, initialize=lambda m, v, t: merged_dict.get(v, {}).get(t, {}).get("soc_need", 0))
    m.fac_chr = Param(m.V, m.T, initialize=lambda m, v, t: merged_dict.get(v, {}).get(t, {}).get("charging_indicator", 0))
    m.lev_chr = Param(m.V, m.T, initialize=lambda m, v, t: merged_dict.get(v, {}).get(t, {}).get("charge_type", "None"))
    m.veh_model = Param(m.V, m.T, initialize=lambda m, v, t: merged_dict.get(v, {}).get(t, {}).get("model", "None"))
    m.location = Param(m.V, m.T, initialize=lambda m, v, t: merged_dict.get(v, {}).get(t, {}).get("location", "None"))

    def init_bat_cap(m, v, t):
        return merged_dict.get(v, {}).get(t, {}).get("bat_cap", 80)

    m.bat_cap = Param(m.V, m.T, initialize=init_bat_cap)

    # Other parameters (unchanged)
    m.MAX = Param(m.V, m.T, initialize=charging_speed)
    m.C_THRESHOLD = Param(initialize=15)
    m.eff_chr = Param(initialize=0.95)
    # m.eff_dri = Param(initialize=3.3)
    m.ghg_cost = Param(initialize=ghg_cost_per_tonne)  # $ per gram

    # Define the efficiency parameter based on the vehicle model
    def eff_dri_rule(model, v, t):
        if model.veh_model[v, t] == "Chevy":
            return 3.5
        elif model.veh_model[v, t] == "Tesla":
            return 3
        else:
            return 3  # Default value if the model is neither Chevy nor Tesla

    m.eff_dri = Param(m.V, m.T, initialize=eff_dri_rule)

    # Charge Level Mapping
    CHARGE_LEVEL_MAX_POWER = {
        "LEVEL_1": charging_speed,
        "LEVEL_2": charging_speed,
        "LEVEL_2/1": charging_speed,
        "DC_FAST_Tesla": 150,
        "DC_FAST_Bolt": 50,
        "DC_FAST_REDUCED": 70,  # New entry for reduced speed
        "None": 0
    }

    # Define the effective price parameter based on location
    def effective_price_rule(m,v, t):
        if t not in m.T:
            # print(f"Time index {t} not in m.T. Returning default value 420.")
            return 420
        if m.location[v, t] == "Work":
            return m.CRTP_Commercial[t] if t in m.CRTP_Commercial else 420
        elif m.location[v, t] == "Home":
            return m.CRTP[t] if t in m.CRTP else 420
        else:
            return 420

    m.CRTP_Effective = Param(m.V, m.T, initialize=effective_price_rule, default=420)

    ################################################################################################################
    ################################################################################################################
    # Decision variables
    m.X_CHR = Var(m.V, m.T, domain=x_chr_domain)
    ################################################################################################################
    ################################################################################################################
    # Dependent variable
    m.SOC = Var(m.V, m.T, bounds=(0, 100))
    m.batt_deg = Var(m.V, m.T, domain=NonNegativeReals)
    # m.dc_fast_speed_reduced = Var(m.V, m.T, domain=Binary)
    m.batt_deg_cost = Var(m.V, m.T, domain=NonNegativeReals)
    ################################################################################################################
    ################################################################################################################

    degradation_parameters = {
        # Battery capacity group: (slope, intercept)
        60: (2.15e-02, 0), 65: (2.15e-02, 0), 66: (2.15e-02, 0), 70: (2.15e-02, 0),
        75: (2.15e-02, 0), 80: (2.15e-02, 0), 85: (2.15e-02, 0), 90: (2.15e-02, 0),
        95: (2.15e-02, 0), 100: (2.15e-02, 0)
    }

    # Define functions to get slope and intercept based on battery capacity
    # Initialize d_slope and d_intercept using degradation_parameters and bat_cap

    def init_d_slope(m, v, t):
        cap = m.bat_cap[v, t]
        return degradation_parameters.get(cap, (0, 1))[0]

    def init_d_intercept(m, v, t):
        cap = m.bat_cap[v, t]
        return degradation_parameters.get(cap, (0, 1))[1]

    # Define Pyomo parameters for battery degradation
    m.d_slope = Param(m.V, m.T, initialize=init_d_slope)
    m.d_intercept = Param(m.V, m.T, initialize=init_d_intercept)

    # Define a variable to track cumulative negative charging
    m.cumulative_charging = Var(m.V, m.T, domain=NonNegativeReals)
    m.X_CHR_neg_part = Var(m.V, m.T, domain=NonNegativeReals)  # Variable for negative part of X_CHR
    # Constraint: Balance battery state of charge
    ################################################################################################################
    ################################################################################################################

    def soc_balance_rule(m, v, t):
        if t == 0:
            # Set initial state of charge to 100% at t=0
            return m.SOC[v, t] == 100
        else:
            # Calculate change in state of charge
            # soc_change = (((m.X_CHR[v, t] * m.eff_chr) / m.bat_cap[v, t]) * 100 - (m.trv_dist[v, t] / (m.bat_cap[v, t] * m.eff_dri[v, t])) * 100)
            chr_term = ((m.X_CHR[v, t] * m.eff_chr) / m.bat_cap[v, t]) * 100
            dri_term = m.soc_cons[v, t]
            soc_change = (chr_term - dri_term)
            # Debugging prints

            # Update the state of charge using the calculated soc_change
            return m.SOC[v, t] == m.SOC[v, t - 1] + soc_change  # Use the numerical index

    m.SOC_Balance = Constraint(m.V, m.T, rule=soc_balance_rule)

    # Constraint: Minimum charging rate
    def x_chr_min_rule(m, v, t):
         return m.X_CHR[v, t] >= -m.MAX[v, t]

    m.X_CHR_Min = Constraint(m.V, m.T, rule=x_chr_min_rule)

    # Parameters using merged_dict
    def max_parameter_init(m, v, t):
        charge_level = m.lev_chr[v, t]
        veh_model = m.veh_model[v, t]
        if veh_model == "Tesla":
            if charge_level == "DC_FAST":
                max_charge_rate = CHARGE_LEVEL_MAX_POWER.get("DC_FAST_Tesla")
            else:
                max_charge_rate = CHARGE_LEVEL_MAX_POWER.get(charge_level, 0)
        elif veh_model == "Bolt":
            if charge_level == "DC_FAST":
                max_charge_rate = CHARGE_LEVEL_MAX_POWER.get("DC_FAST_Bolt")
            else:
                max_charge_rate = CHARGE_LEVEL_MAX_POWER.get(charge_level, 0)
        else:
            max_charge_rate = CHARGE_LEVEL_MAX_POWER.get(charge_level, 0)

        return m.X_CHR[v, t] <= max_charge_rate * m.fac_chr[v, t]

    m.X_CHR_Max = Constraint(m.V, m.T, rule=max_parameter_init)

    def soc_min_departure_rule(m, v, t):
        if t == 0 or (m.fac_chr[v, t-1] == 1 and m.fac_chr[v, t] == 0):
            # Beginning of a charging session
            return m.SOC[v, t] >= m.SOC_REQ[v, t] + 15
        else:
            return Constraint.Skip

    m.SOC_Min_Departure = Constraint(m.V, m.T, rule=soc_min_departure_rule)

    # Constraint: Minimum SOC buffer during charging/discharging
    def soc_buffer_rule(m, v, t):
        if m.fac_chr[v, t] == 1:
            return m.SOC[v, t] >= m.C_THRESHOLD
        else:
            return Constraint.Skip  # No constraint enforced when fac_chr is 0

    m.SOC_Buffer = Constraint(m.V, m.T, rule=soc_buffer_rule)

    def x_chr_non_zero_rule(m, v, t):
        if m.fac_chr[v, t] == 0:
            return m.X_CHR[v, t] == 0
        else:
            return Constraint.Skip
    #
    m.X_CHR_Non_Zero = Constraint(m.V, m.T, rule=x_chr_non_zero_rule)

    def x_chr_min_rule(m, v, t):
        if m.location[v, t] in locs:
            return m.X_CHR[v, t] >= -m.MAX[v, t]
        else:
            return m.X_CHR[v, t] >= 0

    m.X_CHR_Min = Constraint(m.V, m.T, rule=x_chr_min_rule)
    ################################################################################################################
    ################################################################################################################

    # Battery Degradation
    # Constraint to capture the negative part of X_CHR
    def x_chr_neg_part_rule(m, v, t):
        if m.fac_chr[v, t] == 1:
            return m.X_CHR_neg_part[v, t] >= m.X_CHR[v, t]
        else:
            return Constraint.Skip

    m.X_CHR_Neg_Part_Rule = Constraint(m.V, m.T, rule=x_chr_neg_part_rule)

    # Cumulative charging constraint
    def cumulative_charging_rule(m, v, t):
        if m.fac_chr[v, t] == 1:
            if t == 0:
                return m.cumulative_charging[v, t] == 0
            else:
                return m.cumulative_charging[v, t] == m.cumulative_charging[v, t - 1] + m.X_CHR_neg_part[v, t] * m.fac_chr[v, t]
        else:
             return Constraint.Skip

    m.CumulativeCharging = Constraint(m.V, m.T, rule=cumulative_charging_rule)

    # Modified degradation rule based on cumulative charging
    def degradation_rule(m, v, t):
        if m.fac_chr[v, t] == 1:
            return m.batt_deg[v, t] == (m.d_slope[v, t] * m.cumulative_charging[v, t])

        elif m.fac_chr[v, t] == 0:
            return m.batt_deg[v, t] == 0
        else:
            return Constraint.Skip

    m.Degradation_Cost = Constraint(m.V, m.T, rule=degradation_rule)

    def degradation_rule1(m, v, t):
        if m.fac_chr[v, t] == 1:
            return m.batt_deg_cost[v, t] == m.batt_deg[v, t] - m.batt_deg[v, t-1]
        elif m.fac_chr[v, t] == 0:
            return m.batt_deg_cost[v, t] == 0
        else:
            return Constraint.Skip

    m.Degradation_Cost1 = Constraint(m.V, m.T, rule=degradation_rule1)

    ################################################################################################################
    ################################################################################################################
    # Objective function (minimize total electricity cost)
    m.Objective = Objective(expr=
                            sum((m.CRTP_Effective[v, t] * m.X_CHR[v, t]) / 1000 for v in m.V for t in m.T) +
                            # sum(((m.GHG[t] * (m.X_CHR[v, t])) / 1000) * m.ghg_cost for v in m.V for t in m.T) +
                            sum(m.batt_deg_cost[v, t] for v in m.V for t in m.T),
                            sense=minimize)
    ################################################################################################################
    ################################################################################################################
    # Solver
    # Open the LP file for writing
    m.write("my_model.lp", io_options={"symbolic_solver_labels": True})

    # solver = SolverFactory('glpk')

    # Using CBC (ensure you've installed CBC)
    solver = SolverFactory('cbc', executable='/opt/homebrew/bin/cbc')

    # Optional: Set the number of threads (`threads` option)
    solver.options['threads'] = 20  # Adjust based on the number of cores you want to use
    # # Enable verbose output
    # solver.options['mipgap'] = 1  # Example: set MIP gap tolerance to 1%
    solver.options['ratioGap'] = 10
    # solver.options['integerTolerance'] = 0.005  # Example: Allow a tolerance of 0.01
    # Adjust solver options
    solver.options['presolve'] = 'on'  # Ensure presolve is on
    solver.options['ratio'] = 0.05  # Set mip gap ratio, adjust as needed
    solver.options['primalTolerance'] = 1e-2  # Adjust feasibility tolerance
    solution = solver.solve(m, tee=True)
    ################################################################################################################
    ################################################################################################################
    # Construct a data structure for your results
    # Extract results
    results = []
    for v in m.V:
        vehicle_results = {'Vehicle': v}
        for t in m.T:
            x_chr_value = m.X_CHR[v, t].value if m.X_CHR[v, t].value is not None else 0
            x_cum_value = m.cumulative_charging[v, t].value if m.cumulative_charging[v, t].value is not None else 0
            soc_value = m.SOC[v, t].value if m.SOC[v, t].value is not None else 0
            batt_deg_value = m.batt_deg_cost[v, t].value if m.batt_deg_cost[v, t].value is not None else 0
            battery_value = m.bat_cap[v, t]
            fac_chr_value = m.fac_chr[v, t]

            electricity_cost = (m.CRTP[t] * x_chr_value) / 1000 if m.CRTP[t] is not None else 0
            degradation_cost = (batt_deg_value)
            ghg_emissions_cost = m.ghg_cost
            ghg_cost = 0  # Default value for ghg_emissions

            if x_chr_value > 0:  # Calculate ghg_emissions only when x_chr_value is positive
                ghg_cost = ((m.GHG[t] * x_chr_value) / 1000) * ghg_emissions_cost if m.GHG[t] is not None else 0
            vehicle_results[t] = {
                'X_CHR': x_chr_value,
                'X_CUM': x_cum_value,
                'SOC': soc_value,
                'Batt_cap': battery_value,
                'Electricity_Cost': electricity_cost,
                'Degradation_Cost': degradation_cost,
                'GHG_Cost': ghg_cost
            }
        results.append(vehicle_results)

    # Create DataFrame
    df = pd.DataFrame(results)

    # Reorganize DataFrame
    df = df.set_index('Vehicle').stack().apply(pd.Series).reset_index()
    df = df.round(10)
    df.columns = ['Vehicle', 'Hour', 'X_CHR', 'X_CUM', 'SOC', 'Batt_cap', 'Electricity_Cost', 'Degradation_Cost', 'GHG_Cost']

    # Group by Vehicle and sum the costs
    total_costs = df.groupby('Vehicle').agg({'Electricity_Cost': 'sum', 'Degradation_Cost': 'sum', 'GHG_Cost': 'sum', "X_CHR": "sum", 'Batt_cap': "first"}).reset_index()
    # Sum the columns Electricity_Cost, Degradation_Cost, and GHG_Emissions
    sum_values = total_costs.loc[:, ['Electricity_Cost', 'Degradation_Cost', 'GHG_Cost', "X_CHR"]].sum()

    # Create a new DataFrame with the sum values
    sum_df = pd.DataFrame(sum_values, columns=['Total'])

    # Transpose the DataFrame for better readability
    sum_df = sum_df.T
    # Create ExcelWriter object
    # Determine the appropriate Excel file name based on x_chr_domain
    if x_chr_domain == NonNegativeReals:
        excel_file_name = f"4BEV_smart_{charging_speed}g_{ghg_cost_per_tonne}kw_{locs}_{price_name}.xlsx"
    else:
        excel_file_name = f"4BEV_v2g_{charging_speed}g_{ghg_cost_per_tonne}kw_{locs}_{price_name}.xlsx"
    # Create the JSON file name by replacing the .xlsx extension with .json
    json_file_name = excel_file_name.replace('.xlsx', '.json')

    # Print the current working directory
    current_directory = os.getcwd()
    print(f"Current working directory: {current_directory}")

    try:
        # Create ExcelWriter object
        with pd.ExcelWriter(excel_file_name) as writer:

            # # Write the total costs to a sheet named 'Individual Costs'
            # df.to_excel(writer, sheet_name='hourly data', index=False)

            # Write the total costs to a sheet named 'Individual Costs'
            total_costs.to_excel(writer, sheet_name='Individual Costs', index=False)

            # Write the total costs to a sheet named 'Total Costs'
            sum_df.to_excel(writer, sheet_name='Total Costs', index=False)

        # Save 'df' to a JSON file
        df.to_json(json_file_name, orient="records")

        # Confirm the files were saved
        print(f"Excel file saved: {os.path.join(current_directory, excel_file_name)}")
        print(f"JSON file saved: {os.path.join(current_directory, json_file_name)}")

        return excel_file_name, json_file_name

    except ValueError as e:
        if "This sheet is too large" in str(e):
            print("Data exceeds Excel sheet size limits. Please reduce the size of the data.")
        else:
            print("An error occurred:", e)


# %% Run the code for smart charging
# Define the charging speeds and GHG costs

x_chr_domain = [Reals]
charging_speeds = [6.6, 12, 19]
ghg_costs = [0]
locations = [["Home"], ["Home", "Work"]]
prices = {"TOU_pge": tou_prices_pge, "EV_Rate_pge": ev_rate_prices_pge, "RT_Rate_pge": rt_rate_pge,
          "TOU_sce": tou_prices_sce, "EV_Rate_sce": ev_rate_prices_sce, "RT_Rate_sce": rt_rate_sce,
          "TOU_sdge": tou_prices_sdge, "EV_Rate_sdge": ev_rate_prices_sdge, "RT_Rate_sdge": rt_rate_sdge,
          "TOU_smud": tou_prices_smud, "EV_Rate_smud": ev_rate_prices_smud, "RT_Rate_smud": rt_rate_smud,
          }
prices_commercial = {"commercial_pge": commercial_prices_pge, "commercial_sce": commercial_prices_sce,
                     "commercial_sdge": commercial_prices_sdge, "commercial_smud": commercial_prices_smud
                     }

# Iterate over all combinations
for domain in x_chr_domain:
    for speed in charging_speeds:
        for cost in ghg_costs:
            for loc in locations:
                for name, ep in prices.items():
                    # Identify the region based on the price name
                    if "pge" in name.lower():
                        commercial_price = prices_commercial.get("commercial_pge")
                    elif "sce" in name.lower():
                        commercial_price = prices_commercial.get("commercial_sce")
                    elif "sdge" in name.lower():
                        commercial_price = prices_commercial.get("commercial_sdge")
                    elif "smud" in name.lower():
                        commercial_price = prices_commercial.get("commercial_smud")
                    else:
                        if isinstance(ep, dict):
                            commercial_price = {t: 420 for t in ep.keys()}  # Default price of 420
                        else:
                            raise TypeError(f"`ep` must be a dictionary for default commercial price. Found {type(ep)}")

                    # Validate price formats
                    if not isinstance(ep, dict):
                        raise TypeError(f"Price input `ep` must be a dictionary. Found type: {type(ep)} for {name}")
                    if not isinstance(commercial_price, dict):
                        raise TypeError(f"Commercial price input must be a dictionary. Found type: {type(commercial_price)}")

                    # Call the function with correct arguments
                    excel_file, json_file = create_model_and_export_excel(
                        charging_speed=speed,
                        ghg_cost_per_tonne=cost,
                        x_chr_domain=domain,
                        locs=loc,
                        price=ep,  # The selected price (TOU, EV Rate, or RT)
                        commercial_prices=commercial_price,  # Pass the Commercial Rate
                        price_name=name,
                    )

                    # Print confirmation message
                    print(f"File '{excel_file}' has been created with charging speed {speed} kW and GHG cost ${cost} per tonne, V2G at {loc}_{name}.")

