# Define binary variables to represent each segment of the piecewise function
m.dod_initial = Param(m.K, initialize=points2["DOD"].to_dict())
m.dod_final = Param(m.K, initialize=points2["BD"].to_dict())
m.slope = Param(m.K, initialize=points2["slope"].to_dict())
m.delta = Param(m.K, initialize=100 / 5)
m.dod_var1 = Var(m.V, m.T, domain=NonNegativeReals, bounds=(0, 19.99))
m.segment = Var(m.V, m.T, m.K, domain=Binary)

def depth_of_discharge(m, v, t):

    if m.fac_chr[v, t] == 1:
        return m.DOD[v, t] == 100 - m.SOC[v, t]
    else:
        return Constraint.Skip


m.depth_decharge_constrain = Constraint(m.V, m.T, rule=depth_of_discharge)


def batt_degradation0(m, v, t):
    if m.fac_chr[v, t] == 1:
        return sum(m.segment[v, t, k] for k in m.K) == 1
    else:
        return Constraint.Skip  # Skip the constraint if m.fac_chr[v, t] is not equal to 1


m.batt_degradation_constraint0 = Constraint(m.V, m.T, rule=batt_degradation0)


def batt_degradation1(m, v, t):
    if m.fac_chr[v, t] == 1:
        return sum((m.dod_var1[v, t] + m.dod_initial[k] * m.segment[v, t, k]) for k in m.K) == 100 - m.SOC[v, t]
    else:
        return Constraint.Skip  # Skip the constraint if m.fac_chr[v, t] is not equal to 1


m.batt_degradation_constraint1 = Constraint(m.V, m.T, rule=batt_degradation1)

a = list(range(0, 5))


def batt_degradation2(m, v, t):
    if m.fac_chr[v, t] == 1:
        return sum(m.slope[k + 1] * (m.dod_var1[v, t]) + m.dod_final[k] * m.segment[v, t, k] for k in a) == m.rho[v, t]
    else:
        return Constraint.Skip  # Skip the constraint if m.fac_chr[v, t] is not equal to 1


m.batt_degradation_constraint2 = Constraint(m.V, m.T, rule=batt_degradation2)


def batt_degradation3(m, v, t):
    if m.fac_chr[v, t] == 1:
        return m.batt_deg[v, t] == (m.rho[v, t - 1] - m.rho[v, t])
    else:
        return Constraint.Skip  # Skip the constraint if m.fac_chr[v, t] is not equal to 1


m.batt_degradation_constraint3 = Constraint(m.V, m.T, rule=batt_degradation3)

m.abs1 = Var(m.V, m.T, domain=NonNegativeReals)
m.abs2 = Var(m.V, m.T, domain=NonNegativeReals)


def batt_degradation4(m, v, t):
    if m.fac_chr[v, t] == 1:
        return m.batt_deg[v, t] == (m.abs1[v, t] - m.abs2[v, t])
    else:
        return Constraint.Skip  # Skip the constraint if m.fac_chr[v, t] is not equal to 1


m.batt_degradation_constraint4 = Constraint(m.V, m.T, rule=batt_degradation4)

# Print results
# Open a text file for writing
# with open('charging_schedule.txt', 'w') as file:
#     file.write("Optimal charging schedule:\n")
#
#     # Loop through vehicles and time periods
#     for v in m.V:
#         for t in m.T:
#             charge_value = m.X_CHR[v, t].value if m.X_CHR[v, t].value is not None else 0.0
#             soc_value = m.SOC[v, t].value if m.SOC[v, t].value is not None else 0.0
#             DOD_value = m.DOD[v, t].value if m.DOD[v, t].value is not None else 0.0
#             DOD_var = m.dod_var1[v, t].value if m.dod_var1[v, t].value is not None else 0.0
#             rho_value = m.rho[v, t].value if m.rho[v, t].value is not None else 0.0
#             deg_value = m.batt_deg[v, t].value if m.batt_deg[v, t].value is not None else 0.0
#             charge_level = m.lev_chr[v, t]
#             # Write the output to the file
#             file.write(f"Vehicle {v} in Time period {t}: Charge = {charge_value:.2f} kWh, SOC = {soc_value:.2f}%,"
#                        f" DOD = {DOD_value:.2f}%,"
#                        f" DOD = {DOD_var:.2f}%,"
#                        f" rho = {rho_value:.9f},"
#                        f" deg = {deg_value:.9f},"
#                        f" charge_type = {charge_level}\n")
#
# # Print a message indicating that the data has been written to the file
# print("Optimal charging schedule written to charging_schedule.txt")