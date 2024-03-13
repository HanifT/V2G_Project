import pyomo.environ as pyo
from pyomo.environ import *


class OptimalChargingModel:
    def __init__(self, period):
        self.model = ConcreteModel()
        self.period = period

        # Parameters
        self.model.c_t_RTP = Param(self.period, initialize=0)  # Price at time t// Will update based on our inputs
        self.model.battery_capacity = Param(initialize=100)  # Battery capacity // Will update based on our inputs
        self.model.min_charge_rate = Param(initialize=-6.6)  # Minimum charging rate
        self.model.max_charge_rate = Param(initialize=6.6)  # Maximum charging rate
        self.model.soc_req = Param(initialize=30)  # Minimum SOC required at departure time  // Will update based on our inputs
        self.model.soc_buffer = Param(initialize=20)  # SOC buffer during energy dispensing

        # Variables
        self.model.x_chr = Var(self.period, within=pyo.Reals)  # Charging rate at time t
        self.model.x_soc = Var(self.period, within=pyo.NonNegativeReals, bounds=(0, 100))  # SOC at time t

        # Objective function
        self.model.obj = Objective(rule=self.objective_rule, sense=minimize)

        # Constraints
        self.model.soc_update = Constraint(self.period, rule=self.soc_update_rule)
        self.model.charge_rate_bounds = Constraint(self.period, rule=self.charge_rate_bounds_rule)
        self.model.departure_soc = Constraint(self.period, rule=self.departure_soc_rule)
        self.model.soc_buffer = Constraint(self.period, rule=self.soc_buffer_rule)

    def objective_rule(self, model):
        return sum(model.c_t_RTP[t] * model.x_chr[t] for t in self.period)

    def soc_update_rule(self, model, t):
        if t == self.period.first():
            return model.x_soc[t] == 0
        else:
            return model.x_soc[t] == model.x_soc[t - 1] + (model.x_chr[t] / model.battery_capacity) * 100

    def charge_rate_bounds_rule(self, model, t):
        return model.min_charge_rate, model.x_chr[t], model.max_charge_rate

    def departure_soc_rule(self, model, t):
        if t == self.period.last():
            return model.x_soc[t] >= model.soc_req
        else:
            return Constraint.Skip

    def soc_buffer_rule(self, model, t):
        return model.x_soc[t] >= model.soc_buffer

    def solve(self):
        solver = SolverFactory('glpk')
        solver.solve(self.model)

        # Display results
        for t in self.period:
            print(f"periodime period {t}: Charge rate = {value(self.model.x_chr[t])}, SOC = {value(self.model.x_soc[t])}")


# Create model instance and solve
period = range(1, 25)  # periodime periods
charging_model = OptimalChargingModel(period)
charging_model.solve()
