import numpy as np
import pyomo.environ as pyo

# Create macc
# Logistic sigmoid function
def logistic(x, L=1, x_0=0, k=1):
    return L / (1 + np.exp(-k * (x - x_0)))

c = np.linspace(0, 100, 6)
macc = 2000 * logistic(c, L=0.5, x_0=60, k=0.02)
macc = macc - macc[0]

s0 = 800
b0 = 1000
tnac0 = 100

cp0 = 10
ab0 = 100
model = pyo.ConcreteModel()

# Declare sets
model.V = pyo.Set(initialize=['A', 'B'])  # Declare set for index V
model.T = pyo.RangeSet(0, 10)  # Declare set for index T

# Declare variables with two indices
model.x = pyo.Var(model.V, model.T, domain=pyo.NonNegativeReals, bounds=(5, 395), initialize=cp0)
model.y = pyo.Var(model.V, model.T)

# Declare parameters
model.s = pyo.Param(initialize=s0, within=pyo.NonNegativeReals)
model.b = pyo.Param(initialize=b0, within=pyo.NonNegativeReals)
model.tnac = pyo.Param(initialize=tnac0, within=pyo.NonNegativeReals)

# Piecewise constraints
# Piecewise constraints
def piecewise_rule(model, v, t):
    return model.y[v, t] == sum((macc[i+1] - macc[i]) / (c[i+1] - c[i]) * (model.x[v, t] - c[i]) + macc[i] for i in range(len(c)-1) if c[i] <= model.x[v, t] <= c[i+1])

model.piecewise_constraint = pyo.Constraint(model.V, model.T, rule=piecewise_rule)


# Declare objective function
model.Obj = pyo.Objective(expr=sum(model.b * model.x[v, t] - model.y[v, t] * model.x[v, t] for v in model.V for t in model.T), sense=pyo.minimize)

# Solve the model
solver = pyo.SolverFactory('ipopt')
solution = solver.solve(model, tee=True)