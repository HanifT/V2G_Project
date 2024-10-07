import requests
import pandas as pd
import zipfile
import io
from pyomo.environ import ConcreteModel, Set, Param, Var, Objective, Constraint, SolverFactory, Reals
# %%