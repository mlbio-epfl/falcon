import gurobipy as gp
from gurobipy import GRB
import numpy as np
import torch
import math
import os

def print_status(model):
    if model.status == gp.GRB.OPTIMAL:
        print("Optimal solution found.")
    elif model.status == gp.GRB.TIME_LIMIT:
        print("Time limit reached. Best known solution is available.")
    else:
        print("Optimization terminated with status", model.status)

    print('MIP Gap:', model.MIPGap)

 # time limit in seconds
def compute_assignment_from_cost(cost, reg_coef=1.0, time_limit=60, detailed_log=False):
    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag', 0) if not detailed_log else env.setParam('LogToConsole', 1)

        wlsaccessID = os.getenv('GRB_WLSACCESSID', None)
        env.setParam('WLSACCESSID', wlsaccessID) if wlsaccessID is not None else None
        licenseID = os.getenv('GRB_LICENSEID', None)
        env.setParam('LICENSEID', int(licenseID)) if licenseID is not None else None
        wlsSecrets = os.getenv('GRB_WLSSECRET', None)
        env.setParam('WLSSECRET', wlsSecrets) if wlsSecrets is not None else None

        env.start()

        model = gp.Model(env=env)

        model.params.TimeLimit = time_limit
        # model.setParam("MIPGap", mipgap)

        fine_classes, coarse_classes = cost.shape
        assert fine_classes > coarse_classes


        assignments = model.addMVar((fine_classes, coarse_classes), vtype=GRB.BINARY)

        cls_objective = (- cost * assignments).sum()
        set_sizes = assignments.sum(axis=0)
        reg_objective = (set_sizes * set_sizes).sum() / coarse_classes - ((fine_classes / coarse_classes) ** 2)

        objective = cls_objective + reg_coef * reg_objective
        model.setObjective(objective, GRB.MINIMIZE)

        model.addConstr(assignments.sum(axis=1) == np.full(fine_classes, 1))
        model.addConstr(assignments.sum(axis=0) >= np.full(coarse_classes, 1))

        # Optimize the model
        model.optimize()

        # Check the optimization status
        print_status(model) if detailed_log else None

    return torch.from_numpy(assignments.x).float()