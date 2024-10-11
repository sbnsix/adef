from __future__ import annotations

import os
import sys
import ast
import json
import copy
import re
import pandas as pd

# SAT solver - integer variables
from ortools.sat.python import cp_model

# GLOP linear solver - float variables
from ortools.linear_solver import pywraplp
from pulp import LpMaximize, LpProblem, LpStatus, lpSum, LpVariable, LpStatusOptimal
from pulp import PULP_CBC_CMD


PATHS = ["./", "./detectors", "./tools", "./threats", "./metrics"]
for path in PATHS:
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), path))
    if path not in sys.path:
        sys.path.insert(1, path)

# ---------------------------------------------
#                Local imports
# ---------------------------------------------
from tools.log import Log
from json_helper import NpEncoder
from tools.iter_helper import IterHelper


class CPSelector:
    def __init__(self, cfg: dict, logger: Log) -> None:
        """
        CTOR
        Args:
            cfg - Model selector configuration
            logger - logger object
        Returns:
            <None>
        """
        self.log = logger
        self.cfg = cfg
        # Define the model
        self.model = cp_model.CpModel()
        self.multiplier = 100

        # Convert the problem to integer domain as this is non-linear problem
        # SAT solver family have to be used in the process and this is approximated solution
        # due to float -> int -> float conversion.
        self.data_columns = ["auc", "acc", "f1", "eer"]
        self.cd_data_columns = ["auc", "acc", "f1", "eer", "tau"]
        self.default_solution = {
            "tau": -1,
            "tp": 0,
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "fpr": 0.5,
            "tpr": 0.5,
            "recall": 1,
            "prec": 0,
            "auc": 0.5,
            "cfg": {},
            "model": "",
            "file": "",
            "acc": 0.5,
            "eer": 0.5,
            "f1": 0.5,
            "fbeta": 0,
            "support": 0,
            "d1_min": 0,
            "d1_avg": 0,
            "d1_max": 0
        }

    def select(self, data: pd.DataFrame) -> dict:
        """
        Method selects best model/point in the model based on given input data
        and preconfigured selection criteria.
        Args:
            data: input data containing information about models
        Returns:
            <dict> - best model selection with its performance information -
            best row of supplied data converted to dictionary
        """
        def replace_numbers(input_string):
            pattern = r'^(m[\d]+|p[\d]+)'
            return re.sub(pattern, lambda x: re.sub(r'\d', '0', x.group()), input_string)

        # Down select points to 0, 0.5, 0.5, 0.5, 0.5, 1, 0, 1 box
        data_cp = data[data.apply(lambda row: isinstance(row["file"], str) and
                                  row["fpr"] < self.cfg["cp_selection"]["optimal_area"]["fpr"] and
                                  row["tpr"] > self.cfg["cp_selection"]["optimal_area"]["tpr"],
                                  axis=1)]

        result = data_cp[data_cp["auc"] == data_cp["auc"].max()]
        # result = data_cp[data_cp["eer"] == data_cp["eer"].min()]

        # Empty - default 0.5 solution
        if result.empty or result.shape[0] == 0:
            result = copy.copy(self.default_solution)
            return result

        result = result.iloc[0]
        result = ast.literal_eval(result.to_json().replace("\\/", "/"))

        return result

    def cd_select(self, data: pd.DataFrame) -> dict:
        """
        Method selects best model/point in the model based on given input data
        and preconfigured selection criteria.
        Args:
            data: input data containing information about models
        Returns:
            <dict> - best model selection with its performance information -
            best row of supplied data converted to dictionary
        """
        # Convert the problem to integer domain as this is non-linear problem
        # SAT solver family have to be used in the process.
        data_cp = data[data.apply(lambda row: isinstance(row["file"], str) and
                                  row["tau"] > -1 and
                                  row["fpr"] < self.cfg["cp_selection"]["optimal_area"]["fpr"] and
                                  row["tpr"] > self.cfg["cp_selection"]["optimal_area"]["tpr"],
                                  axis=1)]

        result = data_cp[data_cp["auc"] == data_cp["auc"].max()]
        # result = data_cp[data_cp["eer"] == data_cp["eer"].min()]

        # Empty - default 0.5 solution
        if result.empty or result.shape[0] == 0:
            result = copy.copy(self.default_solution)
            return result

        result = result.iloc[0]
        result = ast.literal_eval(result.to_json().replace("\\/", "/"))
        return result

        '''
        resolved_idx = 0
        data_cp = data_cp[~data_cp[self.cd_data_columns].isna().any(axis=1)]

        # data_cp *= self.multiplier
        data_cp["index"] = [i for i in range(len(data_cp))]
        # data_cp = data_cp.astype(int)
        # data_cp.reset_index(inplace=True)
        selected_columns = list(set(data.columns) - set(self.cd_data_columns))
        result = {}

        data_cp = data_cp.loc[:, ["index"] + self.cd_data_columns].copy(deep=True)
        data_cp.set_index("index", inplace=True)

        try:
            result = data_cp[(data_cp["auc"] == data_cp["auc"].max()) &
                             (data_cp["eer"] == data_cp["eer"].min())]

            if not result.empty and result.shape[0] > 0:
                if result.index.tolist()[0] in data.index.tolist():
                    resolved_idx = result.index.tolist()[0]
                else:
                    resolved_idx = data.index.tolist()[result["index"]]
                    self.log.warn(
                        f"Current: {json.dumps(data.loc[resolved_idx, selected_columns].to_json(), indent=4)}")
                    r_idx = data.index.tolist()[result["index"] - 1]
                    self.log.warn(f"Prev: {json.dumps(data.loc[r_idx, selected_columns].to_json(), indent=4)}")
                result = result.iloc[0]
                result = ast.literal_eval(result.to_json())
                result.update(eval(data.loc[resolved_idx, selected_columns].to_json().replace("\\/", "/")))
            else:
                idx = data[data["cfg"] != 0].index[0]
                result = copy.copy(self.default_solution)
                result.update({"tau": 0,
                               "model": data["model"][idx],
                               "file": data["file"][idx],
                               "cfg": data["cfg"][idx]})

        except Exception as ex:
            self.log.warn("================ Data CP ==============")
            self.log.warn(f"Data CP shape: {data_cp.shape}")
            self.log.warn("=======================================")
            self.log.exception(ex)
        '''
        # TODO: Solver with positive and negative gains is now working properly - investigate and fix this
        '''
        try:
            # Create the problem
            problem = LpProblem("AUC_Model", LpMaximize)

            # Create decision variables
            x = LpVariable.dicts("auc", range(len(data_cp)), 0, 1, cat="Integer")

            # Objective function
            problem += lpSum(
                [
                    data_cp.loc[i, "auc"] * x[i]
                    + data_cp.loc[i, "acc"] * x[i]
                    + data_cp.loc[i, "f1"] * x[i]
                    - data_cp.loc[i, "eer"] * x[i]
                    - data_cp.loc[i, "tau"] * x[i]
                    for i in range(len(data_cp))
                ]
            )

            # Constraint
            problem += lpSum([x[i] for i in range(len(data_cp))]) == 1

            # Solve the problem
            #  msg = True use only for debugging problems
            status = problem.solve(PULP_CBC_CMD(msg=False))
            # Print the status of the solution
            # self.log.debug(f"AUC solution status: {LpStatus[status]}")

            # {0: 'Not Solved', 1: 'Optimal', -1: 'Infeasible', -2: 'Unbounded', -3: 'Undefined'}
            # Worst solution - solver criteria not satisfied - manual selection is performed instead
            result = {}

            if status != LpStatusOptimal:
                self.log.warn(
                    f"Solver unable to find correct solution: Status<{status}>"
                )
                result = ast.literal_eval(
                    data_cp[data_cp["auc"] == data_cp["auc"].max()].to_json()
                )
                result["index"] = result.index[0]
            else:
                solution_idx = [i for i in range(len(x.keys())) if x[i].value() == 1]
                solutions = data_cp.iloc[solution_idx]
                solutions = solutions.rename(columns={"Unnamed: 0": "index"})
                solution = solutions[solutions["auc"] == solutions["auc"].max()]
                # Best solution
                if not solution.empty:
                    try:
                        result = ast.literal_eval(
                            solution.loc[solution.index[0], :].to_json()
                            .replace("false", "False")
                            .replace("true", "True")
                        )
                    except Exception as ex:
                        self.log.exception(ex)
                    result["index"] = solution.index[0]
                # Worse solution
                elif not solutions.empty:
                    result = ast.literal_eval(
                        solutions.loc[solutions.index[0], :].to_json()
                    )
                    result["index"] = solutions.index[0]

            result_keys = ["auc", "acc", "f1", "eer"]
            result["tau"] = int(result["tau"] / multiplier)

            best_cfg = data[data["tau"] == result["tau"]]

            for key in result_keys:
                # Convert paths
                if isinstance(result[key], str):
                    result[key] = result[key].replace("\\/", "/")
                # Convert back to float
                elif isinstance(result[key], int):
                    result[key] = best_cfg[key][0]
        except Exception as ex:
            self.log.exception(ex)
        '''

        # self.log.debug(
        #     f"Optimal solution[{solution.index[0]}]: {json.dumps(result, indent=4, cls=NpEncoder)}"
        # )

        self.log.debug(f"CD OS: {json.dumps(result, indent=4, cls=NpEncoder)}")

        return result

    def cp_generic_select(self, data: pd.DataFrame, cfg: dict) -> dict:
        """
        Method selects best model/point in the model based on given input data
        and preconfigured selection criteria.
        Args:
            data: input data containing information about models
            cfg: configuration for CP SAT solver
        Returns:
            <dict> - best model selection with its performance information -
            best row of supplied data converted to dictionary
        """
        cfg = {
            "multiplier": 100,
            "columns": ["auc", "acc", "f1", "eer", "tau"],
            "result_columns": ["auc", "acc", "f1", "eer"],

        }

        multiplier = 100
        # Convert the problem to integer domain as this is non-linear problem
        # SAT solver family have to be used in the process.
        data_columns = ["auc", "acc", "f1", "eer", "tau"]

        data_cp = data.loc[:, cfg["columns"]].copy(deep=True)
        data_cp = data_cp[data_cp["tau"] > 0]
        data_cp = data_cp[~data_cp[cfg["columns"]].isna().any(axis=1)]
        data_cp *= multiplier
        data_cp["index"] = [i for i in range(len(data_cp))]
        data_cp = data_cp.astype(int)
        data_cp.reset_index(inplace=True)

        try:
            data_cp = data_cp.loc[:, ["index"] + cfg["columns"]].copy(
                deep=False
            )
            data_cp.set_index("index", inplace=True)

            # Create the problem
            problem = LpProblem("AUC_Model", LpMaximize)

            # Create decision variables
            x = LpVariable.dicts("auc", range(len(data_cp)), 0, 1, cat="Integer")

            # Objective function
            problem += lpSum(
                [
                    data_cp.loc[i, "auc"] * x[i]
                    + data_cp.loc[i, "acc"] * x[i]
                    + data_cp.loc[i, "f1"] * x[i]
                    - data_cp.loc[i, "eer"] * x[i]
                    - data_cp.loc[i, "tau"] * x[i]
                    for i in range(len(data_cp))
                ]
            )

            # Constraint
            problem += lpSum([x[i] for i in range(len(data_cp))]) == 1

            # Solve the problem
            #  msg = True use only for debugging problems
            status = problem.solve(PULP_CBC_CMD(msg=False))
            # Print the status of the solution
            # self.log.debug(f"AUC solution status: {LpStatus[status]}")

            # {0: 'Not Solved', 1: 'Optimal', -1: 'Infeasible', -2: 'Unbounded', -3: 'Undefined'}
            # Worst solution - solver criteria not satisfied - manual selection is performed instead
            result = {}

            if status != LpStatusOptimal:
                self.log.warn(
                    f"Solver unable to find correct solution: Status<{status}>"
                )
                result = ast.literal_eval(
                    data_cp[data_cp["auc"] == data_cp["auc"].max()].to_json()
                )
                result["index"] = result.index[0]
            else:
                solution_idx = [i for i in range(len(x.keys())) if x[i].value() == 1]
                solutions = data_cp.iloc[solution_idx]
                solutions = solutions.rename(columns={"Unnamed: 0": "index"})

                solution = solutions[solutions["auc"] == solutions["auc"].max()]
                # Best solution
                if not solution.empty:
                    try:
                        result = ast.literal_eval(
                            solution.loc[solution.index[0], :].to_json()
                            .replace("false", "False")
                            .replace("true", "True")
                        )
                    except Exception as ex:
                        self.log.exception(ex)
                    result["index"] = solution.index[0]
                # Worse solution
                elif not solutions.empty:
                    result = ast.literal_eval(
                        solutions.loc[solutions.index[0], :].to_json()
                    )
                    result["index"] = solutions.index[0]

            result_keys = ["auc", "acc", "f1", "eer"]
            result["tau"] = int(result["tau"] / multiplier)

            best_cfg = data[data["tau"] == result["tau"]]

            for key in result_keys:
                # Convert paths
                if isinstance(result[key], str):
                    result[key] = result[key].replace("\\/", "/")
                # Convert back to float
                elif isinstance(result[key], int):
                    result[key] = best_cfg[key][0]

            # self.log.debug(
            #     f"Optimal solution[{solution.index[0]}]: {json.dumps(result, indent=4, cls=NpEncoder)}"
            # )

        except Exception as ex:
            self.log.exception(ex)

        return result


'''
        if data_cp.shape[0] == 0:
            # Default 0.5 solution
            result = copy.copy(self.default_solution)
            # TODO: figure out roc and model file
            # TODO: Replace model and file name parameters with 0
            result["file"] = data.loc[data.index[1], "file"][:-4]

            x = [replace_numbers(x) for x in result["file"].split("_")]
            x = "_".join(x)

            names = IterHelper.extract_names(result["file"])
            algo_name = names["algo_name"]
            ext = self.cfg["model"]["ad"][algo_name]["ext"]

            result["file"] = f"{x}.csv"
            result["model"] = f"{x}.{ext}"
            self.log.warn("Empty solution, applying default AUC = 0.5")

        # Single/Multiple solutions found in optimal space
        elif data_cp.shape[0] >= 1:
            # solution = data_cp[data_cp["auc"] == data_cp["auc"].max()]
            solution = data_cp[data_cp["eer"] == data_cp["eer"].min()]
            result = eval(solution.loc[solution.index[0], :].to_json().replace("\\/", "/"))
'''
'''
            data_cp = data_cp.loc[:, self.data_columns]
            data_cp = data_cp[~data_cp[self.data_columns].isna().any(axis=1)]

            # EER update to adjust for the solver (1/eer) - 1
            # data_cp.loc[:, "eer"] = (1/data_cp.loc[:, "eer"]) - 1

            data_cp.loc[:, self.data_columns] *= self.multiplier
            data_cp = data_cp.astype(int)

            try:
                data_cp["index"] = [i for i in range(len(data_cp))]

                # data_cp.loc[:, data_columns] = data_cp.loc[:, data_columns].astype(int)
                data_cp.reset_index(inplace=True)
                data_cp.set_index("index", inplace=True)

                # Create the problem
                problem = LpProblem("AUC_Model", LpMaximize)

                # Create decision variables
                x = LpVariable.dicts("auc", range(len(data_cp)), 0, 1, cat="Integer")

                # TODO: Resolve dynamic approach for CP variables from configuration
                # Objective function
                problem += lpSum(
                    [
                        data_cp.loc[i, "auc"] * x[i]
                        # + data_cp.loc[i, "acc"] * x[i]
                        #+ data_cp.loc[i, "f1"] * x[i]
                        # - data_cp.loc[i, "eer"] * x[i]
                        for i in range(len(data_cp))
                    ]
                )

                # Constraint
                problem += lpSum([x[i] for i in range(len(data_cp))]) == 1

                # Solve the problem
                #  msg = True use only for debugging problems
                status = problem.solve(PULP_CBC_CMD(msg=False))
                # Print the status of the solution
                # self.log.debug(f"AUC solution status: {LpStatus[status]}")

                # {0: 'Not Solved', 1: 'Optimal', -1: 'Infeasible', -2: 'Unbounded', -3: 'Undefined'}
                # Worst solution - solver criteria not satisfied - manual selection is performed instead
                if status != LpStatusOptimal:
                    self.log.warn(
                        f"Solver unable to find correct solution: Status<{status}>"
                    )
                    result = ast.literal_eval(
                        data_cp[data_cp["auc"] == data_cp["auc"].max()].to_json()
                    )
                    result["index"] = result.index[0]
                else:
                    solution_idx = [i for i in range(len(x.keys())) if x[i].value() == 1]
                    solutions = data_cp.iloc[solution_idx]
                    solutions = solutions.rename(columns={"Unnamed: 0": "index"})
                    solution = solutions[solutions["auc"] == solutions["auc"].max()]

                    # Best solution
                    if not solution.empty:
                        result = ast.literal_eval(
                            solution.loc[solution.index[0], :].to_json()
                            .replace("false", "False")
                            .replace("true", "True")
                        )
                        result["index"] = solution.index[0]
                    # Worse solution
                    elif not solutions.empty:
                        result = ast.literal_eval(
                            solutions.loc[solutions.index[0], :].to_json()
                        )
                        result["index"] = solutions.index[0]

                for key in self.data_columns:
                    result[key] /= self.multiplier

                # self.log.debug(
                #     f"Optimal solution[{solution.index[0]}]: {json.dumps(result, indent=4, cls=NpEncoder)}"
                # )

            except Exception as ex:
                self.log.exception(ex)

            # Augment solution with the rest of the columns from data set containing winning configuration set and model
            # data
            selected_columns = list(set(data_x.columns) - set(self.data_columns))
            if result["index"] in data_x.index.tolist():
                resolved_idx = result["index"]
            else:
                resolved_idx = data_x.index.tolist()[result["index"]]
                self.log.warn(f"Current: {json.dumps(data_x.loc[resolved_idx, selected_columns].to_json(), indent=4)}")
                r_idx = data_x.index.tolist()[result["index"]-1]
                self.log.warn(f"Prev: {json.dumps(data_x.loc[r_idx, selected_columns].to_json(), indent=4)}")

            result.update(eval(data_x.loc[resolved_idx, selected_columns].to_json().replace("\\/", "/")))
'''