""" ICS autoclave profile generation to enable physics based simulation. """
from __future__ import annotations

import gc
import json
import os
import copy
import random
import threading
import concurrent.futures
from io import StringIO
import shutil
import glob
import numpy as np
import pandas as pd

from tools.roc_util import RocUtil
from tools.log import Log

from tools.plotter_helper import Plotter
from mev.metrics.m_delta import MetricDelta
from mev.metrics.m_delta import MetricD2

from mev.metrics.met_point import AnomalyDetectionMetricsPerPoint
from mev.metrics.met_model import AnomalyDetectionMetricsPerModel

from tools.cp_selector import CPSelector
from tools.iter_helper import IterHelper


class CycleDetector:
    """
    Cycle detector class - implementation of de-noising part of the
    simulation for statistical based anomaly detection models.
    """

    def __init__(self, config: dict, meta_params: dict, logger: Log) -> None:
        """
        CTOR
        Args:
            config              - cycle detector configuration
            meta_params         - meta parameters' for AD models
            logger			    - logger object
        Returns:
            <cycle_detector>    - instance of the class <cycle_detector>
        """
        self.log = logger
        self.cfg = config
        self.empty_df = pd.DataFrame()
        self.path = f"{config['global']['path']}experiments/{config['experiment']['path']}"

        # CP solver that will select best model/points in the selection process
        self.cp_selector = CPSelector(self.cfg, self.log)

        self.meta_params = meta_params
        self.cycle_len = (
            self.cfg["process"]["cycle_len"] * self.cfg["process"]["samples"]
        )
        self.best_cfg = {}

        # Length of the cycle in data samples
        self.start = int(
            self.cycle_len * (self.cfg["cycle_detector"]["tau"][0] / self.cycle_len)
        )
        self.stop = int(
            self.cycle_len * (self.cfg["cycle_detector"]["tau"][1] / self.cycle_len)
        )
        self.step = self.cfg["cycle_detector"]["step"]

        self.mutex = threading.Lock()

        self.warning_displayed = False

        self.d2 = MetricD2("step5", self.cfg, self.log)
        self.p_met = AnomalyDetectionMetricsPerPoint("step5", self.cfg, self.log)
        self.m_met = AnomalyDetectionMetricsPerModel("step5", self.cfg, self.log)

        self.min_fscore = 2.0
        self.best_auc = 0
        self.best_acc = 0
        self.best_tau = 0
        self.best_roc_cd = None
        self.best_data_list = None
        self.best_f1_score = 0.0

        self.f1_crossed = False

        # Parallel stop loop - it is being used to optimize amount of tau values to be computed
        # in case where both FPR and TPR reach 0
        self.parallel_stop = False
        self.fpr_tpr_zero_cnt = 0
        self.fpr_tpr_one_cnt = 0

        self.p_results = pd.DataFrame()
        self.roc_data = pd.DataFrame()
        self.enable_line = False

        self.clear_state(True)

        self.cd_ts_default = pd.DataFrame()

        cd_pred = self.cfg["data_labels"]["cd"]["prediction"]
        cd_gt = self.cfg["data_labels"]["cd"]["ground_truth"]
        cycle_len = self.cfg["process"]["cycles"]
        sample_size = self.cfg["process"]["cycle_len"] * self.cfg["process"]["samples"]

        self.roc_data_default = pd.DataFrame({
            "tau": [0, 0],
            "tp": [0, 0],
            "tn": [0, 0],
            "fp": [0, 0],
            "fn": [0, 0],
            "fpr": [0, 1],
            "tpr": [0, 1],
            "recall": [1, 0],
            "prec": [0, 1],
            "auc": [0.5, 0.5],
            "cfg": [{}, {}],
            "model": ["", ""],
            "file": ["", ""],
            "acc": [0.5, 0.5],
            "eer": [0.5, 0.5],
            "f1": [0.5, 0.5],
            "fbeta": [0, 0],
            "support": [0, 0],
            "d1_min": [0, 0],
            "d1_avg": [0, 0],
            "d1_max": [0, 0],
            "d2_min": [0, 0],
            "d2_avg": [0, 0],
            "d2_max": [0, 0]
        })

        # TODO: Debugging for experiment 2
        # self.log.warn(f"CD CTOR: cycle_len  : {cycle_len}")
        # self.log.warn(f"CD CTOR: sample_size: {sample_size}")
        # self.log.warn(f"CD CTOR: sample_size/cycle_len: {int(sample_size/cycle_len)}")

        self.roc_cd_default = pd.DataFrame(
            {
                "cycle": [i for i in range(0, cycle_len)],
                "tau": [i for i in np.arange(0, sample_size, int(sample_size/cycle_len))[:cycle_len]],
                cd_pred: [0 for i in range(0, cycle_len)],
                cd_gt: [0 for i in range(0, cycle_len)],
            },
            dtype="int",
        )
        self.roc_cd_default.set_index("cycle", inplace=True)

    def clear_state(self, enable_line: bool) -> None:
        """
        Method clears internal state of cycle detector object
        Args:
            enable_line: flag to determine whether to treat ROC curve as
                         continuous (plot curve as line or not)
        Returns:
            <None>
        """
        self.min_fscore = 2.0
        self.best_auc = 0
        self.best_acc = 0
        self.best_tau = 0
        self.best_roc_cd = None
        self.best_data_list = None
        self.best_f1_score = 0.0

        self.f1_crossed = False

        # Parallel stop loop - it is being used to optimize amount of tau values to be computed
        # in case where both FPR and TPR reach 0
        self.parallel_stop = False
        self.fpr_tpr_zero_cnt = 0
        self.fpr_tpr_one_cnt = 0

        self.p_results = pd.DataFrame()
        self.roc_data = pd.DataFrame(
            {
                "tau": pd.Series([], dtype="int"),
                "tp": pd.Series([], dtype="int"),
                "tn": pd.Series([], dtype="int"),
                "fp": pd.Series([], dtype="int"),
                "fn": pd.Series([], dtype="int"),
                "fpr": pd.Series([], dtype="float"),
                "tpr": pd.Series([], dtype="float"),
                "auc": pd.Series([], dtype="float"),
                "acc": pd.Series([], dtype="float"),
                "f1": pd.Series([], dtype="float"),
                "eer": pd.Series([], dtype="float"),
                "d1_min": pd.Series([], dtype="float"),
                "d1_avg": pd.Series([], dtype="float"),
                "d1_max": pd.Series([], dtype="float"),
                "d2_min": pd.Series([], dtype="float"),
                "d2_avg": pd.Series([], dtype="float"),
                "d2_max": pd.Series([], dtype="float"),
            }
        )

        self.enable_line = enable_line
        gc.collect()

    def get_tau_limit(self,
                      data: pd.DataFrame,
                      best_auc: float) -> int:
        """
        Method computes tau limit for ground truth data
        Args:
            data: input data containing cd label and cd_results columns
            best_auc: best observed AUC value in the trace
        Returns:
            <int> optimal tau limit value for given data
        """
        # tau_limit = (self.metric_data[(self.metric_data["prob"] == prob_set_str) &
        #                              (self.metric_data["algo"] == algo_name) &
        #                              (self.metric_data["attack"] == attack_name)]
        #             .loc[:, ["d1_min", "d1_avg", "d1_max"]].values[0].tolist())
        # self.best_models[prob_set_str][algo_name][attack_name]["file"]

        # Create a new column for chunk ids
        data["cycle_len"] = np.arange(data.shape[0]) // self.cycle_len

        cycle_cnt = 0
        cycle_errors = 0

        # Iterate through chunks

        for cycle_no, chunk in data.groupby("cycle_len"):
            if sum(chunk["label"]) > 0:
                cycle_errors += sum(chunk["label"])
            cycle_cnt += 1

        # TODO: Implement adaptive and configurable approach
        tau_limit = 0
        tau_limit_adaptive = int(cycle_errors / cycle_cnt)

        # Autonomous value for cycle limit
        if best_auc > 0.5:
            tau_limit = tau_limit_adaptive

        self.log.debug(f"Tau limit, Current = {tau_limit}, Adaptive = {tau_limit_adaptive}")

        return tau_limit

    @staticmethod
    def detect(data_list: list,
               ad_label: str,
               cd_label: str,
               tau: int = 0) -> list:
        """
        Method performs cycle detection ground truth evaluation
        and updates trace with new column marking attacks (1)
        or no attacks(0) in column called cd prediction column of the data set.
        Args:
            data_list: list of pd.DataFrames of autoclave cycles.
            ad_label: AD model per point label
            cd_label: CD model per point label
            tau: required delay after which cycle detector
                 raises alert (ground truth data).
        Returns:
            <list>  - detection for each cycle (0 - no alarm, 1 - anomaly detected).
        """
        trace = []

        cycle_no = 1

        # Convert tau time value to discrete time expressed
        # data measurement in time index
        for data_c in data_list:
            # Marking CD detection
            data_c[cd_label] = 0

            # Marking CD_LABEL - ground truth
            faulty_cycle = data_c[data_c.loc[:, ad_label] != 0]
            if faulty_cycle.shape[0] > tau:
                alert_start = faulty_cycle.index[tau] if tau > -1 else 0
                data_c.loc[alert_start:, cd_label] = 1
                trace.append(1)
            else:
                trace.append(0)
            cycle_no += 1

        return trace

    def detect_ground_truth(self,
                            data_list: list,
                            tau_gt_limit: int = 0) -> list:
        """
        Method performs cycle detection ground truth evaluation
        and updates trace with new column marking attacks (1)
        or no attacks(0) in column called cd prediction column of the data set.
        Args:
            data_list: list of pd.DataFrames of autoclave cycles.
            tau_gt_limit: required delay after which cycle detector
                          raises alert (ground truth data).
        Returns:
            <list>  - ground truth for each cycle (0 - no alarm, 1 - anomaly detected).
        """
        return CycleDetector.detect(data_list,
                                    self.cfg["data_labels"]["ad"]["ground_truth"],
                                    self.cfg["data_labels"]["cd"]["ground_truth"],
                                    tau_gt_limit)

    def detect_prediction(self,
                          data_list: list,
                          ground_truth: list,
                          tau: int) -> (pd.DataFrame, list):
        """
        Method performs per cycle anomaly prediction
        and updates trace with new column marking attacks (1)
        or no attacks(0) in column called cd prediction column of the data set.
        Args:
            data_list: list of pd.DataFrames of autoclave cycles.
            ground_truth: ground truth list of points
            tau: required delay after which cycle detector raises alert.
        Returns:
            <list>  - data list containing new updated cd prediction column with
                      cycle detector values.
            <pd.DataFrame> - detection over cycles.
            <list>  - predicted truth.
        """
        cd_gt = self.cfg["data_labels"]["cd"]["ground_truth"]
        cd_pred = self.cfg["data_labels"]["cd"]["prediction"]

        trace_pr = CycleDetector.detect(data_list,
                                        self.cfg["data_labels"]["ad"]["prediction"],
                                        self.cfg["data_labels"]["cd"]["prediction"],
                                        tau)

        roc_cd = pd.DataFrame(
            {
                "cycle": [i for i in range(1, len(data_list)+1)],
                "tau": [tau for i in np.arange(0, len(data_list))],
                cd_pred: trace_pr,
                cd_gt: ground_truth
            },
            dtype="int",
        )
        roc_cd.set_index("cycle", inplace=True)

        return roc_cd, trace_pr

    def roc_tau_point(self,
                      file_name: str,
                      attack_name: str,
                      ground_truth: list,
                      data_list: list,
                      tau: int
    ) -> None:
        """
        Methods computes single tau detection for cycle detector
        Args:
            file_name: data file name where detection results are stored
            attack_name: name of the attack
            ground_truth: list of ground truth detections performed for each cycle
            data_list: chopped data list ready for per cycle configuration (per point detection available inside
            tau: tau value
        Returns:
            <None>
        """
        trace_gt = ground_truth
        try:
            with self.mutex:
                (roc_cd, trace_pr) = self.detect_prediction(data_list, ground_truth, tau)

            # Double check if trace_gt has always two classes to perform
            # further computation otherwise the test reached limit
            # if sum(trace_gt) == 0 or sum(trace_gt) == len(trace_gt):
            #    return

            # Delta 2 computation
            params = {
                "max_search_limit": self.cfg["metrics"]["d1"]["max_search_limit"],
                "tau": tau,
                "data_list": data_list,
            }

            # TODO: This should be computed on single cycle
            d2_data = self.d2.run(self.empty_df, file_name, False, params)

            # Compute single TPR/FPR point
            # TODO: Validate this routine in terms of accuracy for the CF
            tn, fp, fn, tp = RocUtil.confusion_matrix(trace_gt, trace_pr)

            # out_file_test = f"{self.best_cfg['file'][:-4]}_roc_cd.csv"

            if roc_cd.isna().all().all():
                self.log.error(f"CD Point: ROC_CD is empty")
                return

            m_params = {"data_list": data_list,
                        "d2_data": d2_data}

            result_metric_row = {
                "tau": tau,
                "model": self.best_cfg["model"],
                "file": self.best_cfg["file"],
                "d1_min": self.best_cfg["d1_min"],
                "d1_avg": self.best_cfg["d1_avg"],
                "d1_max": self.best_cfg["d1_max"],
                "d2_min": d2_data["d2_min"],
                "d2_avg": d2_data["d2_avg"],
                "d2_max": d2_data["d2_max"],
                "roc_cd": roc_cd.to_json(),
            }
            result_metric_row.update(self.p_met.run(self.best_cfg["file"],
                                                    roc_cd,
                                                    self.best_cfg["cfg"],
                                                    m_params))

            self.log.debug(f"Tau: {tau} => TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp},"
                            f"FPR[{result_metric_row['fpr']:.2f}],"
                            f"TPR[{result_metric_row['tpr']:.2f}]",
                            end="\r")

            with self.mutex:
                self.roc_data = pd.concat(
                    [
                        self.roc_data,
                        pd.DataFrame([result_metric_row],
                                     columns=list(result_metric_row.keys())),
                    ]
                )

                # CD ROC for animations
                if self.cfg["model"]["animation"]:
                    Plotter.roc_2d(self.roc_data.loc[:, ["fpr", "tpr"]],
                                   self.roc_data.loc[0, :].to_json(),
                                   "CD ROC",
                                   f"{file_name[:-4]}_{tau:03d}_cdb.png",
                                   3.0,
                                   True)

        # TPR/FPR might be nan in narrow cases thus value will not be added
        except ValueError as ve_ex:
            self.log.error(f"Infeasible computation {tau}")
            self.log.error(f"Shape: {data_list[0].shape}")
            self.log.error(f"Columns: {data_list[0].columns}")
            self.log.error(f"Data: {data_list}")
            self.log.error(ve_ex)
            self.log.exception(ve_ex)
        except Exception as ex:
            self.log.exception(ex)

    def compute_gt(self,
                   data_list: list,
                   tau_limit: int) -> list:
        # Data cycles integrity check
        cnt = 0
        for d_data in data_list:
            if len(d_data) != self.cycle_len:
                self.log.error(f"Cycle [{cnt}] length incorrect {len(d_data)}!= {self.cycle_len}")
                cnt += 1

        # STEP 2. Compute ground truth
        ground_truth = self.detect_ground_truth(data_list, tau_limit)

        # TODO: Here build default ts with ground truth
        self.cd_ts_default = pd.DataFrame(
            {
                "tau": [0]*len(ground_truth),
                self.cfg["data_labels"]["cd"]["prediction"]: [0]*len(ground_truth),
                self.cfg["data_labels"]["cd"]["ground_truth"]: ground_truth,
            }
        )
        return ground_truth

    def roc_tau_loop(self,
                     attack_name: str,
                     data_list: list,
                     out_file: str,
                     enable_line: bool,
                     tau_limit: int,
                     ground_truth: list) -> dict:
        """
        Method iterates over Tau parameter and generates ROC curve
        based on best AD data.
        Args:
            file_name - input data CSV file with attack, detection labels
            attack_name - attack name used in the evaluation
            data_list - list of data sets obtained for each cycle
            out_file - output file with path where result will be written to CSV file
            enable_line: flag to determine how to draw plots
            tau_limit: Tau value used to build ground truth values
        Returns:
            <dict>: computed CD ROC metrics
        """
        self.clear_state(enable_line)
        if data_list is None:
            self.log.error("CD Detector - input data is empty")
            return None

        names = IterHelper.extract_names(out_file)
        if int(names["model_no"]) > 0 and int(names["parameter_no"]) > 0:
            # Iteration over Tau for ROC curve generation
            if self.cfg["cycle_detector"]["parallel"]:
                # Concurrent loop processing
                exps = []
                with concurrent.futures.ThreadPoolExecutor(max_workers=int(os.cpu_count() / 2)) as executor:
                    results = {executor.submit(self.roc_tau_point,
                                               out_file,
                                               attack_name,
                                               ground_truth,
                                               data_list,
                                               tau): tau
                                               for tau in np.arange(self.start, self.stop, self.step)
                              }

                    # Perform here parallel results check
                    for result in concurrent.futures.as_completed(results):
                        exp_obj = result.exception()
                        if exp_obj is not None:
                            self.log.error(f"Result: |{result.exception()}|")
                            exps.append(exp_obj)
            else:
                # Performance penalty - used only for debugging
                for tau in np.arange(self.start,
                                     self.stop,
                                     self.step):
                    self.roc_tau_point(out_file,
                                       attack_name,
                                       ground_truth,
                                       data_list,
                                       tau)

        if self.cfg["model"]["animation"]:
            Plotter.animate_image(f"{out_file[:-4]}_*_cdb.png",
                                 f"{out_file[:-4]}_cdb.gif")

        # ------------------------------------------------------------------
        # CP based - point selection
        # ------------------------------------------------------------------
        if self.roc_data.empty:
            # self.log.warn(f"CD data not available for attack: {attack_name}")
            result = copy.copy(self.roc_data_default)
            result.update({
                "tau": 0,
                "model": f"{out_file[:-4]}.{self.cfg['model']['ad'][names['algo_name']]['ext']}",
                "file": f"{out_file[:-4]}_roc.csv",
            })
            return result

        self.roc_data, is_monotonic = RocUtil.normalize(self.roc_data, self.enable_line, self.cfg, self.log)

        best_cfg = self.cp_selector.cd_select(self.roc_data)

        self.best_cfg.update(best_cfg)

        # -------------------------------------------------------------------
        params = {"tau": int(self.best_cfg["tau"]),
                  "auc": round(self.best_cfg["auc"], 4)}

        self.log.info(
            f"Attack {attack_name.upper()} => "
            f"Best parameters:{json.dumps(params, indent=4)}"
        )

        # Save AUC for further optimization investigation
        auc_file = f"{out_file[:-4]}_tau_c.csv"

        # Sort by TAU value
        auc_data = (self.roc_data.loc[:, ["tau",
                                          "auc",
                                          "acc",
                                          "f1",
                                          "eer",
                                          "fpr",
                                          "tpr",
                                          "d2_min",
                                          "d2_avg",
                                          "d2_max"]]
                    .sort_values(["tau"]))
        auc_data.to_csv(auc_file, index=False)

        # Normalize TPR/FPR
        self.roc_data, is_monotonic = RocUtil.normalize(self.roc_data, self.enable_line, self.cfg, self.log)
        result_metric = self.m_met.run(self.roc_data,
                                       self.best_cfg,
                                       False,
                                       enable_line,
                                       {"data_list": data_list},
                                       False)

        self.roc_data.sort_values(by=["fpr", "tpr"], inplace=True)
        self.roc_data.loc[:, "auc"] = result_metric["auc"]

        # Save data for ROC curves and graphs
        self.roc_data.to_csv(f"{out_file[:-4]}_roc.csv")
        result_metric.update({
            "fpr": best_cfg["fpr"],
            "tpr": best_cfg["tpr"]
        })
        return result_metric

    def run(self,
            best_cfg: dict,
            algo_name: str,
            attack_name: str,
            prob: str,
            enable_line: bool) -> dict:
        """
        Method generates last step of the simulation
        Args:
            best_cfg: The best AD model configuration
            algo_name: algorithm name,
            attack_name: attack name,
            prob: probability string
            enable_line: flag to determine how to process ROC curve
        Returns:
            <dict>      - dictionary containing metrics for
                          cycle detector anomaly detection
        """
        # Clear state before running the CD AD
        self.best_cfg = copy.deepcopy(best_cfg)

        result_metric = {}
        result_metric.update(best_cfg)

        file_name = best_cfg["file"].replace("\\", "/").replace("//", "/")
        names = IterHelper.extract_names(file_name)

        # Load previous ROC curve
        out_file = (
            (
                f"{self.path}{self.cfg['results']['path']}"
                f"{file_name[file_name.rindex('/') + 1:]}"
            )
            .replace("\\", "/")
            .replace("//", "/")
        )

        roc_data_file = f"{out_file[:out_file.rfind('/') + 1]}a_data_{attack_name}_{prob}_{algo_name}_roc.csv"
        roc_data = pd.read_csv(roc_data_file)

        data = pd.read_csv(file_name, index_col=0)

        for column in [self.cfg["data_labels"]["cd"]["ground_truth"],
                       self.cfg["data_labels"]["cd"]["prediction"],
                       "cycle_len"]:
            data[column] = 0

        cd_ts = self.roc_cd_default.copy(deep=True)

        # tau_limit = 0
        # TODO: Temporary debugging to find out the incorrect data input
        # if "label" in data.columns:
        # tau_limit = self.get_tau_limit(data, best_cfg["auc"])
        # else:
        #    self.log.warn("Reporting data problem. Missing 'label' column")
        #    self.log.error(f"BData File   : {file_name}")
        #    self.log.error(f"BData Columns: {data.columns}")

        tau_limit = self.get_tau_limit(data, best_cfg["auc"])

        # 1. Split dataset into cycle sets before computation
        data_list = MetricDelta.load_cycles(data, self.cycle_len)

        ground_truth = self.compute_gt(data_list, tau_limit)

        try:
            if (os.path.isfile(file_name) and int(names["model_no"]) > 0 and
               int(names["parameter_no"]) > 0 and best_cfg["auc"] > 0.5):

                # TODO: ROC data - there is no need to generate roc_data - instead
                #  this should
                # 2. Compute d2, roc_cd data based on Tau threshold in each production cycle
                result_metric = self.roc_tau_loop(attack_name,
                                                  data_list,
                                                  out_file,
                                                  enable_line,
                                                  tau_limit,
                                                  ground_truth)
                # Set a couple of criteria to make sure that at least one good record is selected properly
                cd_ts = self.roc_data[self.roc_data["roc_cd"].apply(lambda x: isinstance(x, str))]
                cd_ts = cd_ts[cd_ts["eer"] == cd_ts["eer"].min()]
                cd_ts = cd_ts["roc_cd"]

                if not cd_ts.empty and isinstance(cd_ts, pd.Series):
                    cd_ts = pd.read_json(StringIO(cd_ts[cd_ts.index[0]]))
                else:
                    cd_ts = self.roc_cd_default.copy(deep=True)

            else:
                result_metric.update({"d2_min": 0, "d2_avg": 0, "d2_max": 0,
                                      "tpr0": 0, "tprn": 0})

            if result_metric["auc"] <= 0.5:
                self.roc_data = self.roc_data_default.copy(deep=True)

                roc_file_name = (f"{self.path}detection/a_data_"
                                 f"{attack_name}_{prob}_{algo_name}_m000_p000_roc.csv")
                if not os.path.isfile(roc_file_name):
                    self.roc_data.to_csv(roc_file_name)

                out_file = (f"{self.path}{self.cfg['results']['path']}"
                            f"a_data_{attack_name}_{prob}_{algo_name}_m000_p000.csv")
                # Save file to make sure that the file exists and contains correct data
                if not os.path.isfile(out_file):
                    data.to_csv(out_file)

                result_metric.update({"auc": 0.5,
                                      "acc": 0.5,
                                      "eer": 0.5,
                                      "f1": 0.5,
                                      "fpr": 0.5,
                                      "tpr": 0.5,
                                      "prec": 0.5,
                                      "recall": 0.5})

            self.log.info(
                f"Results => AUC={result_metric['auc']:.2f}, "
                f"ACC={result_metric['acc']:.2f}, "
                f"EER={result_metric['eer']:.2f}, "
                f"FPR/TPR={result_metric['fpr']:.2f}/{result_metric['tpr']:.2f}"
            )

            if roc_data is None and self.roc_cd is None and result_metric is None:
                self.log.warn(f"CD data not available: |{attack_name}|")
                return {}

            if self.roc_data.shape[0] > 2:
                self.roc_data.drop_duplicates(subset=["tau", "auc"], inplace=True)
            optimal_point = [[best_cfg["fpr"], best_cfg["tpr"]],
                             [result_metric["fpr"], result_metric["tpr"]]]

            # Generate all the graphs for cycle detector experiment
            Plotter.generate_graphs({"process": self.cfg["process"]} | self.best_cfg["cfg"],
                                    data,
                                    cd_ts,
                                    roc_data,
                                    self.roc_data.loc[:, ["tau", "auc", "acc", "f1", "eer",
                                                          "fpr", "tpr", "auc",
                                                          "d1_min", "d1_avg", "d1_max",
                                                          "d2_min", "d2_avg", "d2_max"]].sort_values(["tau"]),
                                    optimal_point,
                                    result_metric,
                                    out_file,
                                    self.cfg["process"]["limit"] / 100,
                                    int(self.cfg["cycle_detector"]["tau"][1]),
                                    self.log
            )

            perf_record = {
                "algo": algo_name,
                "attack": attack_name,
                "type": "cd",
                "knowledge": str(best_cfg["knowledge"]),
                "auc": result_metric["auc"],
                "acc": result_metric["acc"],
                "f1": result_metric["f1"],
                "eer": result_metric["eer"],
                "prec": result_metric["prec"],
                "recall": result_metric["recall"],
                "cfg": str(result_metric["cfg"]),
                "model": result_metric["model"],
                "file": result_metric["file"],
                "d1_min": result_metric["d1_min"],
                "d1_avg": result_metric["d1_avg"],
                "d1_max": result_metric["d1_max"],
                "d2_min": result_metric["d2_min"],
                "d2_avg": result_metric["d2_avg"],
                "d2_max": result_metric["d2_max"],
                "tpr0": result_metric["tpr0"],
                "tprn": result_metric["tprn"],
            }

            perf_data = pd.DataFrame([perf_record])
            perf_data.set_index(["algo", "attack"], inplace=True)

            file_path = file_name[: file_name.rindex("/")]
            f_name = file_name[file_name.rindex("/") + 1 :]
            f_tokens = [x for x in f_name.split("_") if "mir" != x]
            perf_file = "_".join(list(f_tokens[0:2] + f_tokens[3:4]))
            perf_file = f"{file_path}/{perf_file}_cd_perf.csv"

            # Update performance data file if it exists
            if os.path.isfile(perf_file):
                perf_d = pd.read_csv(perf_file, index_col=["algo", "attack"])
                perf_data = pd.concat([perf_d, perf_data])

            # Save file with new updates
            perf_data.to_csv(perf_file, index=True)

            return perf_record

        except Exception as ex:
            self.log.error(f"Problem with algorithm {algo_name}:")
            self.log.exception(ex)
