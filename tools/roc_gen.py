""" Generic ADEF ROC generation module."""

from __future__ import annotations

import gc
import warnings
import traceback

import copy
import pickle
from datetime import datetime
import concurrent.futures
import os.path
import json
import threading
import itertools
import joblib
import torch
import torch.nn as nn

import multiprocessing
from multiprocessing.pool import ThreadPool

import pandas as pd

# Local imports
from tools.decorators import WarningTrace
from tools.plotter_helper import Plotter
from tools.log import Log
from mev.data.eng import DataEng
from mev.metrics.met_point import AnomalyDetectionMetricsPerPoint
from mev.metrics.met_model import AnomalyDetectionMetricsPerModel
from tools.roc_util import RocUtil
from tools.cp_selector import CPSelector
from tools.iter_helper import IterHelper


class RocGenerator:
    """
    Receiver Operating Characteristics generator
    """

    def __init__(self, cfg: dict, algo_name: str, logger: Log) -> None:
        """
        CTOR
        Args:
            cfg: configuration object containing information about
            algo_name: algorithm name
            logger: logger object
        Returns:
            <None>
        """
        self.mutex = threading.Lock()
        self.log = logger
        self.cfg = cfg

        # CP solver that will select best model/points in the selection process
        self.cp_selector = CPSelector(self.cfg, self.log)

        self.limit = self.cfg["process"]["limit"]
        self.timeout = self.cfg["model"]["ad"][algo_name]["timeout"]
        self.cycle_len = (
            self.cfg["process"]["cycle_len"] * self.cfg["process"]["samples"]
        )

        self.curr_point = 0
        self.tms = datetime.now()
        self.cnt_reg = []

        self.p_test_cnt = 1
        self.p_results = pd.DataFrame()

        self.best_result = None
        self.out_files = None
        self.results = None

        self.total_points = 0
        self.best_auc = 0.0
        self.best_acc = 0.0
        self.test_cnt = 1

        self.out_files = []
        self.path = f"{self.cfg['global']['path']}experiments/{self.cfg['experiment']['path']}"
        self.p_met = AnomalyDetectionMetricsPerPoint("step4", self.cfg, self.log)
        self.m_met = AnomalyDetectionMetricsPerModel("step4", self.cfg, self.log)
        self.cache = {
            "model": {},
            "result": {}
        }
        self.default_result = pd.DataFrame({
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

        self.default_best_result = {
            "auc": 0.5,
            "acc": 0.5,
            "eer": 0.5,
            "f1": 0.5,
            "fbeta": 0,
            "support": 0,
            "model": "",
            "file": "",
            "cfg": cfg,
            "recall": 1,
            "prec": 0,
            "d1_min": 0,
            "d1_avg": 0,
            "d1_max": 0
        }

    @staticmethod
    def param_product(dict_params: dict) -> list:
        """
        Method produces combinations without repetition for
        all parameters provided in dict_params parameter
        Args:
        dict_params: list(dict_product(dict(number=[1,2], character='ab')))
                    [{'character': 'a', 'number': 1},
                     {'character': 'a', 'number': 2},
                     {'character': 'b', 'number': 1},
                     {'character': 'b', 'number': 2}]
        Returns:
            <list> - list of dictionaries containing combinations
                     of the parameters used in ROC generation
        """
        # Separate
        if "search_param" not in dict_params.keys():
            return [
                    dict(zip(dict_params, x))
                    for x in itertools.product(*dict_params.values())
            ]

        dict_params_clear = dict_params.copy()
        keys = dict_params_clear["search_param"]
        values = dict_params_clear[keys]

        del dict_params_clear[keys]
        del dict_params_clear["search_param"]

        new_params = []
        for param_values in itertools.product(*dict_params_clear.values()):
            clear_keys = list(dict_params_clear.keys())
            new_param_set = dict(zip(clear_keys, list(param_values)))

            roc_set = []
            for val in values:
                row = dict(new_param_set)
                row[keys] = val
                roc_set.append(row)

            new_params.append(roc_set)

        return new_params

    def progress_report(self) -> None:
        """
        Method provides progress report on parallel computation
        Args:
            <None>
        Returns:
            <None>
        """
        with self.mutex:
            percent_compl = int((self.curr_point / self.total_points) * 100)
            if percent_compl % 10 == 0 and percent_compl not in self.cnt_reg:
                exec_time = (datetime.now() - self.tms).total_seconds()
                self.tms = datetime.now()
                self.log.debug(
                    f"ROC point [{self.curr_point}/{self.total_points}] => "
                    f"{percent_compl:.2f}% T[{exec_time:.2f}s]"
                )
                # f" ROC[{self.best_result['auc']:.2f}]",

                self.cnt_reg.append(percent_compl)

    def loop_state_reset(self) -> None:
        """
        Method resets loop state for next model computation
        Args:
            <None>
        Returns:
            <None>
        """
        self.out_files = []
        self.test_cnt = 1
        self.results = pd.DataFrame()
        self.p_results = pd.DataFrame()
        self.best_auc = 0.0
        self.best_acc = 0.0
        self.curr_point = 0
        self.best_result = copy.copy(self.default_best_result)
        self.cnt_reg = []
        self.cache = {
            "model": {},
            "result": {}
        }
        gc.collect()

    @staticmethod
    def load_model(cfg: dict, file_name: str, logger: Log) -> (dict, object):
        """
        Method loads AD model from file to memory.
        Depending on the model type the model is loaded
        Args:
            cfg: AD model configuration.
            file_name: name of the file where model will be saved.
            logger: logger object.
        Returns:
            <dict>: AD model configuration
            <object>:
        """
        stream = None
        # logger.debug(f"Loading model: {file_name}")

        # Joblib based model loading
        if "pkl" in cfg["ext"]:
            stream = joblib.load(file_name)
            model = stream["model"]

        # PyTorch based model loading
        elif "pth" in cfg["ext"]:
            stream = torch.load(file_name)
            model = {"model": stream["model"],
                     "criterion": stream["criterion"],
                     "optimizer": stream["optimizer"]
            }

            # model["model"].eval()

        if stream is not None:
            configuration = stream["cfg"]
            configuration["label"] = cfg["label"]
            return configuration, model

        return {}, None

    @staticmethod
    def save_model(model: dict, cfg: dict, file_name: str, logger: Log) -> None:
        """
        Method loads AD model from file to memory.
        Depending on the model type the model is loaded
        Args:
            model: AD model object.
            cfg: AD model configuration.
            file_name: name of the file where model will be saved.
            logger: logger object.
        Returns:
            <None>
        """
        # logger.debug(f"Saving model: {file_name}")

        # Resolve model type and serialize accordingly - Torch has specific method
        # while the rest of the models are saved using joblib
        if file_name.endswith("pth"):
            # TODO: Fix model validation before saving it
            try:
                # TODO: https://discuss.pytorch.org/t/cant-save-load-model-with-state-dict/120373
                # TODO: How to serialize collections.OrderedDict
                if all(key in model for key in ["model", "criterion", "optimizer"]):
                    torch.save({
                        "model": model["model"].state_dict() if hasattr(model["model"], "state_dict") else model["model"],
                        "criterion": model["criterion"],
                        "optimizer": model["optimizer"],
                        "cfg": cfg
                    }, file_name)
                else:
                    logger.warn(f"PYTORCH ERROR: Unable to save non-compliant model >> {file_name}")
            except AttributeError as ae:
                logger.exception(ae)
                logger.warn(f"Model file: |{file_name}|")
            except Exception as ex:
                logger.exception(ex)

        elif file_name.endswith("pkl"):
            joblib.dump({"model": model, "cfg": cfg}, file_name)

    # --------------------------------------------------------------------------------
    # unified function for ROC generation irrespective of anomaly detection function
    # --------------------------------------------------------------------------------
    def point(
        self,
        data: pd.DataFrame,
        ad_model: object,
        cfg: dict,
        out_file: str,
        s_cfg: dict,
        train: bool,
    ) -> None:
        """
        Method computes single ROC point using
        Args:
            data: Data Frame containing input data used in evaluation
            ad_model: anomaly detection model
            cfg: current function configuration
            out_file: Output file name
            s_cfg: static configuration for AD model
            train: True - model creation, False - model detection
        Returns:
            <None>
        """
        try:
            with self.mutex:
                self.curr_point += 1
                out_file_test = out_file.replace("\\/", "/").replace("\\", "/")

                # Create model
                if train:
                    config = cfg
                    config["label"] = s_cfg["label"]
                    model_file = f"{out_file[:-4]}.{s_cfg['ext']}"
                    model = ad_model.create(cfg)
                # Load model
                else:
                    # model_file = out_file.replace("graph/", "model/")
                    model_file = out_file.replace("detection/", "model/")
                    model_file = f"{model_file[:-4]}.{s_cfg['ext']}"

                    cfg, model = RocGenerator.load_model(s_cfg, model_file, self.log)
                    config = cfg
                    config["label"] = s_cfg["label"]
                    out_file_test = out_file_test.replace("/model/", "/detection/")

                model_file = model_file.replace("\\/", "/").replace("\\", "/")

                self.out_files.append(out_file_test)

                result = pd.DataFrame()

                # Adjust AD model function depending on the model state (training/detection)
                ad_model_func = ad_model.adef_train if train else ad_model.adef_detect

                # Chop data set into number of cycles so the data set will be always the same length
                if s_cfg["continuous"]:
                    # Continuous anomaly detection
                    result = ad_model_func(data, cfg, model)
                else:
                    cycles = DataEng.split_data(data, self.cycle_len, False)
                    # Cycle based anomaly detection
                    for cycle in cycles:
                        cycle_result = ad_model_func(cycle, cfg, model)
                        result = pd.concat([result, cycle_result])

                # Save model
                if train:
                    RocGenerator.save_model(model, cfg, model_file, self.log)

                # If result is empty do not further process data (continue with other points)
                if result.empty:
                    self.log.warn(f"ROC PT [{self.curr_point}] => Timeout: |{config}|")
                    return

                result.fillna(0.0, inplace=True)

                self.cache["model"][out_file_test] = {"cfg": cfg}
                if "pth" == s_cfg["ext"]:
                    self.cache["model"][out_file_test].update({"model":
                        {

                            "model": model["model"].state_dict() if hasattr(model["model"], "state_dict")
                                     else model["model"],
                            "criterion": model["criterion"],
                            "optimizer": model["optimizer"],
                            "cfg": cfg
                        }
                    })
                else:
                    self.cache["model"][out_file_test].update({"model": model})

                self.cache["result"][out_file_test] = result.copy(deep=True)

                result_metric_row = self.p_met.run(out_file_test, result, config)

                self.p_results = pd.concat(
                    [
                        self.p_results,
                        pd.DataFrame(
                            [result_metric_row], columns=list(result_metric_row.keys())
                        ),
                    ]
                )

            self.progress_report()
        except Exception as ex:
            self.log.exception(ex)

    def single_loop(
        self,
        a_data: pd.DataFrame,
        in_params: dict,
        ad_func: object,
        cfg: object,
        roc_out_file: str,
        training: bool,
        enable_line: bool
    ) -> (pd.DataFrame, dict, bool):
        """
        Single ROC loop
        Args:
            a_data:
            in_params:
            ad_func:
            cfg:
            roc_out_file:
            training:
            enable_line: flag to determine how plots will be drawn (True - plot, False - scatter)
        Returns:
            <pd.DataFrame> -
            <dict> - Flag containing
            <bool> - Flag showcasing measure of the ROC monotonicity
        """
        self.p_test_cnt = 1
        self.p_results = pd.DataFrame()

        exps = []

        # ------------------------------------------------------------------------------------
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=int(os.cpu_count() / 2)
        ) as executor:
            point_results = {
                executor.submit(self.point,
                                a_data,
                                ad_func,
                                param[0],
                                param[1],
                                cfg,
                                training):
                    param for param in in_params
            }

            try:
                # Perform here parallel results check
                for result in concurrent.futures.as_completed(point_results, timeout=3600):
                    exp_obj = result.exception()
                    if exp_obj is not None:
                        self.log.error(f"ROC GEN Loop Result: |{exp_obj}|")
                        exps.append(exp_obj)
            except TimeoutError:
                self.log.warn("Loop Timeout")
                return None, None, False

        '''
        # Performance penalty: ~30% slower execution time - only for DEBUG session
        for param in in_params:
            self.point(a_data,
                       ad_func,
                       param[0],
                       param[1],
                       cfg,
                       training)
        '''

        if len(exps) > 0:
            self.log.error(
                f"Exceptions {len(exps)} "
                f"occurred during parallel execution for ROC curve"
            )
            for ex in exps:
                self.log.error(f"EX=> |{ex}|")

        # Enable this part only when parallel results are OK
        if self.p_results is None or self.p_results.empty:
            self.log.error(
                "Results are not available\n Param set: "
                f"{json.dumps(in_params[0], indent=4)},"
                f"{json.dumps(in_params[-1], indent=4)}"
            )
            return None, None, False

        self.p_results.reset_index(drop=True, inplace=True)

        # ROC normalization process
        self.p_results, monotonic_check = RocUtil.normalize(self.p_results, enable_line, self.cfg, self.log)

        best_point = self.cp_selector.select(self.p_results)

        if self.cfg["cp_selection"]["debug"]:
            cbp_file = f"{best_point['file'][:-4]}_cbp.csv"
            self.log.debug(f"Saving CBP to: {cbp_file}")
            self.p_results.to_csv(cbp_file)

        # Save best model data
        if not best_point["file"] in self.cache["result"].keys():
            self.cache["result"][best_point["file"]] = self.default_result.copy(deep=True)
            self.cache["model"][best_point["file"]] = cfg

            names = IterHelper.extract_names(best_point["file"])
            algo_name = names["algo_name"]
            ext = self.cfg["model"]["ad"][algo_name]["ext"]
            if "pth" == ext:
                self.cache["model"][best_point["file"]] = {"model": {
                                                               "model": {},
                                                               "criterion": {},
                                                               "optimizer": {}
                                                           },
                                                           "cfg": cfg}
            elif "pkl" == ext:
                self.cache["model"][best_point["file"]] = {"model": {},
                                                           "cfg": cfg}

        self.cache["result"][best_point["file"]].to_csv(best_point["file"])

        local_cfg = IterHelper.extract_names(best_point["file"])

        self.save_model(self.cache["model"][best_point["file"]]["model"],
                        self.cache["model"][best_point["file"]]["cfg"],
                        f"{best_point['file'][:-4]}.{self.cfg['model']['ad'][local_cfg['algo_name']]['ext']}",
                        self.log)

        if monotonic_check:
            # Compute model metrics and images
            best_result = self.m_met.run(
                self.p_results,
                best_point,
                training,
                enable_line
            )

        # If model is not monotonic show only best point optimized version
        else:
            self.p_results = pd.DataFrame({
                "tau": [0, 0, 0],
                "tp": [0, best_point["tp"], 0],
                "tn": [0, best_point["tn"], 0],
                "fp": [0, best_point["fp"], 0],
                "fn": [0, best_point["fn"], 0],
                "fpr": [0, best_point["fpr"], 1],
                "tpr": [0, best_point["tpr"], 1],
                "recall": [1, best_point["recall"], 0],
                "prec": [0, best_point["prec"], 1],
                "auc": [best_point["auc"], best_point["auc"], best_point["auc"]],
                "cfg": [cfg, cfg, cfg],
                "model": [best_point["model"], best_point["model"], best_point["model"]],
                "file": [best_point["file"], best_point["file"], best_point["file"]],
                "acc": [best_point["acc"], best_point["acc"], best_point["acc"],],
                "eer": [best_point["eer"], best_point["eer"], best_point["eer"]],
                "f1": [best_point["f1"], best_point["f1"], best_point["f1"]],
                "fbeta": [best_point["fbeta"], best_point["fbeta"], best_point["fbeta"]],
                "support": [best_point["support"], best_point["support"], best_point["support"]],
                "d1_min": [best_point["d1_min"], best_point["d1_min"], best_point["d1_min"]],
                "d1_avg": [best_point["d1_avg"], best_point["d1_avg"], best_point["d1_avg"]],
                "d1_max": [best_point["d1_max"], best_point["d1_max"], best_point["d1_max"]]
            })

            best_result = {k: best_point.get(k, 0) for k in self.default_best_result.keys()}

        if best_point["auc"] < 0.5:
            self.p_results = self.default_result.copy(deep=True)
            self.p_results.loc[:, "file"] = best_result["file"]
            self.p_results.loc[:, "model"] = best_result["model"]

        self.p_results.to_csv(f"{best_result['file'][:-4]}_roc.csv")

        # Save results data
        self.p_results.loc[0, "auc"] = best_result["auc"]

        # Clear cache
        self.cache = {
            "model": {},
            "result": {}
        }
        best_result.update({"fpr": best_point["fpr"],
                            "tpr": best_point["tpr"]})

        return self.p_results, best_result, monotonic_check

    @staticmethod
    def get_input_parameters(meta_params: dict,
                             cfg: dict,
                             training: bool,
                             out_file_template: str,
                             logger: Log,
                             model_file: str = None) -> (list, int, list):
        """
        Method prepares input parameters for training and evaluation ADEF steps.
        Args:
            meta_params: meta level parameter search space for given AD model
            cfg: algorithm configuration
            training: flag to determine whether ROC is being generated in training mode (True)
                      or evaluation mode (False)
            out_file_template: file name that will be used to save results and graphs
            logger: logger object
            model_file:
        Returns:
            <list>: input parameter list
            <list>: iteration parameter list
            <list>: single parameter list set
        """
        # Preparation step for parallel looping over multiple parameter sets
        # Generate file names to enable correct synchronization of model generation for
        # each ROC point and avoid model file writing/reading collisions in MT environment
        cnt = 1
        model_cnt = 1
        in_params = []
        param_it = []
        single_roc_list = []
        if model_file and os.path.isfile(model_file):
            param_it, model = RocGenerator.load_model(cfg, model_file, logger)
            del param_it["label"]
            param_it = {x: [y] for x,y in param_it.items()}
            param_it[meta_params["search_param"]] = meta_params[meta_params["search_param"]]
            param_it["search_param"] = meta_params["search_param"]
            param_it = RocGenerator.param_product(param_it)

        # TODO: Adjust parameter such single_roc_loop will
        #  get appropriate number of parameters
        file_template = copy.copy(out_file_template)

        if training:
            param_it = RocGenerator.param_product(meta_params)
            search_roc = "search_param" in meta_params.keys()
            if search_roc:
                for param_list in param_it:
                    cnt_len = len(str(len(param_list)))
                    single_roc_list = []
                    for param in param_list:
                        m_file = f"{file_template}m{model_cnt:0>3}_p{cnt:0>{cnt_len}}.csv"
                        single_roc_list.append([param, m_file])
                        cnt += 1
                    cnt = 1
                    in_params.append(single_roc_list)
                    model_cnt += 1
            else:
                for param_list in param_it:
                    cnt_len = len(str(len(param_list)))
                    m_file = f"{file_template}m{model_cnt:0>3}_p{cnt:0>{cnt_len}}.csv"

                    cnt = 1
                    in_params.append([[param_list, m_file]])
                    model_cnt += 1
        else:
            setup = IterHelper.extract_names(model_file)
            cnt_len = len(setup["parameter_no"])
            for param in param_it[0]:
                m_file = f"{file_template[:-3]}m{setup['model_no']}_p{cnt:0>{cnt_len}}.csv"
                in_params.append([param, m_file])
                single_roc_list.append([param, m_file])
                cnt += 1

            in_params = [in_params]

        # In the non training phase (STEP4/STEP5) prepare only single model(best)
        # with multiple points
        if param_it is None:
            raise ValueError("Loop iterator not initialized!!!")

        total_points = len(param_it) * len(single_roc_list) if training else len(single_roc_list)

        return in_params, total_points, single_roc_list

    # @WarningTrace(DeprecationWarning)
    def roc_search_loop(
        self,
        a_data: pd.DataFrame,
        meta_params: dict,
        ad_func: object,
        cfg: dict,
        out_file_template: str,
        roc_image_label: str,
        training: bool,
        enable_line: bool,
        model_file: str,
        best_cfg: dict = None
    ) -> (list, pd.DataFrame, dict, list, list):
        """
        Method generates ROC data for given anomaly detection algorithm.
        This method is implementing search algorithm over multiple AD parameters with
        search param set as local interator. Across different models the best model will be
        selected in non training mode. The selection is done based on the given criteria.
        Args:
            a_data: pd.DataFrame used in detection
            meta_params: AD anomaly detection parameter search space configuration
            ad_func: pointer to anomaly detection function used in model training or evaluation step
            cfg: AD model configuration
            out_file_template: Initial output template file name
            roc_image_label: ROC image label applied when generating graph
            training: True - create model, False - perform detection
            enable_line: True - draw all plots as a line, otherwise scatter plot will be used (False)
            model_file: model file from where best model configuration is extracted
            best_cfg: best configuration for given model/attack configuration observed during training
        Returns:
            <list>: List of output files
            <pd.DataFrame>: ROC result saved for given detection
            <dict>: optimal parameters
                <model_file>: Best AD model found during training process
                <cfg>: Best parameters of the model
                <f1_score>: Best F1 score computed for the detection
            <list>: List of iteration parameters that are generating monotonic results
            <list>: List containing optimal box parameters [[x1, y1], [x2, y2]] for ROC detection
        """
        # Generate multiple files with different detection scenarios
        # to generate ROC curve.
        self.loop_state_reset()

        results = pd.DataFrame()
        it_params = []
        best_result = {"auc": 0.5, "acc": 0.5, "fpr": 0.5, "tpr": 0.5}

        enable_run = True
        if best_cfg is not None:
            algo_tokens = IterHelper.extract_names(best_cfg["file"])
            algo_name = algo_tokens["algo_name"]
            attack_name = algo_tokens["attack_name"]
            model_no = int(algo_tokens["model_no"])
            param_no = int(algo_tokens["parameter_no"])
            if model_no == param_no == 0:
                enable_run = False
                best_result.update(best_cfg)
                # Save default file for further processing
                def_results = a_data.copy(deep=True)
                def_results["result"] = 0
                def_results["cd_result"] = 0
                def_results.to_csv(best_cfg["file"])

        if best_cfg is None or enable_run:
            (in_params,
             self.total_points,
             single_roc_list) = RocGenerator.get_input_parameters(meta_params,
                                                                  cfg,
                                                                  training,
                                                                  out_file_template,
                                                                  self.log,
                                                                  model_file)

            p_cnt = 1

            if a_data.isna().any().any():
                a_data.fillna(0.0, inplace=True)

            self.best_result.update(eval(self.default_result.loc[0, :].to_json()))

            for in_param in in_params:
                # Check ROC values and update final result
                results, best_result, is_monotonic = self.single_loop(
                    a_data,
                    in_param,
                    ad_func,
                    cfg,
                    f"{out_file_template}{'train_' if training else '_'}{p_cnt}",
                    training,
                    enable_line)
                p_cnt += 1

                if training and is_monotonic or not enable_line:
                    it_params.append(in_param)

                # Compare results and save only the best one
                if (best_result["auc"] > self.best_result["auc"] and
                   best_result["acc"] > self.best_result["acc"]):
                    self.best_result = best_result

        roc_out_file = f"{out_file_template}{'train_roc' if training else ''}"

        opt_box = [[0, 0], [0.3, 0.3]]

        if best_result["auc"] > 0.5 and not results.empty:
            results.loc[0, "auc"] = best_result["auc"]
            results.loc[0, "cfg"] = str(best_result["cfg"])
            results.loc[0, "file"] = best_result["file"]
            results.loc[0, "model"] = best_result["model"]
            results.loc[0, "d1_min"] = best_result["d1_min"]
            results.loc[0, "d1_avg"] = best_result["d1_avg"]
            results.loc[0, "d1_max"] = best_result["d1_max"]

            # Compute industrial optimal use boxed parameters based on the
            # configuration criteria.
            opt_box = RocUtil.opt_detection_box(results, self.limit / 100)
        else:
            results = self.default_result.copy(deep=True)
            results.loc[0, "file"] = best_result["file"]
            results.loc[0, "model"] = best_result["model"]
            results.loc[0, "cfg"] = str(cfg)
            self.best_result = eval(self.default_result.copy(deep=True).loc[0, :].to_json())

        self.best_result["file"] = best_result["file"]
        self.best_result["model"] = best_result["model"]

        # Save results data
        if not training:
            # Group by index and get the first row of each group
            algo_tokens = IterHelper.extract_names(model_file)
            algo_name = algo_tokens["algo_name"]
            attack_name = algo_tokens["attack_name"]
            prob_set_str = algo_tokens["prob"]

            # Extract the first row value for a specific column, e.g., column 'A'
            roc_out_file = f"{self.path}graph/a_data_{attack_name}_{prob_set_str}_{algo_name}_roc"

        self.log.debug(f"Saving ROC to: {roc_out_file}.csv")
        results.to_csv(f"{roc_out_file}.csv")

        Plotter.roc_2d(results.loc[:, [self.cfg["data_labels"]["fpr"],
                                       self.cfg["data_labels"]["tpr"]]],
                       self.best_result,
                       roc_image_label,
                       f"{roc_out_file}.png",
                       self.limit / 100,
                       enable_line)

        self.log.info(
            f"Results => AUC={self.best_result['auc']:.2f}, "
            f"ACC={self.best_result['acc']:.2f}, "
            f"EER={self.best_result['eer']:.2f}, "
            f"FPR/TPR={self.best_result['fpr']:.2f}/{self.best_result['tpr']:.2f}\n"
            f"File|{self.best_result['file']}|\n"
            f"Model|{self.best_result['model']}|"
        )

        return self.best_result, it_params, opt_box
