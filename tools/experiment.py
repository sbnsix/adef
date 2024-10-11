""" Experiment process definition module. """


from __future__ import annotations

import glob
import json
import os
import random
import shutil
import sys
import uuid
import argparse

from datetime import datetime
import pandas as pd

PATHS = ["./", "../icsr/threats", "../mev/adm", "../mev/metrics", "../tools"]
for path in PATHS:
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), path))
    if path not in sys.path:
        sys.path.insert(1, path)

# ---------------------------------------------
#                Local imports
# ---------------------------------------------
from tools.log import Log
from tools.checkpoint import checkpoint
from tools.checkpoint import timing
from tools.iter_helper import IterHelper
import tools.yaml_converter as yc
from tools.os_utility import Utility
from tools.plotter_helper import Plotter
from roc_gen import RocGenerator

from mev.data.filter_factory import DataFilterFactory
from mev.data.eng import DataEng

# Process anomaly detectors
from mev.adm.adm_factory import DetFactory

from mev.cd.cycle_detector import CycleDetector
from tools.adef_post import PostProcessor
from cfg.ad_config import ad_config
from step_state import StepState

from mev.metrics.met_model import AnomalyDetectionMetricsPerModel
from mev.metrics.met_diff import AnomalyDetectionMetricsDiff


class Experiment:
    """
    Class describing ADEF simulation experiment
    """

    def __init__(self,
                 cfg: dict,
                 logger: Log) -> None:
        """
        CTOR
        Args:
            cfg: experiment configuration object
            logger: logger object
        Returns:
            <None>
        """
        random.seed(str(uuid.uuid4()))
        self.log = logger

        # Timing information for each experiment step
        self.timing = {}

        if cfg is None:
            self.log.error(f"Configuration is empty => |{cfg}|")
            return

        # Load simulator configuration
        self.cfg = cfg

        DetFactory.init(self.cfg, logger)

        # self.cache_dir = cfg["cache"]
        # self.mem = joblib.Memory(self.cache_dir)

        if "mode" not in cfg.keys():
            cfg["mode"] = False

        self.meta_mode = cfg["mode"]
        self.id = cfg["experiment"]["id"]
        exp_path = f"{cfg['global']['path']}experiments/{cfg['experiment']['path']}"
        if len(exp_path.split("..")) > 1:
            exp_path = exp_path[exp_path.rfind(".."):]
        # Experiment main configuration
        self.path = exp_path
        self.limit = self.cfg["global"]["detection"]["limit"]

        # Anomaly detectors initialization
        self.detectors = {}
        self.f_ad_training = {}
        self.f_ad_detection = {}

        # Anomaly detectors initialization
        self.log.debug("Anomaly detector initialization:")
        self.algo_max_count = 0
        for key, ad_cfg in self.cfg["model"]["ad"].items():
            if ad_cfg["enabled"]:
                detector_name = key.upper()
                self.log.debug(
                    f"\t\t{detector_name} enabled"
                )
                self.detectors[key] = DetFactory.create(key, self.log)

                self.algo_max_count += 1

        # Compute maximum number of model/attack/probability tests that experiment
        # is required to run.
        attack_count = len(
            {
                k: v
                for k, v in self.cfg["attack"]["types"].items()
                if v["enabled"]
            }.keys()
        )

        self.algo_max_count *= attack_count
        self.algo_max_count *= len(self.cfg["attack"]["densities"])

        # Data - simulated normal data set and mirrored data - differential signal
        # between original data set and attack data
        input_files = [
            file
            for file in glob.glob(f"{exp_path}/input/*.csv")
            if not file.endswith("_mir.csv")
        ]

        for input_file in input_files:
            self.data = pd.read_csv(input_file)

        for input_file in glob.glob(f"{exp_path}/input/*_mir.csv"):
            self.data_mir = pd.read_csv(input_file)

        # Attack data - normal trace combined with given attack scenario
        input_files = [
            file
            for file in glob.glob(f"{exp_path}/attack/*.csv")
            if not file.endswith("_mir.csv")
        ]

        for input_file in input_files:
            self.a_data = pd.read_csv(input_file)
        for input_file in glob.glob(f"{exp_path}/attack/*_mir.csv"):
            self.a_data_mir = pd.read_csv(input_file)

        self.m_path = self.path + self.cfg["model"]["path"]
        self.o_path = self.path + self.cfg["model"]["result"]
        self.a_path = self.path + self.cfg["attack"]["path"]

        # Use only algorithms that are enabled in the experiment
        self.algo = {}
        for algo, cfgv in self.cfg["model"]["ad"].items():
            if cfgv["enabled"]:
                self.algo[algo] = cfgv

        # Setup cycle length
        self.cycle = self.cfg["process"]["cycle_len"]
        self.cycle_len = (
            self.cfg["process"]["cycle_len"] * self.cfg["process"]["samples"]
        )

        self.params = None
        self.attack_files = {}

        self.test_limit = self.cfg["process"]["cycles"] * self.cycle_len

        # List of change parameters to iterate over
        self.meta_params = ad_config

        # Cycle detector object
        self.cd = CycleDetector(self.cfg, self.meta_params, self.log)

        self.attack_names = list(
            {
                x: y
                for (x, y) in self.cfg["attack"]["types"].items()
                if y["enabled"]
            }.keys()
        )

        self.algo_names = {
            x: y for (x, y) in self.cfg["model"]["ad"].items() if y["enabled"]
        }.keys()

        detector_keys = self.detectors.keys()

        self.algo = {
            key: m_cfg
            for (key, m_cfg) in self.cfg["model"]["ad"].items()
            if m_cfg is not None
            and m_cfg["enabled"]
            and m_cfg["training"]
            and key in detector_keys
        }

        # Experiment pipeline definition - function list to be
        # executed sequentially.
        self.steps_table = {
            # Attack adjustment
            "STITCHER": self.stitcher,
            # AD model training
            "STEP 3": self.step_3,
            # AD model detection
            "STEP 4": self.step_4,
            # Cycle detector computation
            "STEP 5": self.step_5,
            # Results visualization (graphs and LaTeX content generation)
            "STEP 6": self.step_6,
        }

        # Prepare storage for combination of the detector best models against all attacks
        # and metrics
        self.best_models = {}
        self.s3_metrics = {}
        self.s4_metrics = {}
        self.s5_metrics = {}

        for prob_set in self.cfg["attack"]["densities"]:
            prob_set_str = f"{int(prob_set * 100):02d}"
            self.best_models[prob_set_str] = {
                key: {attack_name: {} for attack_name in self.attack_names}
                for key in self.algo.keys()
            }
            self.s3_metrics[prob_set_str] = {
                key: {attack_name: {} for attack_name in self.attack_names}
                for key in self.algo.keys()
            }
            self.s4_metrics[prob_set_str] = {
                key: {attack_name: {} for attack_name in self.attack_names}
                for key in self.algo.keys()
            }
            self.s5_metrics[prob_set_str] = {
                key: {attack_name: {} for attack_name in self.attack_names}
                for key in self.algo.keys()
            }

        self.rocg = {}

        for key in self.algo.keys():
            self.rocg[key] = RocGenerator(self.cfg, key, self.log)

        self.metric_data = pd.DataFrame(columns=["prob", "algo", "attack", "type", "acc", "auc", "f1", "eer"])
        self.log.debug("Experiment initialization completed")

        # Attack files categorized based on data set balance, algorithm type and attack name
        for prob_set in self.cfg["attack"]["densities"]:
            prob_set_str = f"{int(prob_set * 100):02d}"
            self.attack_files[prob_set_str] = {}
            for algo_name, m_config in self.algo.items():
                self.attack_files[prob_set_str][algo_name] = {}
                for attack_name in self.attack_names:
                    self.attack_files[prob_set_str][algo_name][attack_name] = {}

                    # TODO: Later add ability to allow models perform with or without knowledge
                    if "thb" == algo_name or m_config["knowledge"]:
                        search_mask = (
                            f"{self.a_path}a_data_{attack_name}_{prob_set_str}_mir*csv"
                        )
                    else:
                        search_mask = (
                            f"{self.a_path}a_data_{attack_name}_{prob_set_str}*csv"
                        )
                    self.attack_files[prob_set_str][algo_name][attack_name] = [
                        file.replace("\\", "/") for file in glob.glob(search_mask)
                    ]

        self.metrics = AnomalyDetectionMetricsPerModel("step4", self.cfg, self.log)
        self.diff_metrics = AnomalyDetectionMetricsDiff("step4", self.cfg, self.log)

        # Post processor
        self.post = PostProcessor(
            self.cfg, self.attack_names, self.algo, self.attack_files, self.log
        )

        # Model files generated in the process that will be selected for
        # further investigation
        self.best_files = []
        self.init_state_size = StepState.get_size(self.best_models)

    @timing
    @checkpoint("step2p")
    def stitcher(self) -> None:
        """
        Method performs stitcher process for ADEF experiment.
        Args:
            <None>
        Returns:
            <None>
        """
        self.log.debug("STITCHER - Environment Model Attack combination for ADEF experiment")

        # TODO: Rewrite this routine to:
        #  1. Take environment influence input for model,
        #  2. Take model input,
        #  3. Apply attacks
        #  4. Store file and copy this to current experiment

        # Checked whether data mirror - with knowledge should be computed
        # For this purpose scan all enabled configurations and check if at least one
        # of them has knowledge flag enabled.
        know_data_flag = len(
            {
                x: y
                for (x, y) in self.cfg["model"]["ad"].items()
                if y["enabled"] and y["knowledge"]
            }.keys()
        )
        know_data_flag = True if know_data_flag > 0 else False

        # Depending on the knowledge included in the trace use specific data set
        # mirrored with original ADEF simulation - knowledge is enabled, otherwise the
        # original ICS trace.

        i_data = self.data
        a_data = self.a_data
        if know_data_flag:
            i_data_mir = self.data_mir
            a_data_mir = self.a_data_mir

        # 2. Depending on the configuration remove specific attacks and replace them with either
        #  other allowed attacks (generate specific one) or replace it with original data

        # 3. Save new results into attack folder for experiment to be using this file
        if know_data_flag:
            self.data_mir = i_data_mir
            self.a_data_mir = a_data_mir

        self.data = i_data
        self.a_data = a_data

    def input_filter(self, data: pd.DataFrame, config: dict) -> pd.DataFrame:
        """
        Input data filter function that applies filtering rules to given detection algorithm
        Args:
            data: input data to be processed
            config: configuration of the AD model
        Returns:
            <pd.DataFrame> - data frame containing filtered data
        """
        # TODO: Apply data filtering and manifold to generated data set
        # TODO: Apply per cycle filtering to avoid long term aftereffects of filters on the dataset
        if "filter" in config.keys() and "before" in config["filter"].keys():
            columns = data.columns.tolist()
            chunks = DataEng.split_data(data, self.cycle_len)
            data_ds = pd.DataFrame()
            for chunk in chunks:
                chunk1 = chunk.copy(deep=False)
                chunk1 = DataFilterFactory.filter(
                    chunk1.loc[:, columns], config["filter"]["before"]
                )
                data_ds = pd.concat([data_ds, chunk1])

            data = data_ds.loc[:, columns].copy(deep=True)
            return data

        return data

    @timing
    @checkpoint("step3")
    def step_3(self) -> None:
        """
        Method performs step 3 of ICS experiment.
        It generates detector AD training models and data.
        Args:
            <None>
        Returns:
            <None>
        """
        self.log.debug("STEP.3 - AD model training")

        if not os.path.isdir(self.m_path):
            os.mkdir(self.m_path)

        i = 1

        roc_files = {}
        for (prob_set_str,
             algo_name,
             attack_name,
             data_file,
             m_config) in IterHelper.exp_loop_generator(self.cfg,
                                                        self.algo,
                                                        self.attack_names,
                                                        self.attack_files,
                                              None):

            # Prune data to be presented as only test data
            # in accordance to the predefined test limits
            t_data = pd.read_csv(data_file)
            t_data = t_data.loc[:self.test_limit-1, ]

            # self.log.info(f"Test Limit set: [{0}:{self.test_limit-1}] => {t_data.shape[0]} samples")

            if prob_set_str not in roc_files.keys():
                roc_files[prob_set_str] = {}
            if algo_name not in roc_files[prob_set_str].keys():
                roc_files[prob_set_str][algo_name] = {}
            if attack_name not in roc_files[prob_set_str][algo_name].keys():
                roc_files[prob_set_str][algo_name][attack_name] = []

            self.log.debug(
                f"[{i}/{self.algo_max_count}] T [{algo_name}][{attack_name}] => "
                f"{data_file} [{str(t_data.shape)}] > limit [{self.test_limit}]"
            )

            t_data = self.input_filter(t_data, m_config)

            out_file_template = (f"{data_file[:-4]}_{algo_name}_".
                                 replace("/attack/", "/model/").
                                 replace("_mir", ""))

            enable_line = "search_param" in ad_config[algo_name].keys()

            # Search for optimal parameters - minimize false positives
            # and select best anomaly detection model based on the given criteria
            (best_result,
             self.meta_params[algo_name + "_it"],
             opt_box) = self.rocg[algo_name].roc_search_loop(
                t_data,
                self.meta_params[algo_name],
                self.detectors[algo_name],
                m_config,
                out_file_template,
                f"{algo_name.upper()} - {attack_name} - AD training ROC",
                True,
                enable_line,
                None
            )

            roc_files[prob_set_str][algo_name][attack_name].append(out_file_template)

            # Model update
            if best_result is not None and "cfg" in best_result.keys():
                self.best_models[prob_set_str][algo_name][attack_name] = best_result.copy()
            else:
                self.log.error(f"Algo |{algo_name}|, Attack |{attack_name}| not trained properly")
                continue

            # Model training parameters printout
            params = best_result["cfg"].copy()
            if params is not None and "label" in params.keys():
                del params["label"]

            self.log.info(f"Train best params: |{json.dumps(params, indent=4)}|")

            i += 1

            if self.cfg["model"]["animation"]:
                Plotter.input_ts(t_data.loc[:, "time"],
                                 t_data.loc[:, ["time", "temp"]],
                                 "STEP3 Input Data",
                                 f"{best_result['file'][:-4]}_input_step3.png")
                try:
                    for(prob_set_str, algo_name, attack_name) in \
                            (IterHelper.exp_loop_short_generator(self.cfg,
                                                                 self.algo,
                                                                 self.attack_names)):
                        # files = glob.glob(f"{roc_files[prob_set_str][algo_name][attack_name][0]}m*roc.png")
                        # self.log.debug(f"Files:{prob_set_str} > {algo_name} > {attack_name}")
                        # self.log.debug(f"{roc_files[prob_set_str][algo_name][attack_name]}")
                        # self.log.debug(f"found files: [{files}]")
                        Plotter.animate_image(f"{roc_files[prob_set_str][algo_name][attack_name][0]}m*roc.png",
                                             f"{roc_files[prob_set_str][algo_name][attack_name][0]}anim.gif")
                except Exception as ex:
                    self.log.exception(ex)

    @timing
    @checkpoint("step4")
    def step_4(self) -> None:
        """
        Method performs step 4 of ICS experiment.
        It generates detectors AD model detection and data.
        Args:
            <None>
        Returns:
            None
        """
        # STEP.4. Perform anomaly detection on various attack data
        self.log.debug("STEP.4 - AD model Anomaly detection")

        i = 1

        for (prob_set_str,
             algo_name,
             attack_name,
             data_file,
             m_config) in IterHelper.exp_loop_generator(self.cfg,
                                                        self.algo,
                                                        self.attack_names,
                                                        self.attack_files,
                                                        self.best_models):
            self.log.debug(
                f"[{i}/{self.algo_max_count}] D [{algo_name}] => {data_file}"
            )

            a_data = pd.read_csv(data_file)
            # a_data_total = a_data.shape[0]
            a_data = a_data.loc[self.test_limit:, ]
            a_data = self.input_filter(a_data, m_config)

            # self.log.info(f"Test Limit set: [{self.test_limit}:{a_data_total}] => {a_data.shape[0]} samples")

            model_files = f"{self.path}model/a_data_{attack_name}_{prob_set_str}_{algo_name}_0_p*.{m_config['ext']}"

            if self.best_models[prob_set_str][algo_name][attack_name]["model"]:
                # Extract model number
                model_number = self.best_models[prob_set_str][algo_name][attack_name]["model"].split("_")[-2]

                # Select all files that belong to the best model only
                model_files = [
                    model_file.replace("\\", "/")
                    for model_file in glob.glob(f"{self.path}model/a_data_{attack_name}_{prob_set_str}_{algo_name}_"
                                                f"{model_number}_p*.{m_config['ext']}")
                    if os.path.isfile(model_file)
                ][0]

            rcm_file = f"{self.path}detection/a_data_{attack_name}_{prob_set_str}_{algo_name}_roc"

            enable_line = "search_param" in ad_config[algo_name].keys()

            # This method generates data for each point for
            # ROC curve for AD evaluation step
            best_result, params_it, opt_box = self.rocg[algo_name].roc_search_loop(
                a_data,
                self.meta_params[algo_name],
                self.detectors[algo_name],
                m_config,
                rcm_file,
                f"{algo_name.upper()} - {attack_name} - AD evaluation ROC",
                False,
                enable_line,
                model_files,
                self.best_models[prob_set_str][algo_name][attack_name]
            )

            if "model" in best_result.keys():
                self.best_models[prob_set_str][algo_name][attack_name] = best_result.copy()
            else:
                best_result = self.best_models[prob_set_str][algo_name][attack_name].copy()

            model_file = self.best_models[prob_set_str][algo_name][attack_name][
                "model"
            ].replace("\\", "/")

            self.best_models[prob_set_str][algo_name][attack_name]["model"] = model_file

            self.best_files.append(model_file[model_file.rfind("/") + 1 :])

            # Remove label to show only AD parameters used in the detection
            display_result = best_result["cfg"].copy()

            if "label" in display_result.keys():
                del display_result["label"]
            self.log.info(
                f"Detect best params: |{json.dumps(display_result, indent=4)}|"
            )

            metric_file = f"{self.path}{self.cfg['results']['path']}a_data_{prob_set_str}_metric.csv"

            if self.cfg["model"]["animation"]:
                Plotter.input_ts(a_data.loc[:, "time"],
                                 a_data.loc[:, ["time", "temp"]],
                                 "STEP4 Input Data",
                                 f"{rcm_file}_input_step4.png")

            self.metric_data = self.diff_metrics.record(prob_set_str,
                                                        algo_name,
                                                        attack_name,
                                                        "ad",
                                                        m_config["knowledge"],
                                                        self.metric_data,
                                                        best_result,
                                                        metric_file)
            i += 1

    @timing
    @checkpoint("step5")
    def step_5(self) -> None:
        """
        Method performs step 5 of ICS experiment.
        It generates graphs for detections and overall experiment outcomes.
        Args:
            <None>
        Returns:
            None
        """
        if not self.cfg["cycle_detector"]["enabled"]:
            return

        # Perform comparison and create confusion matrix
        # ----------------------------------------------------
        self.log.debug("STEP.5 - Results generation")
        # ----------------------------------------------------

        if not self.cfg["results"]["graphs"]["enabled"]:
            return

        self.log.debug("Cycle detector evaluation ...")

        i = 1

        # Only one model and data set from evaluation
        # per prob_set_str/algo_name/attack_name
        for (
            prob_set_str,
            algo_name,
            attack_name,
            m_config,
        ) in IterHelper.exp_loop_model_generator(self.cfg,
                                                 self.algo,
                                                 self.attack_names,
                                                 self.best_models):
            self.log.debug(f"STEP 5 Attack: {attack_name}")

            # TODO: Validate if the loop is required here
            out_files = [f"{self.best_models[prob_set_str][algo_name][attack_name]['model'][:-4]}.csv"]

            # ------------------ Loop computation -------------------
            for file_name in out_files:
                self.log.debug(f"[{i}/{self.algo_max_count}] CD: {file_name}")
                enable_line = "search_param" in ad_config[algo_name].keys()
                l_cfg = {**self.best_models[prob_set_str][algo_name][attack_name],
                        "knowledge": m_config["knowledge"]}

                # Run the model with tau delay added to detect anomalies per cycle
                metric_vals = self.cd.run(l_cfg,
                                          algo_name,
                                          attack_name,
                                          prob_set_str,
                                          enable_line)

                metric_file = f"{self.path}{self.cfg['results']['path']}a_data_{prob_set_str}_metric.csv"

                self.metric_data = self.diff_metrics.record(prob_set_str,
                                                            algo_name,
                                                            attack_name,
                                                            "cd",
                                                            m_config["knowledge"],
                                                            self.metric_data,
                                                            metric_vals,
                                                            metric_file)
                i += 1

            # ------------------ Loop computation -------------------
            self.log.debug(f"Added algo: {algo_name.upper()}")

        self.diff_metrics.run()

    @timing
    @checkpoint("step6")
    def step_6(self) -> None:
        """
        Method generates whole experiment visualization results plus
        Args:
            <None>
        Returns:
            None
        """

        self.log.debug(
            "STEP.6 - Aggregated content generation from all experimental results"
        )

        r_path = f"{self.path}{self.cfg['results']['path']}"

        if not os.path.isdir(r_path):
            os.mkdir(r_path)

        # Clean up the files after experiment to save disk space (before deletion
        # process introduction it was more than 1GB per single experiment).
        # Remove *.pkl and *.csv files from training and detection except best model
        if self.cfg["results"]["size"]["remove_files"]:
            # Keep pkl file that belong to best models
            ftd = [x[x.rfind("\\") + 1 :] for x in glob.glob(f"{self.m_path}*.pkl")]
            ftd = [x for x in ftd if x not in self.best_files]

            # Remove other model files
            for file in [x for x in ftd if os.path.isfile(f"{self.m_path}{x}")]:
                os.remove(f"{self.m_path}{file}")

        # Run post experiment information and pass each step timing info
        self.post.run(r_path, self.timing)

    def run(self) -> None:
        """
        Method performs ADEF experiment in accordance to
        required configuration.
        Args:
            <None>
        Returns:
            <None>
        """
        start_exp = datetime.now()

        for step_name, step in self.steps_table.items():
            self.log.debug(f"Running: {step_name}")
            try:
                start = datetime.now()
                if step is not None:
                    step()
                stop = datetime.now()
                self.timing[step_name] = (stop - start).total_seconds()
            except Exception as ex:
                self.log.exception(ex)
                self.log.error(f"Simulation stopped at {step_name}")
                break
            finally:
                self.log.debug(f"Completed {step_name}")

        # Post-processing
        # Check if auto-shutdown should be performed to minimize electricity usage
        self.log.debug(
            f"Experiment completed in {(datetime.now() - start_exp).total_seconds():.2f} seconds"
        )

        # Automatically shut down PC after experiment
        # to save energy
        if not self.meta_mode and self.cfg["global"]["autoshutdown"]:
            Utility.auto_shutdown(datetime.now(), self.log)


def main(argv: dict) -> None:
    """
    Method starts simulation process with given input parameters
    Args:
        argv: <dict> - list of key value pairs with argument name
                       and value that will be used in the experiment run
    Returns:
        None
    """
    random.seed(str(uuid.uuid4()))
    log_file = "cps_experiment.log"
    config_file = "./cfg/config.yml"

    # TODO: Add argument parsing so ics_experiment can run independently
    # TODO: on its own without meta experiment level, so single experiment
    # TODO: can be executed as well

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", help="YAML configuration file name")
    parser.add_argument("--log", help="Logging file name")

    args = parser.parse_args(argv)

    if args.cfg:
        config_file = args.cfg
    if args.log:
        log_file = args.log

    # Multi process logger object initialization
    l2f = Log(log_file, 20)

    # Load simulator configuration
    cfg = yc.toJson(config_file)

    exp = Experiment(cfg, l2f)
    exp.run()

    if l2f is not None:
        l2f.close()


if __name__ == "__main__":
    main(sys.argv[1:])
