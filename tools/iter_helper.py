""" ADEF generators used in the experiment """


from __future__ import annotations

import copy
import gc
import itertools
import os
import re


class IterHelper:
    """
    Iterator helper class that implements generators for ADEF experiments
    """
    @staticmethod
    def exp_loop_generator(
        cfg: dict,
        algos: dict,
        attack_names: list,
        attack_files: dict,
        best_models: dict,
    ) -> (str, str, str, str, dict):
        """
        This is experiment loop iterator designed to flatten complex
        loop infrastructure and simplify workflow in steps 3-4

        # Usage:
        # for (prob_set_str, algo_name, attack_name, data_file, m_config, best_models) in self.exp_loop_iterator():
        #    self.log.debug(ProbSet: prob_set_str, AlgoName: algo_name,
        #                   AttackName: attack_name, FileName: file_name, Config: m_config)
        Args:
            cfg<dict> - configuration object containing experiment configuration
            algos<dict> -
            attack_names <list> -
            attack_files <dict> -
            best_models <dict> -
        Returns:
            <str> - probability string
            <str> - algorithm name
            <str> - attack name
            <str> - data file
            <dict> - algorithm configuration
        """
        for prob_set, algo_name, attack_name in itertools.product(
            cfg["attack"]["densities"], algos.keys(), attack_names
        ):
            # Deep neural networks are not supported yet - work in progress
            if "stat" != cfg["model"]["ad"][algo_name]["type"]:
                continue

            prob_set_str = f"{int(prob_set * 100):02d}"

            if (
                best_models is not None
                and {} == best_models[prob_set_str][algo_name][attack_name]
            ):
                # self.log.error(
                #    f"Algorithm[{algo_name}] <=> Attack[{attack_name}] => model not found."
                #    f" Please check model search parameters !!!"
                # )
                continue

            if (
                best_models is not None
                and "model"
                not in best_models[prob_set_str][algo_name][attack_name].keys()
            ):
                continue

            a_files = attack_files[prob_set_str][algo_name][attack_name]
            if not a_files:
                # self.log.warn(f"Attack |{attack_name}| not present in the input set.")
                continue

            m_config = algos[algo_name]

            if not m_config["knowledge"]:
                a_files = [x for x in a_files if not x.endswith("_mir.csv")]
            else:
                a_files = [x for x in a_files if x.endswith("_mir.csv")]

            if not a_files:
                # self.log.error(f"Attack files in {self.a_path} not found!!")
                continue

            for data_file in a_files:
                # On very loop iteration clear memory to keep usage to minimum
                gc.collect()

                if not os.path.isfile(data_file):
                    # self.log.error(f"Data file |{data_file}| is missing!!!")
                    continue

                if m_config["knowledge"] and "mir" not in data_file:
                    continue

                if not m_config["knowledge"] and "mir" in data_file:
                    continue

                yield (
                    prob_set_str,
                    algo_name,
                    attack_name,
                    data_file,
                    m_config,
                )

    @staticmethod
    def exp_loop_model_generator(
        cfg: dict,
        algos: dict,
        attack_names: list,
        best_models: dict,
    ) -> (str, str, str, dict):
        """
        This is experiment loop iterator designed to flatten complex
        loop infrastructure and simplify workflow in step 5

        # Usage:
        # for (prob_set_str, algo_name, attack_name, data_file, m_config, best_models) in self.exp_loop_model_generator():
        #    self.log.debug(ProbSet: prob_set_str, AlgoName: algo_name,
        #                   AttackName: attack_name, FileName: file_name, Config: m_config)
        Args:
            cfg<dict>: configuration object containing experiment configuration
            algos <dict>:
            attack_names <list>:
            best_models <dict>:
        Returns:
            <str> - probability string
            <str> - algorithm name
            <str> - attack name
            <dict> - algorithm configuration
        """
        for prob_set, algo_name, attack_name in itertools.product(
            cfg["attack"]["densities"], algos.keys(), attack_names
        ):
            # On very loop iteration clear memory to keep usage to minimum
            gc.collect()

            # Deep neural networks are not supported yet - work in progress
            if "stat" != cfg["model"]["ad"][algo_name]["type"]:
                continue

            prob_set_str = f"{int(prob_set * 100):02d}"

            if (
                best_models is not None
                and {} == best_models[prob_set_str][algo_name][attack_name]
            ):
                # self.log.error(
                #    f"Algorithm[{algo_name}] <=> Attack[{attack_name}] => model not found."
                #    f" Please check model search parameters !!!"
                # )
                continue

            if (
                best_models is not None
                and "model"
                not in best_models[prob_set_str][algo_name][attack_name].keys()
            ):
                continue

            m_config = algos[algo_name]

            yield (
                prob_set_str,
                algo_name,
                attack_name,
                m_config,
            )

    @staticmethod
    def exp_loop_short_generator(
        cfg: dict,
        algos: dict,
        attack_names: list,
    ) -> (str, str, str):
        """
        This is experiment loop iterator designed to flatten complex
        loop infrastructure and simplify iteration over ROC files

        # Usage:
        # for (prob_set_str, algo_name, attack_name) in self.exp_loop_short_generator():
        #    self.log.debug(ProbSet: prob_set_str, AlgoName: algo_name,
        #                   AttackName: attack_name)
        Args:
            cfg<dict>: configuration object containing experiment configuration
            algos <dict>:
            attack_names <list>:
            best_models <dict>:
        Returns:
            <str> - probability string
            <str> - algorithm name
            <str> - attack name
            <dict> - algorithm configuration
        """
        for prob_set, algo_name, attack_name in itertools.product(
            cfg["attack"]["densities"], algos.keys(), attack_names
        ):
            # On very loop iteration clear memory to keep usage to minimum
            gc.collect()

            # Deep neural networks are not supported yet - work in progress
            if "stat" != cfg["model"]["ad"][algo_name]["type"]:
                continue

            prob_set_str = f"{int(prob_set * 100):02d}"

            yield (prob_set_str,
                   algo_name,
                   attack_name)

    @staticmethod
    def extract_names(file_name) -> dict:
        """
        Method extracts the names of te
        """
        # String normalization
        f_name = copy.copy(file_name)
        f_name = f_name.replace("\\", "/") if "\\" in f_name else f_name
        f_name = f_name[f_name.rfind("/")+1:] if "/" in f_name else f_name

        result = {
            "algo_name": "",
            "attack_name": "",
            "model_no": "",
            "parameter_no": "",
            "prob": ""
        }

        for regex in [(r"a_data_(?P<attack_name>[a-z]+)_"
                       r"(?P<prob>[\d]{2,3})_"
                       r"(?P<algo_name>[a-z_]+)_"
                       r"m(?P<model_no>[\d]+)_"
                       r"p(?P<parameter_no>[\d]+)"),
                      (r"a_data_(?P<attack_name>[a-z]+)_"
                       r"(?P<prob>[\d]{2,3})_"
                       r"(?P<algo_name>[a-z_]+)_")]:
            match = re.match(regex, f_name, re.MULTILINE)
            if match is not None:
                result.update(match.groupdict())

        return result

    @staticmethod
    def extract_meta_names(file_name) -> dict:
        """
        Method extracts the names of te
        """
        # String normalization
        f_name = copy.copy(file_name)
        f_name = f_name.replace("\\", "/") if "\\" in f_name else f_name
        f_name = f_name[f_name.rfind("/")+1:] if "/" in f_name else f_name
        f_name = f_name.replace("skl_", "").replace("pytorch_", "").replace("pyod_", "")

        # meta_amountofdata_acc_pytorch_autoencoder_lin_mixed_acc_perf.png

        result = {
            "exp_name": "",
            "metric_name": "",
            "algo_name": "",
            "scale": "",
            "attack_name": ""
        }

        for regex in [(r"meta_(?P<exp_name>[a-z]+)_"
                       r"(?P<metric_name>[a-z\d]+)_"
                       r"(?P<algo_name>[a-z]+)_"
                       r"(?P<scale>[a-z]+)_"
                       r"(?P<attack_name>[a-z]+)_")]:
            match = re.match(regex, f_name, re.MULTILINE)
            if match is not None:
                result.update(match.groupdict())

        return result
