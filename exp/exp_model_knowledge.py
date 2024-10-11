""" Configuration builder for model knowledge experiment """

import os
import sys
import copy
import pandas as pd

PATHS = ["./", "../tools", "../threats", "../pres/ppt", "../pres/latex"]
for path in PATHS:
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), path))
    if path not in sys.path:
        sys.path.insert(1, path)

from base_exp import BaseExperimentConfig
from tools.log import Log
from tools.intr import Interpreter

from cfg_builder import CfgBuilderHelper
from tools.plotter_helper import Plotter
from icsr.threats.attacker import ICS_Attacker
from icsr.threats.threat_class import ThreatClass
from icsr.threats.threat_factory import ThreatFactory
from icsr.threats.attack_scenario import AttackScenario
from mev.result.ppt.ppt_model_knowledge import ModelKnowledgePresentation


class ExpModelKnowledge(BaseExperimentConfig):
    """
    Model knowledge experiment configuration class
    """

    @staticmethod
    def generate_attack_lists(cfg: dict) -> list:
        """
        Method generates ranges of attacks that will be injected in attack
        section for training set.
        Args:
            cfg: experiment configuration
        Returns:
            <list>: list of list containing attacks
        """
        if "attacks_selection" not in cfg.keys():
            return []

        attack_list = cfg["attacks_selection"]["attacks"]
        start = cfg["attacks_selection"]["from"] - 1
        stop = cfg["attacks_selection"]["to"] + 1
        step = cfg["attacks_selection"]["step"]
        attack_lists = []
        for i in range(start + 1, stop, step):
            attack_lists.append(attack_list[start:i])
        return attack_lists

    @staticmethod
    def count_attacks(cfg: dict, data: pd.DataFrame) -> dict:
        """
        Method counts number of attacks found in the data set.
        Args:
            cfg: experiment configuration
            data: input data used to count attacks
        Returns:
            <dict>: returns attack count for each type grouped in
                    attack_name: attack_count key/value pair.
        """
        result = {}
        attack_map = {i.name.lower(): i.value for i in ThreatClass}

        at_map = pd.DataFrame(
            {"key": list(attack_map.keys()), "val": list(attack_map.values())}
        )

        cycle_len = cfg["process"]["cycle_len"] * cfg["process"]["samples"]

        for i in range(data.index[0], data.index[-1] - cycle_len, cycle_len):
            cycle = data.loc[i : i + cycle_len - 1, "type"]
            if cycle.sum() == 0:
                continue

            attack = cycle[cycle != 0]

            attack_value = attack[attack.index[0]]
            key = at_map[at_map.val == attack_value]["key"].tolist()[0]

            if key not in result.keys():
                result[key] = 0

            result[key] += 1

        return result

    @staticmethod
    def plot_ts(data: pd.DataFrame,
                cfg: dict,
                file_name: str) -> None:
        data.set_index("time", inplace=True)
        Plotter.input_ts_color_split(
            data,
            cfg["global"]["detection"]["limit"],
            f"Model knowledge original input",
            cfg,
            file_name,
            False,
        )
        data.reset_index(inplace=True)

    def get(self, *args) -> list:
        """
        Method generates model knowledge experiment configuration.
        Args:
            <None>
        Returns:
            <list>: list of generated configurations of atomic experiment
        """
        n_cfg = []

        d_cfg = copy.copy(self.d_cfg)
        d_cfg["pre"] = self.cfg["pre"]
        d_cfg["mode"] = True
        Interpreter.run(
            [
                f"md {d_cfg['global']['path']}experiments/{d_cfg['experiment']['path']}",
                f"md {d_cfg['global']['path']}experiments/{d_cfg['experiment']['path']}summary",
                f"rd {d_cfg['global']['path']}experiments/{d_cfg['experiment']['path']}checkpoint",
                f"rd {d_cfg['global']['path']}experiments/{d_cfg['experiment']['path']}output",
            ]
        )

        attack_lists = ExpModelKnowledge.generate_attack_lists(d_cfg["pre"])

        full_attack_list = self.cfg["global"]["attacks_selection"]["attacks"]
        cycle_len = d_cfg["process"]["cycle_len"] * d_cfg["process"]["samples"]
        cycle_tr_limit = d_cfg["global"]["detection"]["limit"]
        training_limit = (cycle_tr_limit * cycle_len) - 1

        if not attack_lists:
            self.log.error(f"Couldn't find correct attack list ")
            n_cfg.append(None)
            return n_cfg

        # TODO: Fix the problem that generates data for all attacks while this experiment
        #  design shouldn't.
        for attack_list in attack_lists:
            local_exp_cfg = copy.deepcopy(d_cfg)
            original_attack_config = copy.deepcopy(local_exp_cfg["attack"]["types"])

            exp_path = (f"{d_cfg['global']['path']}experiments/{d_cfg['experiment']['path']}"
                        f"{len(attack_list)}/")
            local_exp_cfg["experiment"]["path"] = f"{local_exp_cfg['experiment']['path']}{len(attack_list)}/"
            local_exp_cfg["experiment"]["pre"]["cycle_loop"] = [d_cfg["global"]["detection"]["limit"]]
            local_exp_cfg["process"]["cycles"] = d_cfg["global"]["detection"]["limit"]
            local_exp_cfg["process"]["detection_cycles"] = d_cfg["global"]["detection"]["limit"]
            local_exp_cfg["attack"]["types"]["mixed"]["sets"][0]["attack_list"] = attack_list
            Interpreter.run(
                [
                    f"md {exp_path}",
                    f"md {exp_path}attack",
                    f"md {exp_path}input",
                    f"md {exp_path}model",
                    f"md {exp_path}graph",
                    f"md {exp_path}error",
                    f"md {exp_path}output",
                    f"md {exp_path}detection",
                    f"md {exp_path}checkpoint",
                    f"md {exp_path}timing",
                    f"cp {d_cfg['global']['path']}input/*.png {exp_path}input/",
                    f"cp {d_cfg['global']['path']}input/*.csv {exp_path}input/",
                ]
            )

            # TODO: This is working incorrectly
            # attacks_to_remove = [el for el in full_attack_list if el not in attack_list]

            ThreatFactory.init(d_cfg, self.log)

            # attack_ids = sorted(
            #    [
            #        ThreatClass.values[attack_name.upper()]
            #        for attack_name in attacks_to_remove
            #    ]
            #)

            data_file = CfgBuilderHelper.get_file(
                f"{d_cfg['global']['path']}/input/*.csv", "_mir.csv"
            )
            attack_file = CfgBuilderHelper.get_file(
                f"{d_cfg['global']['path']}/attack/*mixed*.csv", "_mir.csv"
            )

            full_data = pd.read_csv(data_file)
            a_data = pd.read_csv(attack_file)
            # a_tr_data = a_data.loc[:training_limit,].copy(deep=True)

            attack_file_d = attack_file[attack_file.rindex("\\") + 1:]
            attack_file_d_list = attack_file_d.split("_")
            attack_file_d_list[2] = "mixed"
            attack_file_d = "_".join(attack_file_d_list)
            attack_file_dst = f"{exp_path}attack\\{attack_file_d}"
            attack_mir_file_orig = CfgBuilderHelper.get_file(f"{d_cfg['global']['path']}/attack/*_mir.csv")

            # MIR file update is computed from difference between attack and original data
            attack_mir_file = f"{attack_file_dst[:-4]}_mir.csv"

            a_mir_data_orig = pd.read_csv(attack_mir_file_orig)

            ExpModelKnowledge.plot_ts(a_data,
                                      local_exp_cfg,
                                      f"{attack_file_dst[:-4]}_orig.png")

            ExpModelKnowledge.plot_ts(a_mir_data_orig,
                                      local_exp_cfg,
                                      f"{attack_mir_file[:-4]}_oa_orig.png")

            if len(attack_list) == (len(local_exp_cfg["attack"]["types"]) - 1):
                a_data.to_csv(attack_file_dst, index=False)
                a_mir_data_orig.to_csv(attack_mir_file, index=False)
            else:
                data = full_data.loc[:training_limit,].copy(deep=True)

                # Find cycles for attack replacement and regeneration in accordance to
                # new attack list (random selection
                density = local_exp_cfg["attack"]["densities"][0]
                data_mir = None

                # Toggle on all attacks for purpose of attack generation
                local_exp_cfg["attack"]["types"] = {k: {**v, 'enabled': True}
                                                    for k, v in local_exp_cfg["attack"]["types"].items()}
                p_attacker = ICS_Attacker(local_exp_cfg, self.log)

                a_tr_data_x = (AttackScenario.
                               attack_routine(local_exp_cfg,
                                              data,
                                              data_mir if data_mir is not None else None,
                                              p_attacker,
                                              density,
                                              attack_list,
                                              self.log))

                local_exp_cfg["attack"]["types"] = copy.deepcopy(original_attack_config)

                # Instead of replacing - just generate new trace of X length with specific number of attacks
                '''
                for val in range(0, self.cfg["global"]["detection"]["limit"]):
                    idx_low = val * cycle_len
                    idx_hi = ((val + 1) * cycle_len) - 1
                    attack_id = sorted(a_tr_data.loc[idx_low:idx_hi, "type"].unique())[-1]
    
                    if (
                        a_tr_data.loc[idx_low:idx_hi, "type"].sum() == 0
                        or attack_id not in attack_ids
                    ):
                        a_tr_data_x = pd.concat(
                            [a_tr_data_x, a_tr_data.loc[idx_low:idx_hi, a_tr_data.columns]]
                        )
                        continue
    
                    new_a_data = p_attacker.attack(data.loc[idx_low:idx_hi,],
                                                   local_exp_cfg,
                                                   density,
                                                   attack_list)
    
                    new_a_data.reset_index(inplace=True)
    
                    a_tr_data_x = pd.concat(
                        [a_tr_data_x, new_a_data.loc[:, a_tr_data.columns]]
                    )
                '''
                # Reset indexing after multiple df concatenation
                a_tr_data_x.drop("index", axis=1, inplace=True)
                a_tr_data_x.reset_index(inplace=True)

                # Update the rows in dfA with the rows from dfB
                """
                cnt_a = ExpModelKnowledge.count_attacks(local_exp_cfg, a_tr_data_x)
                cnt_b = ExpModelKnowledge.count_attacks(local_exp_cfg, a_data.loc[training_limit + 1:, ])
                log.debug(f"Req : |{attack_list}|")
                log.debug(f"Counted: |{sorted(list(cnt_a.keys()))}|")
                log.debug(f"Changed: |{cnt_a}|")
                log.debug(f"Orig   : |{cnt_b}|")
                """

                m_data = pd.concat([a_tr_data_x, a_data.loc[training_limit + 1 :,]])
                if m_data.index.isna().all():
                    self.log.error("a_data index is incorrect!!! please FIX IT!!!")
                    continue
                else:
                    ExpModelKnowledge.plot_ts(m_data,
                                              local_exp_cfg,
                                              f"{attack_file_dst[:-4]}_oa.png")

                m_data.to_csv(f"{attack_file_dst}", index=False)
                m_data = m_data.loc[:training_limit, :]
                temps = m_data.loc[:, "temp"].copy(deep=True)
                m_data.loc[:, "temp"] = temps - full_data.loc[:training_limit, ["tc"]]["tc"]
                m_data = pd.concat([m_data, a_mir_data_orig.loc[training_limit+1:, :]])

                # Validate new result a proper count of attacks vs. normal
                # traces is required before and after update of training data section
                m_data.to_csv(f"{attack_mir_file}", index=False)
                ExpModelKnowledge.plot_ts(m_data,
                                          local_exp_cfg,
                                          f"{attack_mir_file[:-4]}_oa.png")

            n_cfg.append(local_exp_cfg)

            self.log.debug(f"Configuration build for model knowledge "
                           f"experiment with attacks for training: {attack_list}")

        self.cfg["cfgs"] = n_cfg

        return n_cfg

    def post(self, *args) -> None:
        """
        Method generates data for model knowledge summary presentation
        Args:
            cfgs: configuration of the information to be fetched from experiment
            log: logger object
            *args: other input arguments required to complete data preparation
        Returns:
            <None>
        """
        files = []
        sum_data = pd.DataFrame()
        for cfg in self.cfg["cfgs"]:
            cycle_no = cfg["experiment"]["pre"]["cycle_loop"][0]
            for prob_set in cfg["attack"]["densities"]:
                prob_set_str = f"{int(prob_set * 100):02d}"
                exp_sum_file = (
                    f"{cfg['global']['path']}experiments/"
                    f"{cfg['experiment']['path']}"
                    f"{cfg['results']['path']}"
                    f"final_{cycle_no}_{prob_set_str}_sum.csv"
                )
                if not os.path.isfile(exp_sum_file):
                    self.log.warn(f"File summary data {exp_sum_file} not found!")
                    continue

                exp_sum_row = pd.read_csv(exp_sum_file)
                exp_sum_row.insert(0, "cycle", [cycle_no] * exp_sum_row.shape[0])
                # exp_sum_row.insert(
                #    0,
                #    "prob",
                #    [np.round(prob_set, 2)] * exp_sum_row.shape[0],
                # )
                sum_data = pd.concat([sum_data, exp_sum_row])

        # Write a meta level information to a CSV file
        file_path = f"{self.cfg['cfgs'][0]['experiment']['path']}"[:-1]
        file_path = file_path[: file_path.rindex("/")]
        file_name = (
            f"{self.cfg['cfgs'][0]['global']['path']}experiments/"
            f"{file_path}/summary/"
            f"meta_{self.cfg['cfgs'][0]['experiment']['id'].lower()}"
        )

        if sum_data is None or sum_data.empty:
            self.log.error(f"Data for {file_name}.csv is not available or is empty")
            return
        if os.path.isfile(f"{file_name}.csv"):
            prev_sum_data = pd.read_csv(f"{file_name}.csv")
            if sum_data is not None or sum_data.empty:
                sum_data = pd.concat([prev_sum_data, sum_data])
            else:
                sum_data = prev_sum_data

        sum_data.to_csv(f"{file_name}.csv", index=False)

        # Generate per training data performance graph
        if (
            len(sum_data.loc[:, "cycle"].unique().tolist()) > 1
            or len(sum_data.loc[:, "attack_no"].unique().tolist()) > 1
        ):
            def_labels = ["prob", "cycle", "algo", "attack", "type", "attack_no"]
            for label in ["auc", "acc", "f1", "eer"]:
                columns = def_labels + [label]
                Plotter.meta_per_attack(
                    sum_data.loc[:, columns],
                    f"Experiment - Model knowledge {label.upper()}evaluation",
                    f"{file_name}.png",
                    self.log,
                    label,
                )

            # TODO: Change name of the presentation
            # Generate presentation
            sum_prs = ModelKnowledgePresentation(self.cfg["cfgs"][0], self.log)
            exp_name = self.cfg["cfgs"][0]["experiment"]["id"].lower()

            sum_pres_file = f"{file_path}/summary/{exp_name}_summary.pptx"
            sum_prs.run(None, sum_pres_file)

            # TODO: Add LaTeX routine here
