""" Autoclave process attacker implementation """


from __future__ import annotations
import os
import sys

import random
import uuid
import numpy as np
import pandas as pd

import tools.log as log
from tools.plotter_helper import Plotter

PATHS = ["./detectors", "./tools", "./threats", "./ics"]
for path in PATHS:
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), path))
    if path not in sys.path:
        sys.path.insert(1, path)

from autoclave.model.process import AutoclaveProcess
from ics_process_base import ICSAttackBase
from threat_factory import ThreatFactory
from threat_class import ThreatClass


class AutoclaveAttack(ICSAttackBase):
    """
    Class will define certain type of attack classes
    and inject them to provided profile
    """

    def attack(self,
               data: pd.DataFrame,
               cfg: dict,
               density: float,
               attack_type: object = None) -> pd.DataFrame:
        """
        Method generates attacks on autoclave profile
        depending on the configuration
        Args:
            data            - original data set
            cfg             - attack configuration
            density         - probability of an attack on the given cycle
            attack_type     - attack type that will be used during generation process
                              otherwise the attack trace is not generated
                              None - All enabled random of attacks will be used
                              list - Supplied list of attacks will be used to generate different attacks
        Returns:
            <pd.DataFrame>  - data with an attack
        """

        def __gen_attack_table(prob: float, num_cycles: int) -> list:
            """
            Method generates a list of time slots with
            enabled/disables autoclave profile slots that
            will be used to generate attack (True) or
            leave original trace without any changes.
            Args:
                prob: value of 0.0-1.0 representing probability
                      that will be applied in the experiment
                num_cycles: number of cycles used in the generation
                            process.
            Returns:
                <list> - list of True/False values labeling
                         attack windows for specific autoclave profile
            """
            t_attack_cycles = int(num_cycles * prob)
            t_non_attack_cycles = num_cycles - t_attack_cycles
            half_cycle = int(num_cycles / 2)
            h_attack_cycles_1 = int(t_attack_cycles / 2)
            h_attack_cycles_2 = t_non_attack_cycles - h_attack_cycles_1
            h_non_attack_cycles_1 = half_cycle - h_attack_cycles_1
            h_non_attack_cycles_2 = half_cycle - h_attack_cycles_2

            attack_list_1 = np.concatenate(
                (
                    np.zeros(h_non_attack_cycles_1, dtype=bool),
                    np.ones(h_attack_cycles_1, dtype=bool),
                )
            ).tolist()
            attack_list_2 = np.concatenate(
                (
                    np.zeros(h_non_attack_cycles_2, dtype=bool),
                    np.ones(h_attack_cycles_2, dtype=bool),
                )
            ).tolist()

            # Make sure that both half's of attack data
            # set have roughly same number of attacks
            # Reason being that test and validation will not be
            # skewed towards single class, making detection
            # much harder.
            random.shuffle(attack_list_1)
            random.shuffle(attack_list_2)

            attack_list = attack_list_1 + attack_list_2
            # Normalization
            if len(attack_list) < num_cycles:
                attack_list += [False] * (num_cycles - len(attack_list))
            return attack_list

        a_data = data.copy(deep=True)
        a_data.reset_index(inplace=True)
        a_data["label"] = 0
        a_data["type"] = 0
        data_len = len(a_data["label"])

        cycle = cfg["process"]["cycle_len"] * cfg["process"]["samples"]
        if data_len == cycle:
            attack_table = attack_type
        else:
            attack_table = __gen_attack_table(density, int(data_len / cycle))

        # Filter appropriate attack configuration
        attack_ranges = {}
        # Dictionary generating attack sample traces for precise attack review
        # it contains that enables/disables given single attack visualization
        attack_samples = {}

        attack_configs = {
            k: v for k, v in cfg["attack"]["types"].items() if v["enabled"]
        }

        for attack_t, config in attack_configs.items():
            if isinstance(attack_type, list) and attack_t not in attack_type:
                continue

            attack_ranges[attack_t] = config
            attack_samples[attack_t] = False

        cycle_idx = -1
        completion = 0
        i = 0

        mixed_attack = False
        if (
            attack_type is None
            or isinstance(attack_type, list)
            and len(attack_type) > 1
        ):
            mixed_attack = True

        ThreatFactory.init(cfg, self.log)

        # Change Tc values in accordance to attack schema
        for i in range(0, len(a_data.index), cycle):
            attack_per_cycle = 0
            cycle_idx += 1

            if not attack_table[cycle_idx]:
                continue

            attack_type_val = attack_type
            if mixed_attack:
                attack_idx = random.randint(0, len(attack_ranges.keys()) - 1)
                attack_type_val = list(attack_ranges.keys())[attack_idx]

            if isinstance(attack_type_val, list):
                attack_type_val = attack_type_val[0]

            config = attack_ranges[attack_type_val]

            for acfg in config["sets"]:
                # Attack shape randomization
                start = random.randint(acfg["start"][0], acfg["start"][1])
                stop = random.randint(acfg["stop"][0], acfg["stop"][1])

                # Normalize stop
                if (i + stop - 1) > len(a_data):
                    stop -= ((i + stop - 1) - len(a_data)) + 1

                # Prevent injecting attacks outside data allocation
                if (i + start) > len(data.index):
                    break

                # self.log.debug(f"Injecting attack: {config['name'].upper()}")

                # Call threat emulation on specified part of the data
                a_data.loc[i : i + cycle, ["tc", "type", "label"]] = ThreatFactory.run(
                    ThreatClass[config["name"].upper()],
                    config,
                    a_data.loc[i : i + cycle,].copy(False),
                    i + start,
                    i + stop,
                    i + cycle,
                    cfg["attack"]["soft_filter"],
                    self.log,
                ).loc[:, ["tc", "type", "label"]]
                attack_per_cycle += 1

                if cfg["attack"]["number_of_attacks_per_cycle"] == attack_per_cycle:
                    break

            # Rerun process simulation with changed Tc values to affect temperatures
            tt = a_data.index.values[i : i + cycle]

            # Take exactly N number of samples
            Tp = a_data.loc[:, "tc"].values[i : i + cycle]

            pid = AutoclaveProcess.PIDController(cycle, self.log)
            x, T, Tc = pid.run(tt, Tp)
            nlen = 29

            # Apply smoothing only in case of full data
            if len(T) > 9 and T[0] - T[8] > 2:
                T[:nlen] = np.random.random_sample(nlen) / 4 + T[8]

            completion = int((i / data_len) * 100)
            completion = int(round(completion / 10) * 10)
            if i % cfg["process"]["cycle_len"] == 0:
                self.log.debug(f"i[{i}] => Completed [{completion}]%", end="\r")

            a_data.loc[i : i + len(T) - 1, "temp"] = T
            a_data.loc[i : i + len(Tp) - 1, "tc"] = Tp

            if not attack_samples[attack_type_val]:
                if "file" in cfg["attack"].keys():
                    # Generate original attack traces without manifold
                    Plotter.input_ts_color(
                        data.loc[i : i + cycle,],
                        a_data.loc[i : i + cycle,],
                        f"Attack {config['name'].upper()}",
                        cfg,
                        f"{cfg['attack']['file'][:-4]}_sam.png",
                        False,
                    )
                attack_samples[attack_type_val] = True

            pid = None

        i += cycle
        completion = int((i / a_data.shape[0]) * 100)
        self.log.debug(f"i[{i}] => Completed [{completion}]%", end="\r")
        a_data.set_index("time", inplace=True)

        return a_data
