"""ICS Attack Scenario Generator module"""


from __future__ import annotations
import pandas as pd
import os
import sys

PATHS = ["./", "./tools"]
for path in PATHS:
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), path))
    if path not in sys.path:
        sys.path.insert(1, path)

from log import Log
from tools.plotter_helper import Plotter
from mev.data.eng import DataEng
from ics_attack_base import ICSAttackBase
from t_mixed import T_MIXED


class AttackScenario:
    """Class describing simple and complex attack scenarios"""

    @staticmethod
    def single_attack(
        enabled_attacks: dict,
        a_path: str,
        density: int,
        p_attacker: ICSAttackBase,
        data: pd.DataFrame,
        data_mir: pd.DataFrame,
        cfg: dict,
        log: Log,
    ) -> None:
        """
        Method generates attacks on ICS production cycle
        Args:
            enabled_attacks - list of enabled attacks
            a_path - attack path where attacks files will be written
            density - probability of attack density
            p_attacker - attacker object
            data - input process data
            data_mir - mirrored data used with apriori knowledge
            cfg - configuration object
            log - logger object
        Returns:
            <None>
        """
        # a_path = f"{cfg['global']['path']}attack/"

        # Separate attacks generation for training data
        for attack_id, attack in enabled_attacks.items():

            attack_name = attack["name"].lower()
            a_data = None
            det_limit = cfg["process"]["detection_cycles"]

            cfg["attack"]["file"] = f"{a_path}a_data_{attack_name}_{int(density * 100):02d}.csv"
            if os.path.isfile(f"{a_path}{cfg['attack']['types'][attack_name]['file']}"):
                cfg["attack"]["file"] = f"{a_path}{cfg['attack']['types'][attack_name]['file']}"
                a_data = pd.read_csv(cfg["attack"]["file"])
            else:
                split_idx = (
                        data.shape[0]
                        - (cfg["process"]["cycle_len"] * cfg["process"]["samples"] * det_limit)
                )

                if "mixed" == attack_name:
                    a_data = T_MIXED.mixed_attack(enabled_attacks,
                                                  a_path,
                                                  density,
                                                  p_attacker,
                                                  data,
                                                  data_mir,
                                                  cfg,
                                                  log)
                else:
                    try:
                        # Inject faults to attack trace
                        if cfg["attack"]["training"]:
                            a_data = p_attacker.attack(data, cfg, density, attack_id)
                        # Training is clear from attacks and first half of the trace has to be set
                        # clear of any attack indication
                        else:
                            a_data = pd.concat([data.loc[:data.index[split_idx]],
                                                p_attacker.attack(data.loc[data.index[split_idx]:,],
                                                                  cfg,
                                                                  density,
                                                                  attack_id)
                                                ])
                            a_data.loc[:split_idx, ["label", "type"]] = 0

                    except Exception as ex:
                        log.exception(ex)

            # Manifold attack data if process knowledge will be used in anomaly detection
            a_data_mir = DataEng.manifold_data(data,
                                               data_mir,
                                               a_data,
                                               f"{cfg['attack']['file'][:-4]}")

            for d in [[a_data, "_oa.png"],
                      [a_data_mir, "_oa_mir.png"]]:
                if not d[0].empty:
                    attack_file_name = cfg["attack"]["file"] \
                        if "_mir" not in d[1] else \
                        f"{cfg['attack']['file'][:-4]}_mir.csv"
                    d[0].to_csv(attack_file_name)
                    log.debug(f"Attack {attack_name}: {cfg['attack']['file']} => OK")
                    # Generate original attack traces
                    Plotter.input_ts_color_split(
                        d[0],
                        det_limit,
                        f"Attack {attack_name.upper()}",
                        cfg,
                        f"{cfg['attack']['file'][:-4]}{d[1]}",
                        False,
                    )
                else:
                    log.warn(f"Data NOT generated for: {str(attack_id).upper()} => {attack_file_name}")
                    continue

    @staticmethod
    def attack_routine(
        cfg: dict,
        data: pd.DataFrame,
        data_mir: pd.DataFrame,
        p_attacker: object,
        density: int,
        attack_list: list,
        log: Log,
    ) -> pd.DataFrame:
        """
        Args:
            cfg: attack configuration
            data: ICS process input data
            data_mir: ICS mirrored ideal trace in case of apriori proces knowledge
            p_attacker: attack class responsible for generating an attack
            density: attack probability on a given trace
            attack_list: list of attacks used in mixed mode attack type
            log: logger object
        Returns:
            <pd.DataFrame> - attacked cycles
        """
        a_data = p_attacker.attack(data, cfg, density, attack_list)

        if a_data is not None:
            if "file" in cfg["attack"]:
                a_data.to_csv(cfg["attack"]["file"])
                if len(attack_list) < 2:
                    log.debug(f"Attack [{attack_list}]: {cfg['attack']['file']} => OK")
            elif len(attack_list) < 2:
                log.debug(f"Attack [{attack_list}]: => OK")
        else:
            log.error(f"Data NOT generated for: {str(attack_list).upper()}")
            return None

        return a_data

    @staticmethod
    def mixed_attack(
        enabled_attacks: dict,
        a_path: str,
        density: int,
        p_attacker: object,
        data: pd.DataFrame,
        data_mir: pd.DataFrame,
        cfg: dict,
        log: Log,
    ) -> None:
        """
        Method generates attacks on ICS process cycle
        Args:
            enabled_attacks - list of enabled attacks
            a_path - attack path where attacks files will be written
            density - probability of attack density
            p_attacker - attacker object
            data - input process data
            data_mir - mirrored data used with apriori knowledge
            cfg - configuration object
            log - logger object
        Returns:
            <None>
        """

        cfg["attack"]["file"] = f"{a_path}a_data_mixed_{int(density * 100):02d}.csv"

        a_data = AttackScenario.attack_routine(
            cfg,
            data,
            data_mir if data_mir is not None else None,
            p_attacker,
            density,
            list(enabled_attacks.keys()),
            log,
        )

        log.debug(
            f"Attack Detection [{list(enabled_attacks.keys())}] {cfg['attack']['file']} => OK"
        )

        det_limit = cfg["process"]["detection_cycles"]

        split_idx = (
            int(
                a_data.shape[0]
                / (cfg["process"]["cycle_len"] * cfg["process"]["samples"])
            )
            - det_limit
        )

        # Generate original attack traces without manifold
        Plotter.input_ts_color_split(
            a_data,
            split_idx,
            f"Attack MIXED",
            cfg,
            f"{cfg['attack']['file'][:-4]}_oa.png",
            False,
        )

        # Manifold attack data if process knowledge will be used in anomaly detection
        a_data_mir = DataEng.manifold_data(
            data, data_mir, a_data, f"{cfg['attack']['file'][:-4]}"
        )

        if a_data_mir is not None:
            a_data_mir.to_csv(f"{cfg['attack']['file'][:-4]}_mir.csv")
            Plotter.input_ts_color_split(
                a_data_mir,
                split_idx,
                f"Attack MIXED - Mirrored",
                cfg,
                f"{cfg['attack']['file'][:-4]}_mir_oa.png",
                False,
            )
        else:
            log.error(
                f"Problem with generating mir file: {cfg['attack']['file'][:-4]}_mir.csv"
            )
