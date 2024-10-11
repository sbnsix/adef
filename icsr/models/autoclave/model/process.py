""" ICS autoclave profile generation to enable physics based autoclave process simulation. """

from __future__ import annotations

import gc
import os
import random
import sys
import warnings
from datetime import datetime
from scipy.integrate import ODEintWarning

from typing import Tuple

import numpy as np
import pandas as pd
from scipy.integrate import odeint

from scipy.stats import linregress

PATHS = ["./detectors", "./tools"]
for path in PATHS:
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), path))
    if path not in sys.path:
        sys.path.insert(1, path)

# ---------------------------------------------
#                Local imports
# ---------------------------------------------
from log import Log
from mt import interpolate_slope
from mt import noise_gen


class AutoclaveProcess:
    class PIDController:
        """
        PID controller that emulates physics based behavior of autoclave
        room where production parts are processed.
        """

        def __init__(self,
                     cycle_length: int,
                     logger: Log,
                     noise_flag: bool = True) -> None:
            """
            CTOR
            Args:
                cycle_length: length of the cycle
                logger: logger object
                noise_flag: flag that determines whether noise is present in the signal
            Returns:
                <None>
            """
            # Input parameters
            self.log = logger
            self.noise_flag = noise_flag
            self.cycle_length = cycle_length
            # Steady State Initial Conditions for the States
            self.Ca_ss = 0.87725294608097
            self.T_ss = 20.475443431599
            self.x0 = np.empty(2)
            self.x0[0] = self.Ca_ss
            self.x0[1] = self.T_ss

            # Steady State Initial Condition
            self.Tc_ss = 180.0

            # Noise generation
            sps = 10
            x = np.linspace(0, int(cycle_length / 3), int(cycle_length * sps / 3))
            n = noise_gen(x, sps / (2 * sps))

            # Time domain generation
            self.nx = [n[random.randint(0, 50 * sps)] for i in range(0, cycle_length+1)]

            # Optimal autoclave temperatures
            self.minimum_heating_temperature = 110
            self.optimal_heating_temperature = 175
            # Maximum operating range
            self.system_min = 5
            self.system_max = 250

            self.font = {
                "family": "arial",
                "weight": "bold",
                "size": 12,
            }

        def p_controller(self,
                         x: np.ndarray,
                         t: np.ndarray,
                         Tc: np.ndarray) -> np.ndarray:
            """
            Proportional controller emulating - autoclave process
            Args:
                x: current values
                t: time domain (sampling)
                Tc: control signal
            Returns:
                <np.ndarray>: precomputed model data
            """
            Ca = x[0]
            T = x[1]

            # Division by zero protection
            if T == 0:
                T = 0.00001

            # Controller parameters
            Tf = 21
            Caf = 1.0
            q = 1
            V = 200
            rho = 1000
            Cp = 0.239
            mdelH = 5e4
            EoverR = 8750
            k0 = 7.2e10
            UA = 5e4

            # Non-linear component cap to protect against
            # infeasible solutions.
            abb = round(-EoverR / T, 3)
            if abb > 20:
                abb = 20

            rA = k0 * np.exp(abb) * Ca
            dCadt = q / V * (Caf - Ca) - rA
            dTdt = (
                q / V * (Tf - T)
                + mdelH / (rho * Cp) * rA
                + UA / V / rho / Cp * (Tc - T)
            )
            xdot = np.zeros(2)
            xdot[0] = dCadt
            xdot[1] = dTdt

            return xdot

        def run(self, t, Tc) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            """
            Method runs autoclave process simulation
            with specific parameters designed to emulate autoclave profile
            Args:
                t: time domain
                Tc: desired autoclave profile
            Returns:
                (t, T, Tc): - time domain, autoclave temperature vector, temperature control vector
            """
            # Shuffle noise for each run
            random.shuffle(self.nx)

            # PID controller based simulation
            # Store results for plotting
            Ca = np.ones(len(t)) * self.Ca_ss
            T = np.ones(len(t)) * self.T_ss

            # Set tolerances
            atol = 1e-6
            rtol = 1e-9

            # Proportional controller to emulate autoclave process
            for i in range(len(t) - 1):
                ts = [t[i], t[i + 1]]
                Tc[i + 1] = np.clip(Tc[i + 1], self.system_min, self.system_max)

                with warnings.catch_warnings():
                    warnings.filterwarnings("error", category=ODEintWarning)
                    try:
                        y, info = odeint(
                            self.p_controller,
                            self.x0,
                            ts,
                            args=(Tc[i + 1] + self.nx[i + 1],),
                            atol=atol,
                            rtol=rtol,
                            full_output=True,
                        )
                    except ODEintWarning as ex:
                        print(
                            f"ODEINT {i} params=> Xo={self.x0}, Ts={ts},\n"
                            f"T [i] = {T[i]}, Tc[i+1]={T[i + 1]},\n"
                            f"Tc[i] = {Tc[i]}, Tc[i+1]={Tc[i + 1]},\n"
                            f"Ca[i] = {Ca[i]},\n"
                            f"nx[i+1]={self.nx[i + 1]}"
                        )
                        pass
                    except Warning as ode_exp:
                        print("-" * 40)
                        print("ODEintWarning")
                        print(info)
                        print("-"*40)

                Ca[i + 1] = y[-1][0]
                T[i + 1] = y[-1][1]
                T[i + 1] = np.clip(T[i + 1], self.system_min, self.system_max)

                self.x0[0] = Ca[i + 1]
                self.x0[1] = T[i + 1]

            return t, T, Tc

    @staticmethod
    def profile(t: np.ndarray, length: list, temps: list) -> np.ndarray:
        """
        Method generates autoclave profile of from the production cell
        Args:
            t           - time domain for which trace will be generated (0-N)
            length      - array representing durations of autoclave profile phases
                        0 - length[0] - duration of initial flat state
                        length[0] - length [1] duration of heating state
                        length[1] - length[2] duration of autoclave treatment
                        length[2] - length[3] - duration of cooling state
                        length[3] - N - duration of post autoclave treatment state
            temps       - temperatures - Y axis
                        temps[0]  - observed initial room temperature
                        temps[1] - maximum autoclave profile temperature T1
                        temps[2]  - maximum autoclave profile temperature T2
                        temps[3] - observed final room temperature
        Returns:
            <np.ndarray>  - new autoclave profile over time
        """
        # Temperature profile modeling
        Tc = np.ones(len(t))

        # Step cooling temperature
        Tc[0 : length[0]] = temps[0]
        tx = [length[0], length[1]]
        y = [temps[0], temps[1]]
        t1, y1 = interpolate_slope(tx, y)

        Tc[length[0] : length[1] - 1] = y1
        Tc[length[1] - 1 : length[2]] = temps[2]

        tx = [length[2], length[3]]
        y = [temps[2], temps[3]]
        t1, y1 = interpolate_slope(tx, y)

        Tc[length[2] : length[3] - 1] = y1
        Tc[length[3] - 1 : len(t)] = temps[3]

        return Tc

    @staticmethod
    def get(t: np.ndarray, length: list, temps: list) -> np.ndarray:
        """
        Generator that provides next autoclave profile based on changing parameters
        Args:
            t   -   - time X axis
            length  - lengths of each section of the temperature profile
            temps   - list containing temperatures of the profile
        Returns:
            <nd.ndarray>    - array of data containing autoclave profile
        """
        profile = AutoclaveProcess.profile(t, length, temps)
        yield profile

    @staticmethod
    def assign_value(value: object) -> int:
        """
        Method checks value type and assigns value depending on the
        type.
        Args:
            value    - object (static value or list containing with max/min values)
        Returns:
            <int>    - number representing value
        """
        if isinstance(value, list) and len(value) == 2:
            return random.randint(value[0], value[1])
        elif isinstance(value, list) and len(value) == 1:
            return int(value[0])
        elif isinstance(value, int) or isinstance(value, float):
            return int(value)

    @staticmethod
    def read_temp(data: pd.DataFrame, approx_idx: int, tcol: str, vcol: str) -> float:
        """
        Method reads temperature value based on approximated approach
        b_data, i*cycles + duration_set[0], "timestamp", config["environment"]["in_col"]
        Args:
            data        - input data
            approx_idx  - approximate value of the index from where value should be read
            tcol        - time column name
            vcol        - value column name
        Returns:
            <float> - approximated value of the temperature
        """
        temp_value = 0.0
        index_val = data.loc[:, tcol].values.tolist()

        # Check if original index exists
        if approx_idx in index_val:
            temp_value = data.loc[approx_idx, [vcol]][0]
        # Drop an error
        else:
            print(f"Index {approx_idx} doesn't exists")

        return temp_value

    @staticmethod
    def background_temp(cfg: dict, exp_path: str) -> pd.DataFrame:
        """
        Method processes background temperature data that is added to the
        original trace
        Args:
              cfg: ICS process configuration
              exp_path: experiment path
        Returns:
              <pd.DataFrame>
        """
        b_path = cfg["path"]

        if "environment" in cfg.keys() and cfg["environment"]["in_file"] == "":
            return None
        back_input_file = cfg["environment"]["in_file"]
        back_input_file = back_input_file[back_input_file.rfind("/") + 1 :]

        # Read file and select only start and the end amount of temperatures
        file_name = f"{exp_path}{b_path}/background/{back_input_file}"

        b_file_name = (
            f"{exp_path}{b_path}data_{cfg['environment']['in_col'].lower()}2.csv"
        )

        b_data = None

        if not os.path.isfile(file_name):
            return b_data

        # 1 Read background temperature data
        b_data = pd.read_csv(file_name, skiprows=cfg["environment"]["ignore_rows"])
        # TODO: Convert date column to timestamp column
        b_data["timestamp"] = pd.to_datetime(
            b_data["date"], format=cfg["environment"]["date_format"]
        )

        b_data["timestamp"] = (
            b_data["timestamp"] - b_data["timestamp"][0]
        ).dt.total_seconds()

        # Adding multiplier for number of measures per second - convert indices to int!
        b_data["timestamp"] = (b_data["timestamp"] / cfg["samples"]).astype("int32")

        b_data = pd.DataFrame(
            b_data.loc[:, ["timestamp", cfg["environment"]["in_col"]]]
        )
        # bdata = b_data.reset_index()
        # b_data.to_csv(b_file_name, index=False)

        # elif os.path.isfile(b_file_name):
        #     b_data = pd.read_csv(b_file_name)

        return b_data

    @staticmethod
    def duration_set(cfg: dict, temp: list) -> list:
        """
        Method sets duration set for the temperature cycle
        Args:
            cfg: configuration of the ICS process
            temp: list of start and stop temperatures
        Returns:
            <list> - list containing timestamps at which ICS process
             changes are applied (heating, plasma treatment and cooling).
        """
        # autoclave process duration - Std dev. ~10% for
        duration_set = []
        for val in cfg["profile"]["duration"]["set"]:
            duration_set.append(AutoclaveProcess.assign_value(val))

        # Read temperatures for given time
        # Find most corresponding index to fetch data
        read_init_temp = temp[0]
        read_final_temp = temp[1]

        # Adjust duration sets
        # If temperature is lower add more data points to duration set (warming up)
        # If temperature is higher shorten cycle (no need to warm up thus cycle is shorter)
        adjustment = round(read_init_temp - cfg["profile"]["temp"]["room"][0])

        duration_set[0] += adjustment
        # duration_set[1] += adjustment

        adjustment = round(cfg["profile"]["temp"]["room"][0] - read_final_temp)
        # duration_set[2] += adjustment
        duration_set[3] += adjustment

        return duration_set

    @staticmethod
    def get_data(
        config: dict, exp_path: str, ts: datetime, logger: Log, save: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Method returns data based on autoclave process configuration
        Args:
            config:  - configuration information for data generation
            exp_path:- path to the experiment
            ts:      - time stamp
            save:    - save results to disk
        Returns:
            <pd.DataFrame, pd.DataFrame> - first set contains data frame
            containing newly generated data set and second set contains
            parameters generated for data set.
        """
        # ----------------------------------------------------
        #           Configuration of the experiment
        # ----------------------------------------------------
        cycles = config["cycles"] + config["detection_cycles"]
        cycle_len = config["cycle_len"]

        # Number of samples per single unit of measurement (frequency)
        samples = config["samples"]

        c_path = config["path"]

        # ----------------------------------------------------
        #             Training data generation
        # ----------------------------------------------------

        # Data columns that later will be converted to DataFrame and stored
        mTc = []
        mTp = []
        mx = []
        mT = []
        duration_params = []
        cycle = []
        temps = []

        # Time Interval for experiment
        t = np.linspace(0, cycle_len * cycles, samples * cycle_len * cycles)

        # Generate simulation data
        start = datetime.now()

        # Read values from the additional background file and adjust profile
        # durations 0 and 3 (start and stop temperatures)
        b_data = AutoclaveProcess.background_temp(config, exp_path)

        cycle_limit = 0
        if config["environment"]["exclude_test"]:
            cycle_limit = (int)(cycles / 2)

        if b_data is not None and not b_data.empty:
            b_set_low = b_data[: cycle_limit * 2]
            b_set_high = b_data[-cycle_limit * 2 :]

            # duration = b_data.loc[2, "timestamp"] - b_data.loc[1, "timestamp"]

        for i in range(0, cycles):
            # Initial good parameters
            # Generate room temp and autoclave treatment temp profile
            temperatures = [
                AutoclaveProcess.assign_value(config["profile"]["temp"]["room"]),
                AutoclaveProcess.assign_value(
                    config["profile"]["temp"]["optimal_heating"]
                ),
                AutoclaveProcess.assign_value(
                    config["profile"]["temp"]["optimal_heating"]
                ),
                AutoclaveProcess.assign_value(config["profile"]["temp"]["room"]),
            ]

            # TODO: In case of background data active adjust control profile
            #  with real measured temperatures
            if b_data is not None and not b_data.empty:
                if i >= cycle_limit:
                    b_set = b_set_high
                    temperatures[0] = b_set.loc[
                        b_set.index[((i - cycle_limit) * 2)], "temp"
                    ]
                    temperatures[3] = b_set.loc[
                        b_set.index[((i - cycle_limit) * 2 + 1)], "temp"
                    ]

                else:
                    b_set = b_set_low
                    temperatures[0] = b_set.loc[b_set.index[(i * 2)], "temp"]
                    temperatures[3] = b_set.loc[b_set.index[(i * 2 + 1)], "temp"]

            # Read temperatures for given time
            # Find most corresponding index to fetch data
            duration_set = AutoclaveProcess.duration_set(
                config, [temperatures[0], temperatures[3]]
            )

            # print(f"Temp profile [{i}]: |{temperatures}| => |{duration_set}| => {c_set.shape}")

            tt = t[: int(len(t) / cycles)]

            Tp = AutoclaveProcess.profile(tt, duration_set, temperatures)

            pid = AutoclaveProcess.PIDController(samples * cycle_len, logger)
            x, T, Tc = pid.run(tt, np.around(Tp, 2))

            # Clear memory before next round
            pid = None
            gc.collect()

            # Normalize T start as it goes un-naturally above the temperature
            # This part smooths non-linear spikes
            nlen = 29

            T[:nlen] = np.random.random_sample(nlen) / 4 + Tc[0]

            # Apply lower precision to simplify numerics processing and avoid double precision
            # number to save memory and CPU time to process (in case of coarse temperature process this
            # is within the control limits of a process).
            mT.extend(np.around(T, 2))
            mx.extend(x)
            mTp.extend(np.around(Tp, 2))
            mTc.extend(np.around(Tc, 2))

            duration_params.append(duration_set)
            temps.append(temperatures)
            cycle.append(i)

            if i % 100 == 0:
                logger.debug(
                    f"Completed {i} cycle => {int((i / cycles) * 100)}%", end="\r"
                )

        stop = datetime.now()
        # logger.debug(f"Matrix size mTc[{len(mTc)}] <=> mTp[{len(mTp)}]")
        logger.debug(
            f"Avg generation time {((stop - start).total_seconds() / cycles):.2f} seconds"
        )

        # Save generated data
        data = pd.DataFrame(
            {"time": t, "temp": mT, "tc": mTc}, columns=["time", "temp", "tc"]
        )
        # data.set_index("time", inplace=True)

        params = pd.DataFrame(
            {"cycle": cycle, "params": duration_params, "temps": temps},
            columns=["cycle", "params", "temps"],
        )
        params.set_index("cycle", inplace=True)

        if save:
            params.to_csv(f"{c_path}{ts}_p.csv")
            data.to_csv(f"{c_path}{ts}_d.csv")

        return params, data

    @staticmethod
    def gen_data(cfg: dict, logger: Log) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Method generates data
        Args:
            cfg - process configuration
        Returns:
            <pd.DataFrame, pd.DataFrame>    - tuple parameters
        """
        ts = datetime.strftime(datetime.now(), "%Y_%m_%d_%H-%M-%S")
        cfg["process"].update({"environment": cfg["environment"]})
        params, data = AutoclaveProcess.get_data(
            cfg["process"],
            cfg["experiment"]["path"],
            ts,
            logger,
            False,
        )

        return params, data
