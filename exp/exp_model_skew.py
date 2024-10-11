""" Configuration builder for model skew experiment """

import random
import glob
import pandas as pd

import copy
from sklearn.utils import shuffle

from base_exp import BaseExperimentConfig
from tools.intr import Interpreter

from mev.data.attack_scenarios import DataGeneration


class ExpModelSkew(BaseExperimentConfig):
    """
    Model skew experiment configuration class
    """
    def shuffle(self,
                in_data: pd.DataFrame,
                cycle_len: int) -> pd.DataFrame:
        """
        Method shuffles cycles inside data frame.
        Args:
            in_data: input data required to be shuffled
            cycle_len: length of the production cycle measured in samples
        Returns:
            <pd.DataFrame>:
        """
        # Create a list of DataFrames, each containing 300 rows
        groups = [in_data[i:i + cycle_len] for i in range(0, in_data.shape[0], cycle_len)]

        # Shuffle the list of DataFrames
        shuffled_groups = shuffle(groups, random_state=random.randint(0, 100))

        # Concatenate the shuffled groups back into a single DataFrame
        output = pd.concat(shuffled_groups).reset_index(drop=True)

        return output

    def get(self, *args) -> list:
        """
        Method generates model skew experiment configuration where:
        1. Training is using original data set,
        2. Evaluation set is malformed by the environment data set
        Args:
            <None>
        Returns:
            <list> - list of generated configurations of atomic experiment
        """
        local_exp_cfg = copy.copy(self.d_cfg)
        local_exp_cfg["pre"] = self.cfg["pre"]
        local_exp_cfg["mode"] = True
        exp_path = f"{self.d_cfg['global']['path']}experiments/{self.d_cfg['experiment']['path']}"

        # This is done to avoid deletion of the files
        checkpoint_files = glob.glob(
            f"{exp_path}checkpoint/*.pkl"
        )
        cmds = []
        if not checkpoint_files:
            cmds = [
                f"rm {exp_path}/attack/*",
                f"rm {exp_path}/input/*",
            ]

        local_exp_cfg["environment"]["in_col"] = "temp"
        local_exp_cfg["environment"]["exclude_test"] = True
        local_exp_cfg['environment']['in_file'] = self.d_cfg['experiment']['pre']['trend_input']

        cmds += [
            f"md {exp_path}/summary",
            f"md {exp_path}/input/background",
            f"cp {local_exp_cfg['global']['path']}/attack/*.csv {exp_path}/attack/",
            f"cp {local_exp_cfg['environment']['in_file']} {exp_path}/input/background",
        ]

        Interpreter.run(cmds)

        # Modify process to include background input data
        cycle_len = (
            local_exp_cfg["process"]["cycle_len"] * local_exp_cfg["process"]["samples"]
        )
        cycle_tr_limit = local_exp_cfg["global"]["detection"]["limit"]
        training_limit = cycle_tr_limit * cycle_len

        # Enforce PID control process to run file generation to model_skew folder
        local_exp_cfg["ntr"] = True
        old_process_path = local_exp_cfg['process']['path']
        old_exp_path = local_exp_cfg['experiment']['path']
        local_exp_cfg['process']['path'] = f"experiments/model_skew/input/"
        local_exp_cfg['experiment']['path'] = f"{self.d_cfg['global']['path']}"
        # ["attack"]["file"]
        local_exp_cfg['process']["cycles"] = int(local_exp_cfg['process']["cycles"] / 2)
        local_exp_cfg['process']["detection_cycles"] = int(local_exp_cfg['process']["detection_cycles"] / 2)

        # STEP 1 - regenerate input data - half simulation time
        DataGeneration.step_1(local_exp_cfg, self.log)

        # Restore normal cycle length
        local_exp_cfg['process']["cycles"] *= 2
        local_exp_cfg['process']["detection_cycles"] *= 2

        # Merge the output data - 50% of the trace with original test data into data set that will be run as
        # Normal attack trace. The output is written in the experiment/model_skew/attack folder.

        # Combine environment influenced input (evaluation phase) with half of the original trace (training)
        for file_name in glob.glob(f"{local_exp_cfg['global']['path']}/input/*.csv"):

            train_data = pd.read_csv(file_name)
            time_set = train_data.loc[:, "time"].copy(deep=True)
            train_data = train_data.loc[:int(train_data.index.max()/2), ]
            f_name = file_name[file_name.rfind('\\')+1:]
            eval_file = f"{exp_path}/input/{f_name}"
            eval_data = pd.read_csv(eval_file)

            if self.cfg["shuffle"]:
                eval_data = self.shuffle(eval_data, cycle_len)

            data = pd.concat([train_data, eval_data])
            data.reset_index(inplace=True)
            data.drop("time", axis=1, inplace=True)
            data.drop("index", axis=1, inplace=True)
            data.insert(loc=0, column="time", value=time_set)
            data.to_csv(eval_file, index=False)

        local_exp_cfg['experiment']['path'] = f"{self.d_cfg['global']['path']}/experiments/{old_exp_path}"

        # STEP 2 - inject only last 50% of the trace and enable
        # only evaluation part modification
        DataGeneration.step_2(local_exp_cfg, self.log)

        # TODO: Generate all attack images from the traces supplied in attack folder (this is not generated)
        '''
        Plotter.input_ts_color_split(
            data,
            local_exp_cfg["global"]["detection"]["limit"],
            f"Model skew diff - input",
            local_exp_cfg,
            f"{file_name[:-4]}_oa.png",
            False,
        )

        Plotter.input_ts_color(
            data.loc[i: i + cycle, ],
            a_data.loc[i: i + cycle, ],
            f"Attack {config['name'].upper()}",
            cfg,
            f"{cfg['attack']['file'][:-4]}_sam.png",
            False,
        )
        '''

        # Restore original paths so other steps of ADEF simulator can work correctly.
        local_exp_cfg['process']['path'] = old_process_path
        local_exp_cfg['experiment']['path'] = old_exp_path

        return [local_exp_cfg]

    def post(self, *args) -> None:
        """
        This method generates summary output in form of Power-Point and LaTeX
        for the final experimental results for model skew experiment.
        Args:
            *args:
        Returns:
            <None>
        """
        # Here summary is not required as this experiment runs only single instance of the experiment.
        # This function is a placeholder just in case if in the future experiment requirements would change.
        pass
