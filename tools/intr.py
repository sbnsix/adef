""" Shell command interpreter to enable experiment layout construction """


import os
import shutil
import glob
import yaml_converter as yc


class Interpreter:
    """
    Class defining OS level interpreter commands used in
    experiment structure preparation.
    """

    @staticmethod
    def run(cmds_list: list, log: object = None) -> None:
        """
        Method check and runs meta level simulator commands
        Args:
            cmds_list: list of commands to be executed
            log: logger object
        Returns:

        """
        for cmd in cmds_list:
            # cmds = cmd.split(" ")
            # self.log.debug(f"Processing cmd: {cmd}")

            # Detect task type
            """
            # Process configuration file - run experiment
            if os.path.isfile(cmd) and not cmd.endswith(".py"):
                # self.log.debug(f"CFG: {json.dumps(local_exp_cfg, indent=4)}")
                log.debug(f"Running experiment: {cmd}")

                local_exp_cfg = ConfigHelper.load(cmd)
                self.main_path = local_exp_cfg["experiment"]["path"]
                attack_lists = self.generate_attack_lists()
                # TODO: Generate attacks here
                #  input for se element will be data file, attack file, mirrored file

                if attack_lists:
                    for attack_list in attack_lists:
                        local_exp_cfg["attack"]["attacks_per_cycle"] = attack_list
                        log.debug(f"Attack list set is for experiment: {attack_list}")
                        # log.debug(f"CFG: |{json.dumps(local_exp_cfg, indent=4)}|")
                        self.run_se(local_exp_cfg)
                else:
                    log.debug(f"DEFAULT Attack list is set for experiment: "
                                   f"{local_exp_cfg['attack']['attacks_per_cycle']}")
                    self.run_se(local_exp_cfg)
                    # log.debug(f"CFG: |{json.dumps(local_exp_cfg, indent=4)}|")

            # Running Python script
            elif cmds[0].endswith(".py") and os.path.isfile(cmds[0]):
                try:
                    subprocess.run(cmds.split(" "), check=False)
                except Exception as ex:
                    log.exception()
            """
            # File(s) copy
            if cmd.startswith("cp"):
                # Filtering and cleaning
                cmd = cmd.replace(" \\", "")
                args = cmd.split(" ")[1:]
                file_mask = args[0]
                dsts = []
                for l_path in args[1:]:
                    c_path = l_path.strip()

                    # Complex path that will be extracted from config file
                    if ":" in c_path:
                        els = c_path.split(":")
                        real_path = ""
                        # Extract real path
                        if os.path.isfile(els[0]):
                            cfg_tmp = yc.toJson(els[0])
                            real_path = cfg_tmp[els[1]][els[2]]
                        dsts.append(real_path.replace("//", "/"))

                    # Direct file copy
                    elif os.path.isdir(c_path):
                        if len(dsts) == 0:
                            if "*" in args[0]:
                                dsts.append(args[0][: args[0].rindex("/") + 1])
                                file_mask = args[0][args[0].rindex("/") + 1 :]
                            else:
                                dsts.append(args[0])
                                file_mask = ""
                        dsts.append(c_path.replace("//", "/"))

                    # Unknown case for further development
                    else:
                        if log is not None:
                            log.error(f"Unknown command {c_path}")

                # Perform copy operation
                for dst in dsts[1:]:
                    if file_mask:
                        copy_path = os.path.join(dsts[0], file_mask).replace("//", "/")
                        search_result = glob.glob(copy_path)
                        # self.log.debug(f"Processing: {copy_path}")
                        # self.log.debug(f"Search result: {str(glob.glob(copy_path))}")
                        if not search_result:
                            if log is not None:
                                log.error("Unable to find files for copy operation")
                                log.error(f"Line {cmd}")
                            continue

                        for filename in glob.glob(copy_path):
                            if os.path.isdir(dst) and os.path.isfile(filename):
                                shutil.copy(filename, dst)
                                if log is not None:
                                    log.debug(f"Coping file {filename} to folder {dst}")
                            elif os.path.isfile(filename):
                                if log is not None:
                                    log.error(f"File {filename} doesn't exists")
                            elif os.path.isdir(dst):
                                if log is not None:
                                    log.error(f"Directory {dst} doesn't exists")
                    else:
                        if os.path.isdir(dst) and os.path.isfile(dsts[0]):
                            shutil.copy(dsts[0], dst)
                            if log is not None:
                                log.debug(f"Coping file {dsts[0]} to folder {dst}")

            # File(s) move
            elif cmd.startswith("mv"):
                # First parameter file mask
                # Second parameter value to search in file name
                # Third parameter value to be put into file name
                args = cmd.split(" ")[1:]
                # self.log.debug(f"Move args: {str(args)}")

                for file_name in glob.glob(args[0]):
                    new_file_name = file_name.replace(args[1], args[2])
                    if os.path.isfile(new_file_name):
                        os.remove(file_name)
                    else:
                        os.rename(file_name, new_file_name)

            # File(s) removal
            elif cmd.startswith("rm"):
                args = cmd.split(" ")[1:]
                # self.log.debug(f"Delete args: {str(args)}")

                for file_name in glob.glob(args[0]):
                    if os.path.isfile(file_name):
                        os.remove(file_name)

            # Directories creation
            elif cmd.startswith("md"):
                args = cmd.split(" ")
                for arg in args[1:]:
                    if not os.path.isdir(arg):
                        if log is not None:
                            log.debug(f"Creating dir {arg}")
                        os.makedirs(arg, exist_ok=True)

            # Directories deletion
            elif cmd.startswith("rd"):
                args = cmd.split(" ")
                for arg in args[1:]:
                    if os.path.isdir(arg):
                        if log is not None:
                            log.debug(f"Removing dir {arg}")
                        shutil.rmtree(arg)

            # Unknown command
            else:
                if log is not None:
                    log.error(f"Unrecognized command |{cmd}|")
                continue
