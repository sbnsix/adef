""" Power point experiment presentation module"""


import pandas as pd
from ppt_gen import PptGenerator


class PresentationBase:
    """
    Class describing experiment presentation
    """

    def __init__(self, cfg: dict, logger: object) -> None:
        """
        CTOR
        Args:
            logger: logger object
        Returns:
             <None>
        """
        self.log = logger
        self.cfg = cfg
        self.prs = PptGenerator(cfg, logger)
        # Data set ordering list used in documents generation
        self.detector_names = {
            x: y for (x, y) in self.cfg["model"]["ad"].items() if y["enabled"]
        }.keys()
        self.algo_name = ""
        self.alg_sh = ""
        self.prob = "50%"
        self.path = f"{cfg['global']['path']}experiments/{cfg['experiment']['path']}"

    @staticmethod
    def extract_figure_info(files: list) -> dict:
        """
        Methods extracts figure information
        Args:
            files: list of files containing images
        Returns:
            <dict> - containing image information
        """
        result = {}
        for file in files:
            key = file.split("_")[3]
            result[key] = file
        return result

    @staticmethod
    def extract_figure_infos(files: list) -> dict:
        """
        Methods extracts figure information
        Args:
            files: list of files containing images
        Returns:
            <dict> - containing image information
        """
        result = {}
        attacks_to_display = []
        for file in files:
            file_n = file[file.rfind("\\") + 1 :]
            key = file_n.split("_")[2]

            if key in attacks_to_display:
                continue
            attacks_to_display.append(key)

            if key not in result.keys():
                result[key] = [file]
            else:
                result[key].append(file)

        # if not mixed:

        result = {x: v for x, v in result.items() if "mixed" not in x.lower()}

        return result

    def fill_info_data(self, ad_data: dict, cd_data: dict) -> dict:
        """
        Method extracts AD and CD models test data into single dictionary
        Args:
            ad_data:
            cd_data:
        Returns:
            <dict>: Reformatted dictionary for presentation generation
        """

        def set_field(k: str, val: object) -> str:
            fmt_size = 0
            if val is None or k is None or k == "":
                return "-"

            if "d" not in key:
                fmt_size = 2
            if "d" in key:
                fmt_size = 0

            if isinstance(val, float) and k.lower() not in ["auc", "acc"]:
                if str(val) == "nan":
                    return "-"
                return f"{val:.{fmt_size}f}"

            if isinstance(val, float) and k.lower() in ["auc", "acc"]:
                return f"{(val*100):.{fmt_size}f}"

            if isinstance(val, list):
                lst = ["-" if str(v) == "nan" else int(v) for v in val]
                return f"{lst}"

            if isinstance(val, pd.Series):
                lst = [f"{v:.{fmt_size}f}" for v in val.tolist()]
                return f"{lst}"

            if isinstance(val, pd.DataFrame):
                fmt_min = 0.0
                fmt_max = 0.0
                try:
                    self.log.debug(f"{k}=>{val.columns}")
                    self.log.debug(f"{k}=>{val.columns}")
                except Exception as ex:
                    self.log.error(f"Incorrect result on: |{k}|=|{val}|")
                    self.log.exception(ex)

                return f"[{fmt_min}, {fmt_max}]"
            return "-"

        # Prepare tab data
        tab_data = {}

        # Iterate over AD data
        if self.alg_sh in ad_data.keys():
            for ad_info in ad_data[self.alg_sh].values():
                for key, val in ad_info.items():
                    if key not in tab_data.keys():
                        tab_data[key] = {}

                    tab_data[key]["ad"] = set_field(key, val)

        # Iterate over CD data
        if self.alg_sh in cd_data.keys():
            for cd_info in cd_data[self.alg_sh].values():
                for key, val in cd_info.items():
                    if key not in tab_data.keys():
                        tab_data[key] = {}
                    if "cd" not in tab_data[key].keys():
                        tab_data[key]["cd"] = ""
                    tab_data[key]["cd"] = set_field(key, val)

        return tab_data
