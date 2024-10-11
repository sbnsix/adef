import pandas as pd
import shutil
from tools.plotter_helper import Plotter

from log import Log


class CompAnomalyReport:
    """
    Class implementing computational anomalies observed during simulation
    """

    def __init__(self, logger: Log):
        self.log = logger
        self.rep_path = "../exp/outliers/"

    def report_roc(self, report_label: str, data: pd.DataFrame, file_name: str) -> None:
        """
        Method dumps data and logs ROC anomaly output via logger object
        Args:
            report_label: report label used in the anomaly report
            data: DataFrame object containing anomaly
            file_name: file name where anomaly data will be saved (CSV format)
        Returns:
            <None>
        """
        data.to_csv(f"{file_name[:-4]}_roc.csv")
        f_name = f"{file_name[:-4]}"
        f_name = f_name[f_name.rindex("/") + 1 :]
        Plotter.roc_2d(
            data,
            {"auc": 0.0},
            "STEP 5 CD Bad ROC",
            f"../exp/outliers/{f_name}_roc_cd.png",
            0.04,
            "search_param" in data[data["cfg"] != 0.0]["cfg"].keys()
        )
        self.log.error(f"Anomaly data saved in {self.rep_path}{f_name}_roc_cd.csv")
        shutil.copy(f"{file_name[:-4]}_roc.csv", f"{self.rep_path}{f_name}_roc_cd.csv")
