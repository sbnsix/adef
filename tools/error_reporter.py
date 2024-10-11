from __future__ import annotations
import os
import shutil
import pandas as pd
import copy


from plotter_helper import Plotter


class ErrorReporter:
    """
    Class describing series of the error reporting routines
    used in ADEF framework
    """

    @staticmethod
    def auc_report(
        results: pd.DataFrame,
        f_name: str,
        fpr_label: str,
        tpr_label: str,
        auc_val: float,
        limit: float,
        log2f: object,
    ) -> None:
        """
        Method reporting incorrect AUC graph generated from input model data
        Args:
            results: data frame containing FPR/TPR information
            f_name: file name
            fpr_label: False Positive Rate label for pd.DataFrame
            tpr_label: True Positive Rate label for pd.DataFrame
            auc_val: AUC value
            limit: AUC limit for industrial ICS
            log2f: logger object
        Returns:
            <None>
        """
        roc_out_file = copy.copy(f_name)
        f_name = f_name[f_name.rindex("/") + 1 :]
        log2f.error(f"Incorrect data saved in ../exp/outliers/{f_name}_step_4.csv")
        if results is not None:
            Plotter.roc_2d(
                results.loc[:, fpr_label],
                results.loc[:, tpr_label],
                {"auc": auc_val},
                "STEP 4 AD Bad ROC",
                f"../exp/outliers/{f_name}_step_4.png",
                limit / 100,
                "search_param" in results[results["cfg"] != 0.0]["cfg"].keys()
            )
        if os.path.isfile(f"{roc_out_file}.csv"):
            shutil.copyfile(
                f"{roc_out_file}.csv", f"../exp/outliers/{f_name}_step_4.csv"
            )
