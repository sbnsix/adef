""" Anomaly detection threshold based model. """


from __future__ import annotations

import pandas as pd

import tools.log as log
from adm_generic import Det_Generic


class Det_THB(Det_Generic):
    """
    Threshold based detector algorithm implementation
    """

    def __init__(self, logger: log.Log) -> Det_THB:
        """
        CTOR
        Args:
            logger  - logger object
        Returns:
            <Det_Thb>  - instance of the class
        """
        super().__init__("thb", logger)

    def __format_output(self,
                        outliers: pd.DataFrame,
                        data: pd.DataFrame
    ) -> pd.DataFrame:
        # Mark only values above threshold
        outliers["result"] = 0
        outliers.loc[data.index, "result"] = 1

        data["result"] = outliers.loc[:, "result"]

        data.set_index("time", inplace=True)

        return data

    def create(self, cfg: dict) -> object:
        """
        Method create AD model with given configuration.
        This model is not trained and requires input data to
        become fully functional AD model.
        Args:
            cfg: AD model configuration
        Returns:
            <object> - AD model
        """
        return None

    def adef_train(self,
              t_data: pd.DataFrame,
              cfg: dict,
              model: object) -> pd.DataFrame:
        """
        Method trains Thb model
        Args:
            t_data: attack data from where model will be created
            cfg: configuration for the AD model
            model: model object used in detection process
        Returns:
            <DataFrame> - DataFrame containing SVM outliers
        """
        # x = self.x_scaler(t_data.loc[:, ["temp"]])
        x = abs(t_data.loc[:, ["temp"]])
        outliers_df = (x > cfg["threshold"]).astype(int)

        ds = self.format_output(outliers_df,
                                t_data)

        return ds

    def adef_detect(self,
               a_data: pd.DataFrame,
               cfg: dict,
               model: object
    ) -> pd.DataFrame:
        """
        Method performs SVM model outlier classification
        Args:
            a_data: attack data from where detections will be predicted
            cfg: AD model configuration
            model: AD model object
        Returns:
            <pd.DataFrame> - DataFrame containing SVM outliers
        """
        # x = self.x_scaler(a_data.loc[:, ["temp"]])
        x = abs(a_data.loc[:, ["temp"]])
        outliers_df = (x > cfg["threshold"]).astype(int)

        ds = self.format_output(outliers_df, a_data)

        return ds
