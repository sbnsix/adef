""" ICS autoclave profile detector  for AD model using DBSCAN algorithm wrapper. """


from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn import preprocessing

import tools.log as log
from adm_generic import Det_Generic


class Det_SKL_DBSCAN(Det_Generic):
    """
    DBCAN algorithm detector implementation
    """

    def __init__(self, logger: log.Log) -> Det_SKL_DBSCAN:
        """
        CTOR
        Args:
            logger  - logger object
        Returns:
            <Det_SKL_DBSCAN>  - instance of the class
        """
        super().__init__("dbscan", logger)


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
        return DBSCAN(eps=cfg["eps"], min_samples=int(cfg["samples"]))

    def x_scaler(self,
                 data: pd.Series) -> np.array:
        """

        Args:
            data:
        Returns:
            <np.array>
        """
        x = np.array(abs(data.values).tolist())
        x_scaler = preprocessing.MinMaxScaler()

        return x_scaler.fit_transform(x)[:, 1].reshape(-1, 1)

    def adef_train(self,
                   t_data: pd.DataFrame,
                   cfg: dict,
                   model: object,
                   debug: bool = False) -> pd.DataFrame:
        """
        Method trains DBSCAN model
        Args:
            t_data: attack data from which model will be trained
            cfg: DBSCAN algorithm configuration
            model: AD model object
            debug: debug flag to output additional information
        Returns:
            <DataFrame>: DataFrame containing DBSCAN outliers
        """
        # x = self.x_scaler(t_data.loc[:, ["time", "temp"]])
        x = abs(t_data.loc[:, ["temp"]].values).tolist()

        # Fit and predict model to data - the data labels are stored in the model
        cluster_labels = model.fit_predict(x, t_data.loc[:, "label"])

        # Identify outliers (labels == -1)
        outliers_df = (cluster_labels == -1).astype(int)

        # Format output data
        ds = self.format_output(outliers_df, t_data)

        return ds

    def adef_detect(self,
                    a_data: pd.DataFrame,
                    cfg: dict,
                    model: object) -> pd.DataFrame:
        """
        Method trains DBSCAN model
        Args:
            a_data: attack data from where detections will be predicted
            cfg: DBSCAN algorithm configuration
            model: AD model object
        Returns:
            <DataFrame> - DataFrame containing DBSCAN outliers
        """
        # x = self.x_scaler(a_data.loc[:, ["time", "temp"]])
        x = abs(a_data.loc[:, ["temp"]].values).tolist()

        # Fit and predict model to data - the data labels are stored in the model
        cluster_labels = model.fit_predict(x, a_data.loc[:, "label"])

        # Identify outliers (labels == -1)
        outliers_df = (cluster_labels == -1).astype(int)

        # Format output data
        ds = self.format_output(outliers_df, a_data)

        return ds
