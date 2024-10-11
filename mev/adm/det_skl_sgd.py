""" AD model based on Support Vector Machines algorithm. """


from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from sklearn.exceptions import NotFittedError

import tools.log as log
from adm_generic import Det_Generic

import matplotlib.pyplot as plt


class Det_SKL_SGD(Det_Generic):
    """
    Stochastic Gradient Decent anomaly detector algorithm implementation
    """

    def __init__(self, logger: log.Log) -> Det_SKL_SGD:
        """
        CTOR
        Args:
            logger  - logger object
        Returns:
            <Det_SKL_SGD>  - instance of the class
        """
        super().__init__("sgd", logger)

    def create(self,
               cfg: dict) -> object:
        """
        Method create AD model with given configuration.
        This model is not trained and requires input data to
        become fully functional AD model.
        Args:
            cfg: AD model configuration
        Returns:
            <object> - AD model
        """
        return SGDClassifier(
            loss=cfg["loss"],
            penalty=cfg["penalty"],
            max_iter=cfg["max_iter"],
            tol=cfg["tol"],
        )

    def input_scaler(self,
                     data: pd.Series) -> np.array:
        x_scaler = preprocessing.MinMaxScaler()
        x = abs(data).values.tolist()
        x1 = x_scaler.fit_transform(x)[:, 1].reshape(-1, 1)
        # x_scaler = preprocessing.StandardScaler()
        # x1 = x_scaler.fit_transform(x)
        # x_scaler1 = preprocessing.MaxAbsScaler()
        # x_scaler1.fit_transform(x1)
        return x1

    def output_scaler(self,
                      data: pd.Series,
                      data_size: int) -> np.array:
        y_scaler = preprocessing.MinMaxScaler()
        y1 = y_scaler.fit_transform(data).reshape(-1, 1).reshape((data_size,))
        y_scaler1 = preprocessing.MaxAbsScaler()
        return y_scaler1.fit_transform(y1)

    def adef_train(self,
                   t_data: pd.DataFrame,
                   cfg: dict,
                   model: object,
                   debug: bool = False) -> pd.DataFrame:
        """
        Method trains SGD model
        Args:
            t_data: attack data from where model will be created
            cfg: AD model configuration
            model: model object used in training
            debug: prints debugging information
        Returns:
            <DataFrame> - DataFrame containing SVM outliers
        """
        # x = self.input_scaler(t_data.loc[:, ["time", "temp"]])

        x = abs(t_data.loc[:, ["temp"]]).values.tolist()

        model.fit(x, t_data.loc[:, cfg["label"]])

        # Print classification report
        outliers_df = model.decision_function(x)

        # Remap results back to the matching binary class classifier output
        # outliers_df = self.output_scaler(y, t_data.shape[0])

        # Apply threshold based configuration
        outliers_df = (outliers_df > cfg["threshold"]).astype(int)

        # Debug display
        if debug:
            self.log.debug(f"SGD best params:\n{model.get_params()}")
            self.log.debug(
                f"\n{classification_report(t_data.loc[:, cfg['label']], outliers_df)}"
            )

        ds = self.format_output(outliers_df, t_data)

        return ds

    def adef_detect(self,
                    a_data: pd.DataFrame,
                    cfg: dict,
                    model: object) -> pd.DataFrame:
        """
        Method performs SGD model outlier classification
        Args:
            a_data: attack data from where detections will be predicted
            cfg: AD model configuration
            model: model object used in detection process
        Returns:
            <pd.DataFrame> - DataFrame containing SGD outliers
        """
        # This is still questionable as the model should work
        # without this step
        # x = self.input_scaler(a_data.loc[:, ["time", "temp"]])

        x = abs(a_data.loc[:, ["temp"]]).values.tolist()

        outliers_df = model.decision_function(x)

        # Remap results back to the matching binary class classifier output
        # outliers_df = self.output_scaler(y, a_data.shape[0])

        # Apply threshold based configuration
        outliers_df = (outliers_df > cfg["threshold"]).astype(int)

        ds = self.format_output(outliers_df, a_data)

        return ds
