
""" Generic AD model implementation """


from __future__ import annotations

import os
import pickle
import inspect
import re

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn import preprocessing

import tools.log as log
import importlib
import warnings


class Det_Generic:
    """
    Generic anomaly detection class implementation
    """

    def __init__(self,
                 class_name: str,
                 logger: log.Log) -> Det_Generic:
        """
        CTOR
        Args:
            class_name: name of AD model class
            logger: logger object
        Returns:
            <Det_Generic>  - instance of the class
        """
        self.class_name = class_name
        self.log = logger

    def format_output(
        self,
        outliers: np.ndarray,
        ds: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Method extends data set with new result column containing
        binary class classifier result from anomaly detection process.
        Args:
            outliers: list of binary outliers (0 - normal point, 1 - anomaly)
            ds: current data set
        Returns:
            <pd.DataFrame>: updated data frame
        """
        ds["result"] = outliers
        # if "time" in ds.columns:
        #    ds.set_index("time", inplace=True)

        return ds

    def x_scaler(self,
                 data: pd.Series) -> np.array:
        """
        Args:
            data:
        Return:
            <np.array>
        """
        x = np.array(data.values.tolist())
        x_scaler = preprocessing.MaxAbsScaler()
        return x_scaler.fit_transform(x).reshape(-1, 1)

    def y_scaler(self,
                 data: np.ndarray,
                 dataset_size: int) -> np.array:
        """
        Args:
            data:
        Return:
            <np.array>
        """
        y_scaler = preprocessing.MaxAbsScaler()
        y1 = y_scaler.fit_transform(data.reshape(-1, 1))
        y_scaler1 = preprocessing.MinMaxScaler()
        return y_scaler1.fit_transform(y1).reshape(dataset_size)

    def adef_train(self,
              t_data: pd.DataFrame,
              cfg: dict,
              model: object,
              debug: bool = False) -> pd.DataFrame:
        """
        Method implements generic train method for compatible models
        Args:
            t_data: attack data from where model will be created
            model: AD model object
            cfg: AD model configuration
            debug       - prints debugging information
        Returns:
            <DataFrame> - DataFrame containing SVM outliers
        """
        x = t_data.loc[:, ["time", "temp"]]

        functions = self.resolve_function_names(model, ["fit", "predict", "fit_predict"])

        # Fitting the model for grid search
        try:
            model.fit(x, t_data.loc[:, cfg["label"]])
            if debug:
                self.log.debug(f"SVC best params:\n{model.best_params_}")

            # Print classification report
            outliers_df = functions["predict"][0](x)

            if debug:
                self.log.debug(
                    f"\n{classification_report(t_data.loc[:, cfg['label']], outliers_df)}"
                )
        except ValueError:
            outliers_df = np.zeros(t_data.shape[0])

        ds = self.__format_output(outliers_df, t_data)

        return ds

    def adef_detect(self,
               a_data: pd.DataFrame,
               cfg: dict,
               model: object) -> pd.DataFrame:
        """
        Method performs SVM model outlier classification
        Args:
            a_data      - attack data from where detections will be predicted
            model: AD model object
            cfg: AD model configuration
        Returns:
            <pd.DataFrame> - DataFrame containing SVM outliers
        """
        x = a_data.loc[:, ["time", "temp"]]

        try:
            functions = self.resolve_function_names(model, ["fit", "predict", "fit_predict"])
            outliers_df = functions["predict"][0](x)
        except ValueError:
            outliers_df = np.zeros(a_data.shape[0])
        except Exception as ex:
            self.log.exception(ex)

        ds = self.__format_output(outliers_df, a_data)

        return ds
