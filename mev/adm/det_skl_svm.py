""" ICS autoclave profile detector for AD model based on Support Vector Machines algorithm. """

from __future__ import annotations

from sklearn import preprocessing
import numpy as np
import pandas as pd
from sklearn import svm
# from sklearn.svm import OneClassSVM

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

import tools.log as log
from adm_generic import Det_Generic


class Det_SKL_SVM(Det_Generic):
    """
    SVM detector algorithm implementation
    # TODO: Review this: https://medium.com/learningdatascience/anomaly-detection-techniques-in-python-50f650c75aaf
    """

    def __init__(self, logger: log.Log) -> Det_SKL_SVM:
        """
        CTOR
        Args:
            logger  - logger object
        Returns:
            <Det_SKL_SVM>  - instance of the class
        """
        super().__init__("svm", logger)

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
        return svm.OneClassSVM(
            kernel=cfg["kernel"],
            degree=cfg["degree"],
            gamma=cfg["gamma"],
            nu=cfg["nu"])

    def input_scaler(self,
                     x: pd.Series) -> np.array:
        x_scaler = preprocessing.MinMaxScaler()
        x1 = x_scaler.fit_transform(np.array(x.values.tolist()).reshape(-1, 1))
        x_scaler1 = preprocessing.MaxAbsScaler()
        return x_scaler1.fit_transform(x1)

    def adef_train(self,
                   t_data: pd.DataFrame,
                   cfg: dict,
                   model: object,
                   debug: bool = False,
    ) -> pd.DataFrame:
        """
        Method trains SVM model
        Args:
            t_data: attack data from where model will be created
            cfg: AD model configuration
            model: AD model object
            debug: flat to print debugging information
        Returns:
            <DataFrame> - DataFrame containing SVM outliers
        """
        x = self.input_scaler(abs(t_data.loc[:, ["temp"]]))
        #  When all 0s are presented in the label column
        #  (single class instead of 2 classes) - when data is attacked
        #  the SVC model fit will not work

        # Fitting the model for grid search
        model.fit(x, t_data.loc[:, cfg["label"]])

        if debug:
            self.log.debug(f"SVC best params:\n{model.best_params_}")

        y = model.decision_function(x)

        # Print classification report
        if debug:
            self.log.debug(
                f"\n{classification_report(t_data.loc[:, cfg['label']], y)}"
            )

        # Remap results back to the matching binary class classifier output
        y_scaler = preprocessing.MinMaxScaler()
        outliers_df = y_scaler.fit_transform(y.reshape(-1, 1)).reshape((t_data.shape[0],))

        # Apply threshold based configuration
        outliers_df = (outliers_df > cfg["threshold"]).astype(int)

        ds = self.format_output(outliers_df, t_data)

        return ds

    def adef_detect(self,
                    a_data: pd.DataFrame,
                    cfg: dict,
                    model: object) -> pd.DataFrame:
        """
        Method performs SVM model outlier classification
        Args:
            a_data      - attack data from where detections will be predicted
            cfg: AD model configuration
            model: AD model object
        Returns:
            <pd.DataFrame> - DataFrame containing SVM outliers
        """
        x = self.input_scaler(abs(a_data.loc[:, ["temp"]]))

        y = model.decision_function(x)

        # Remap results back to the matching binary class classifier output
        y_scaler = preprocessing.MinMaxScaler()
        outliers_df = y_scaler.fit_transform(y.reshape(-1, 1)).reshape((a_data.shape[0],))

        # Apply threshold based configuration
        outliers_df = (outliers_df > cfg["threshold"]).astype(int)

        ds = self.format_output(outliers_df, a_data)

        return ds
