""" ICS autoclave profile detector for AD model using K-Means algorithm. """


from __future__ import annotations

import os
import pickle
from datetime import datetime
import random

import numpy as np
import pandas as pd
from numpy import random

from sklearn.cluster import KMeans
from scipy.spatial import distance
from sklearn import preprocessing

import tools.log as log
from adm_generic import Det_Generic


# Check out
# https://tslearn.readthedocs.io/en/stable/auto_examples/clustering/plot_kernel_kmeans.html#sphx-glr-auto-examples-clustering-plot-kernel-kmeans-py

# https://www.datatechnotes.com/2020/05/anomaly-detection-with-kmeans-in-python.html


class Det_SKL_KMEANS(Det_Generic):
    """
    K-Means detector algorithm implementation
    """

    def __init__(self, logger: log.Log) -> Det_SKL_KMEANS:
        """
        CTOR
        Args:
            logger  - logger object
        Returns:
            <Det_SKL_KMEANS>  - instance of the class
        """
        super().__init__("kmeans", logger)

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
        rng = np.random.RandomState(
            random.randint(cfg["random_state"][0], cfg["random_state"][1])
        ) if cfg["random_state"] is not None else cfg["random_state"]

        return KMeans(n_clusters=cfg["n_clusters"],
                      random_state=rng,
                      n_init="auto")

    def adef_train(self,
                   t_data: pd.DataFrame,
                   cfg: dict,
                   model: object,
                   debug: bool = False) -> pd.DataFrame:
        """
        Method trains KMEANS AD model
        Args:
            t_data: attack data from where model will be made
            cfg: AD model configuration
            model: model object used in detection process
            debug: additional log output
        Returns:
            <DataFrame> - DataFrame containing SST outliers
        """
        x = self.x_scaler(t_data.loc[:, ["temp"]])

        # Apply KMeans clustering
        model.fit(x)
        labels = model.labels_

        # Calculate the distance of each point from its cluster center
        distances = pd.Series([distance.euclidean(x[i], model.cluster_centers_[label]) for i, label in
                              enumerate(labels)])

        y_scaler = preprocessing.MinMaxScaler()
        outliers_df = y_scaler.fit_transform(distances.to_numpy().reshape(-1, 1)).reshape(distances.shape)

        # Detect anomalies (points that are farther from their cluster center than the threshold)
        outliers_df = (outliers_df > cfg["threshold"]).astype(int)

        # Results postprocessing
        ds = self.format_output(outliers_df,
                                t_data)

        return ds

    def adef_detect(self,
               a_data: pd.DataFrame,
               cfg: dict,
               model: object) -> pd.DataFrame:
        """
        Method performs KMEANS model outlier classification
        Args:
            a_data: attack data from where detections will be performed
            cfg: AD model configuration
            model: model object used in detection process
        Returns:
            <DataFrame> - DataFrame containing SST outliers
        """

        x = self.x_scaler(a_data.loc[:, ["temp"]])

        labels = model.predict(x)

        # Calculate the distance of each point from its cluster center
        distances = pd.Series([distance.euclidean(x[i], model.cluster_centers_[label]) for i, label in
                               enumerate(labels)])

        # Detect anomalies (points that are farther from their cluster center than the threshold)
        y_scaler = preprocessing.MinMaxScaler()
        outliers_df = y_scaler.fit_transform(distances.to_numpy().reshape(-1, 1)).reshape(distances.shape)
        outliers_df = (outliers_df > cfg["threshold"]).astype(int)

        # Results postprocessing
        ds = self.format_output(outliers_df,
                                a_data)

        return ds
