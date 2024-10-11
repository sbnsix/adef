""" AD model based on AutoEncoder algorithm. """
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

PATHS = ["./", "../detectors", "../tools", "../threats"]
for path in PATHS:
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), path))
    if path not in sys.path:
        sys.path.insert(1, path)

import tools.log as log
from adm_generic import Det_Generic

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


class Det_PyTorch_AutoEncoder(nn.Module, Det_Generic):
    """
    Autoencoder anomaly detection model with
    threshold based binary class classification (mean square error)
    """
    def __init__(self, logger: log.Log):
        nn.Module.__init__(self)
        Det_Generic.__init__(self,"pytorch_autoencoder", logger)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

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
        # Construct shape of the autoencoder depending on the
        # input signal shape
        input_shape_size = 300
        self.encoder = nn.Sequential(
            nn.Linear(input_shape_size, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True))
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, input_shape_size),
            nn.Sigmoid())

        # ac = Autoencoder()
        # Initialize the model, loss function and optimizer
        model = {"model": self,
                 "criterion": nn.MSELoss(),
                 "optimizer": torch.optim.Adam(self.parameters(), lr=cfg["lr"])
                 }
        model["model"].eval()

        return model

    def evaluate(self,
                 model: object,
                 data: object,
                 cfg: dict) -> (pd.Series, pd.Series):
        """
        Autoencoder evaluation phase used in training and detection to
        streamline outputs for both methods in the same way.
        Args:
            model: input model used to perform anomaly detection process
            data: data obtained from Pytorch dataloader
            cfg: AD model configuration
        Returns:
            (<pd.Series>, <pd.Series>): resulting vector for
        """
        # Evaluation
        inputs, labels = data
        outputs = model["model"](inputs)
        # Calculate reconstruction error for each point
        scores = (outputs - inputs).pow(2)
        # flatten the labels
        labels = labels.view(-1).numpy().astype(int)
        labels = (labels > labels.mean()).astype(int)
        # Convert scores to binary labels
        preds = (scores > cfg["threshold"]).int()

        predictions = preds.view(-1).numpy()
        return labels, predictions

    def adef_train(self,
                   t_data: pd.DataFrame,
                   cfg: dict,
                   model: object,
                   debug: bool = False) -> pd.DataFrame:
        """
        Method trains Autoencoder model
        Args:
            t_data: attack data from where model will be trained
            cfg: AD model configuration
            model: model file from where model will be written
            debug: debugging flag
        Returns:
            <DataFrame> - DataFrame containing outliers
        """

        # Convert the numpy arrays to PyTorch tensors
        X_train = torch.from_numpy(t_data.loc[:, ["temp"]].values).float()
        X_train = X_train.reshape(-1, t_data.shape[0])

        # Create TensorDatasets for training and testing
        # For an autoencoder, the target is the same as the input
        train_data = TensorDataset(X_train, X_train)

        # Define the data loaders
        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        ds = None
        # Training loop
        for epoch in range(10):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data
                model["optimizer"].zero_grad()
                outputs = model["model"](inputs)
                loss = model["criterion"](outputs, inputs)
                loss.backward()
                model["optimizer"].step()
                running_loss += loss.item()

                labels, predictions = self.evaluate(model, data, cfg)
                ds = self.format_output(predictions, t_data)
                return ds

        return ds

    def adef_detect(self,
               a_data: pd.DataFrame,
               cfg: dict,
               model: object) -> pd.DataFrame:
        """
        Method performs SST model outlier classification (banpei implementation)
        Args:
            a_data: attack data from where detections will be predicted
            cfg: AD model configuration
            model: AD model object
        Returns:
            <DataFrame> - DataFrame containing outliers
        """
        # Convert the numpy arrays to PyTorch tensors
        X_test = torch.from_numpy(a_data.loc[:, ["temp"]].values).float()
        X_test = X_test.reshape(-1, a_data.shape[0])

        n_model = self.create(cfg)
        n_model["model"].load_state_dict(model["model"])
        n_model["model"].eval()

        # Create TensorDatasets for training and testing
        # For an autoencoder, the target is the same as the input
        test_data = TensorDataset(X_test, X_test)
        test_loader = DataLoader(test_data, batch_size=32, shuffle=True)
        for i, data in enumerate(test_loader, 0):
            labels, predictions = self.evaluate(n_model, data, cfg)
            ds = self.format_output(predictions, a_data)

            return ds
