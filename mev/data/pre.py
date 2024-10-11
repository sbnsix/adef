""" Data preprocessing module used to prepare input data using various techniques """


from __future__ import annotations
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import SpectralEmbedding
from sklearn.manifold import Isomap
import padasip as pa


class DataPreprocessing:
    """
    Class storing some mathematical manifold techniques
    used to cure input data
    """

    @staticmethod
    def tr_lda(data: np.array, label_names: list, n: int = 1) -> np.ndarray:
        """
        Linear Discriminant Analysis (LDA)
        Args:
            data:
            label_names:
            n:
        Returns:
            <np.ndarray>
        """
        b = np.reshape(data, (-1, 2)).T
        transformed_data = pa.preprocess.LDA(b, label_names, n=n)

        return transformed_data

    @staticmethod
    def tr_pca(data: np.array, label_names: list) -> np.ndarray:
        """
        Principal Component Analysis transformation
        Args:
            data: input data that will be used
            label_names: columns that will be looked at
                         during manifold process
        Returns:
            <np.ndarray> - array containing manifolded data
        """
        # Create a PCA instance
        pca = PCA(n_components=len(label_names))
        # Fit and transform the data using the fitted PCA instance
        transformed_data = pca.fit_transform(data)
        return transformed_data

    @staticmethod
    def tr_tsne(data: np.array, label_names: list) -> np.ndarray:
        """
        t-Distributed Stochastic Neighbor Embedding transformation
        Args:
            data: input data that will be used
            label_names: columns that will be looked at
                         during manifold process
        Returns:
            <np.ndarray> - array containing manifolded data
        """
        tsne = TSNE(n_components=len(label_names), random_state=0)
        transformed_data = tsne.fit_transform(data)
        return transformed_data

    @staticmethod
    def tr_lle(data: np.array, label_names: list) -> np.ndarray:
        """
        Local Linear Embedding transformation
        Args:
            data: input data that will be used
            label_names: columns that will be looked at
                         during manifold process
        Returns:
            <np.ndarray> - array containing manifolded data
        """
        lle = LocallyLinearEmbedding(
            n_components=len(label_names), eigen_solver="dense"
        )
        transformed_data = lle.fit_transform(data)
        return transformed_data

    @staticmethod
    def tr_se(data: np.array, label_names: list) -> np.ndarray:
        """
        Spectral Embedding transformation
        Args:
            data: input data that will be used
            label_names: columns that will be looked at
                         during manifold process
        Returns:
            <np.ndarray> - array containing manifolded data
        """
        se = SpectralEmbedding(n_components=len(label_names))
        transformed_data = se.fit_transform(data)
        return transformed_data

    @staticmethod
    def tr_isomap(data: np.array, label_names: list) -> np.ndarray:
        """
        ISO Mapping transformation
        Args:
            data: input data that will be used
            label_names: columns that will be looked at
                         during manifold process
        Returns:
            <np.ndarray> - array containing manifolded data
        """
        isomap = Isomap(n_components=len(label_names))
        transformed_data = isomap.fit_transform(data)
        return transformed_data
