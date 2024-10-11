import inspect
import pandas as pd
from mev.data.filters import DataFilter


class DataFilterFactory:
    """
    Data filtering factory - implements automation over multiple filters
    over input/output data used in anomaly detection process
    """

    # Static filter mapping - it scans content of the DataFilter object and initialize
    #
    filters = dict(inspect.getmembers(DataFilter, predicate=inspect.isfunction))

    @staticmethod
    def filter(
        data: pd.DataFrame, config: dict, columns: list = ["temp"]
    ) -> pd.DataFrame:
        """
        Multiple filter implementation
        Args:
            data: input data frame
            config: Filter configuration (all parameters required
                           to run given filter successfully)
            columns: list of data columns on which filter will be applied
        Returns:
            <pd.DataFrame> - filtered data result
        """
        # This applied here as the data set can be used by many detection algorithms
        # that might have a different filtering requirements for the data set.
        # https://towardsdatascience.com/fourier-transform-for-time-series-292eb887b101
        # https://docs.scipy.org/doc/scipy/reference/fft.html
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.fftpack.dct.html

        f_data = data.copy(deep=True)

        # Iterate data over filters supplied from configuration
        for filter_config in config:
            for filter_name, filter_cfg in filter_config.items():
                if filter_name in DataFilterFactory.filters.keys():
                    try:
                        if "kalman" == filter_name.lower():
                            filter_cfg["initial_state_mean"] = f_data.loc[0, columns]

                        if filter_cfg is None:
                            f_data.loc[:, columns] = DataFilterFactory.filters[
                                filter_name
                            ](f_data.loc[:, columns]).values.reshape(
                                (f_data.shape[0], f_data.loc[:, columns].shape[1])
                            )
                        else:
                            f_data.loc[:, columns] = DataFilterFactory.filters[
                                filter_name
                            ](
                                f_data.loc[:, columns], *(list(filter_cfg.values()))
                            ).values.reshape(
                                (f_data.shape[0], f_data.loc[:, columns].shape[1])
                            )
                    except Exception as ex:
                        print(ex)

        return f_data
