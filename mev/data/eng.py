""" Data engineering module used to clean up data """


import pandas as pd


class DataEng:
    """
    Data engineering class that prepares ICS autoclave process
    input data into more AD simple data set where anomaly
    detection is more visible and gives much higher precision
    and recall vs. unprocessed data sets.
    """

    @staticmethod
    def compute_shift(
        t_data: pd.DataFrame,
        a_data: pd.DataFrame,
        file_name: str = None,
        shift_val: int = 0,
    ) -> int:
        """
        Method performs manifold operation on attack data to simplify
        detection by anomaly detection algorithms.
        Args:
            t_data: input template data representing original process (pre-recorded)
            a_data: input attack data to be analysed
            file_name: debugging prefix for image file name
            shift_val: given shift value that will be added in the data shift process
        Returns:
            <int> - number of point that attack data is behind(negative) or
                    ahead (positive) vs template data.
        """
        # Initial computation parameters
        # Maximum number of points that are required
        # to compute data shift
        end = 60
        zero_length = 20

        if not isinstance(t_data.index, pd.Index): # .Float64Index
            data_mir = pd.Series(t_data.loc[:end, "temp"])
            a_data_mir = pd.Series(a_data.loc[:end, "temp"])
        else:
            data_mir = pd.Series(t_data.loc[:t_data.index[end], "temp"])
            a_data_mir = pd.Series(a_data.loc[:a_data.index[end], "temp"])

        # Template data
        # data_mir = pd.DataFrame(t_data.loc[:end, "temp"])
        # data_mir.reset_index(drop=True, inplace=True)
        # data_mir = pd.Series(t_data.loc[:end, "temp"])
        data_mir = data_mir.astype(int)
        # data_avg = int(data_mir[:zero_length].sum() / zero_length)
        data_avg = int(data_mir.loc[:zero_length, ].mean())
        data_idx = data_mir[data_mir.loc[:, ] <= data_avg].index[-1]

        # Attack data
        # a_data_mir = pd.DataFrame(a_data.loc[:end, "temp"])
        # a_data_mir.reset_index(drop=True, inplace=True)
        # a_data_mir = pd.Series(a_data_mir.loc[:end, "temp"])
        a_data_mir = a_data_mir.astype(int)

        a_data_avg = int(a_data_mir.loc[:zero_length, ].mean())
        # a_data_mir[:zero_length].sum() / zero_length))

        a_data_idx = a_data_mir[
            (a_data_mir.loc[:, ] > a_data_avg - 3)
            & (a_data_mir.loc[:, ] < a_data_avg + 3)
        ].index[-1]

        a_data_mir.loc[:a_data_idx, ] = data_avg
        # a_data_idx = a_data_mir[a_data_mir.loc[:,] == data_avg].index[-1]

        # TODO: Shift is still inaccurate causing models not to fit to
        # TODO: data as expected. This needs to be fixed before proceeding
        # TODO: further.

        # Debugging plot
        # Add Normal and Attack trace + shift label

        # Compute shift
        shift = 0
        if a_data_idx < data_idx:
            shift = (data_idx + 1) - a_data_idx
        elif a_data_idx == data_idx:
            shift = 0
        elif a_data_idx > data_idx:
            # Magic X shift
            # TODO: Figure out how to compute gradient on 0 label
            # TODO: for sum up data sets
            shift = (a_data_idx + shift_val) - data_idx

        # if file_name:
        #    Plotter.aligned_plot(
        #        data_mir, a_data_mir, data_idx, a_data_idx, file_name, False
        #    )
        return shift

    @staticmethod
    def manifold_data_step(
        t_data: pd.DataFrame,
        t_mir_data: pd.DataFrame,
        a_data: pd.DataFrame,
        file_name: str = None,
        magic_val: int = 0,
    ) -> pd.DataFrame:
        """
        Method performs manifold operation on attack data to simplify
        detection by anomaly detection algorithms.
        Args:
            t_data: input template data representing original process (pre-recorded)
            t_mir_data: reverse input template data representing original process (pre-recorded)
            a_data: input attack data to be folded
        Returns:
            <pd.DataFrame> - Data frame containing folded attack data
        """
        # Initial parameters
        shift = DataEng.compute_shift(t_data, a_data, file_name, magic_val)

        # Attack data
        a_data_mir = a_data.copy(deep=True)
        a_data_mir.reset_index(inplace=True)

        # Mirrored data of the original production cycle from template (ground truth data)
        data_mir = t_mir_data.copy(deep=True)
        data_mir.reset_index(inplace=True)

        a_data_mir.loc[:, "tc"] += data_mir.loc[:, "tc"]

        # X-axis adjustment - shift process template on X axis to align with
        # the attack data. This helps contrast attacks vs. recorded
        # autoclave profile and enhance anomaly detection.

        # Move front to back
        if shift > 0:
            a = data_mir.loc[shift:, ["temp", "tc"]].copy(deep=True)
            b = data_mir.loc[: shift - 1, ["temp", "tc"]].copy(deep=True)
            c = pd.concat([a, b])
            c.reset_index(drop=True, inplace=True)
            data_mir.loc[:, ["temp", "tc"]] = c

        # Move tail to front
        elif shift < 0:
            length = data_mir.shape[0] + shift
            a = data_mir.loc[: length - 1, ["temp", "tc"]].copy(deep=True)
            b = data_mir.loc[length:, ["temp", "tc"]].copy(deep=True)
            c = pd.concat([b, a])
            c.reset_index(drop=True, inplace=True)
            data_mir.loc[:, ["temp", "tc"]] = c

        # Add shifted template to attack data to generate final trace.
        a_data_mir.loc[:, "temp"] += data_mir.loc[:, "temp"]
        a_data_mir.set_index("time", inplace=True)

        return a_data_mir

    @staticmethod
    def manifold_data(
        t_data: pd.DataFrame,
        t_mir_data: pd.DataFrame,
        a_data: pd.DataFrame,
        file_name: str = None,
    ) -> pd.DataFrame:
        """

        Args:
            t_data:
            t_mir_data:
            a_data:
            file_name:

        Returns:

        """
        best_dev = 10
        aligned_ds = None

        if "index" in t_data.columns:
            t_data.drop("index", axis=1, inplace=True)

        for m_value in range(0, 7):
            # Perform alignment test
            ds = DataEng.manifold_data_step(t_data,
                                            t_mir_data,
                                            a_data,
                                            file_name,
                                            m_value)

            # Validate data
            ds.reset_index(inplace=True)
            ds.set_index("time", inplace=True)

            # TODO: Compute standard deviation for all 0 label values
            ds_dev = ds.loc[(ds.loc[:, "label"] == 0), "temp"]
            dev = ds_dev.sum() / ds_dev.shape[0]

            if abs(dev) < best_dev:
                best_dev = abs(dev)
                aligned_ds = ds.copy(deep=True)

        return aligned_ds

    @staticmethod
    def split_data(
        data: pd.DataFrame, cycle_size: int, set_index: bool = False
    ) -> list:
        """
        Method splits the DataFrame into smaller chunks
        equal to single autoclave cycle that is used in further processing.
        Args:
            data: input Dataframe
            cycle_size: chunk size that df will be split into
            set_index: set time index into chunks
        Returns:
            <list>  - df chunks formed to a list
        """
        chunks = []
        data_c = data.copy(deep=True)
        data_c.reset_index(inplace=True)
        num_chunks = int(data_c.shape[0] / cycle_size)
        for i in range(0, num_chunks + 1):
            chunk = data_c.iloc[(i * cycle_size) : ((i + 1) * cycle_size),]
            chunk_size = len(chunk)
            if chunk_size == cycle_size or abs(cycle_size - chunk_size) < 4:
                if set_index:
                    chunk.set_index("time", inplace=True)
                chunks.append(chunk)
        return chunks
