""" ADEF Plotter module """

from __future__ import annotations
import os
import math
import glob

import numpy
import pandas as pd
import numpy as np
from PIL import Image

import matplotlib
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
# import matplotlib.patches as mpatches
# from matplotlib import colormaps
# import matplotlib.colors as mcolors
from matplotlib.legend_handler import HandlerBase
from matplotlib.patches import Circle

from mpl_toolkits.axes_grid1 import ImageGrid
from mpl_toolkits.axes_grid1 import make_axes_locatable

from sklearn import metrics
from tools.log import Log
from tools.iter_helper import IterHelper

matplotlib.use("Agg")


class MidpointNormalize(matplotlib.colors.Normalize):
    """
    Midpoint normalization for diff comparison graphs
    """

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        matplotlib.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        if self.vmax - self.vmin != 0:
            zero_ratio = abs(self.vmin / (self.vmax - self.vmin))
            x, y = [self.vmin, self.midpoint, self.vmax], [0, zero_ratio, 1]
        else:
            # v_ext = np.max([np.abs(self.vmin), np.abs(self.vmax)])
            if self.midpoint is None:
                x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
            else:
                x, y = [self.vmin, self.vmax], [0, 1]

        return np.ma.masked_array(np.interp(value, x, y))


class Plotter:
    """
    Class implementing plotting functions for
    ICS AD model experiments.
    """

    # Default color map for all heatmaps plots (Red, Yellow and Green colors are used)
    cm_ryg = matplotlib.colors.LinearSegmentedColormap.from_list("", ["red", "yellow", "green"])
    # Reversed color map used for EER display
    cm_ryg_r = matplotlib.colors.LinearSegmentedColormap.from_list("", ["green", "yellow", "red"])

    # Only positive map
    cm_yg = matplotlib.colors.LinearSegmentedColormap.from_list("", ["yellow", "green"])
    cm_yg_r = matplotlib.colors.LinearSegmentedColormap.from_list("", ["green", "yellow"])

    # Only negative map
    cm_ry = matplotlib.colors.LinearSegmentedColormap.from_list("", ["red", "yellow"])
    cm_ry_r = matplotlib.colors.LinearSegmentedColormap.from_list("", ["yellow", "red"])

    @staticmethod
    def save_and_display(fig_file, bbox, show) -> None:
        """
        Method saves plot and close matplotlib graphics
        Args:
            fig_file - figure file name
            bbox - layout for bounding box -
                    normal = None,
                    tight - removal of empty white space around image
            show - boolean flag to show whether display plot
        Returns:
            <None>
        """
        plt.margins(0)

        if fig_file is not None and fig_file.endswith(".png"):
            plt.savefig(fig_file, bbox_inches=bbox)

        if show:
            plt.show()

        plt.clf()
        plt.close("all")

    @staticmethod
    def input_ts(x: pd.Series,
                 y: pd.Series,
                 label: str,
                 fig_file: str,
                 show: bool = False) -> None:
        """
        Method plots input time series
        Args:
            x: Time axis
            y: Y axis data for TS
            label: time series name
            fig_file: figure file name
            show: flag to show
        Returns:
            <None>
        """

        plt.ylim([y["temp"].min(), y["temp"].max()+y["temp"].max()*0.15])
        plt.xlim([x[x.index[0]], x[x.index[-1]]])

        plt.grid(True, color="lightgrey", linestyle="-", linewidth=0.5)

        plt.plot(x, y, color="blue", label=f"{label}", linestyle="-", linewidth=1)

        plt.legend()

        Plotter.save_and_display(fig_file, None, show)

    @staticmethod
    def input_ts_color(data: pd.Series,
                       a_data: pd.Series,
                       title: str,
                       cfg: dict,
                       fig_file: str,
                       show: bool = False,
                       log2f: Log = None) -> None:
        """
        Method plots input time series
        Args:
            data: normal time series representing process
            a_data: attacked time series representing process
            title: Name of the graph
            cfg: detection configuration
            fig_file: figure file name
            show: flag to show
            log2f: logger object
        Returns:
            <None>
        """
        # Flags for legend labels that will be marked only once on the graph
        label_blue = True
        label_red = True

        # Split data set into consecutive sets that will be colored based on the label value
        split_data = []
        t_data = a_data.copy(deep=False)

        # Aligning time to 0
        index_list = t_data.index.tolist()
        t_data.index = [i - index_list[0] for i in index_list]

        split_idx = t_data.loc[:, "label"].diff()
        split_idx = split_idx.where(split_idx != 0).dropna().index.tolist()

        plt.grid(True, color="lightgrey", linestyle="-", linewidth=0.5)

        prev_idx = t_data.index[0]
        for idx in split_idx:
            c_idx = t_data.loc[prev_idx:idx,].index[-2]
            split_data.append(t_data.loc[prev_idx:c_idx,])
            prev_idx = c_idx

        # Add last part of the trace
        split_data.append(t_data.loc[prev_idx : t_data.index[-1],])

        plt.xlim([0, cfg["process"]["cycle_len"] * cfg["process"]["samples"]])
        plt.ylim([0, cfg["process"]["max_value"]])

        # Normal trace - before attack
        if data is not None:
            n_data = data.copy(deep=False)
            n_data.reset_index(inplace=True)
            index_list = n_data.index.tolist()
            n_data.index = [i - index_list[0] for i in index_list]
            n_data = n_data.loc[
                n_data.index[0] : n_data.index[
                    cfg["process"]["cycle_len"] * cfg["process"]["samples"]
                ],
            ]
            plt.plot(
                n_data.index,
                n_data.loc[:, "temp"],
                "darkgreen",
                linestyle="--",
                label="Ideal cycle",
            )

        for data_set in split_data:
            if data_set is None:
                log2f.warn("Data set is empty")
                continue
            if data_set.index.isna().all():
                log2f.warn("Index incorrect")
                continue

            # One incorrect sample allowed - it will not make a huge difference on the graph
            if data_set.loc[:, "label"].sum() < 2:
                if label_blue:
                    plt.plot(
                        data_set.index,
                        data_set.loc[:, "temp"],
                        "b",
                        linestyle="-",
                        label="Normal operation",
                    )
                    label_blue = False
                else:
                    plt.plot(
                        data_set.index, data_set.loc[:, "temp"], "b", linestyle="-"
                    )
            else:
                if label_red:
                    plt.plot(
                        data_set.index,
                        data_set.loc[:, "temp"],
                        "r",
                        linestyle="-",
                        label="Attack",
                    )
                    label_red = False
                else:
                    plt.plot(
                        data_set.index, data_set.loc[:, "temp"], "r", linestyle="-"
                    )

        # TODO: Debug this section
        # Plot thresholds if attack type is threshold
        if "thb" in fig_file and "threshold" in cfg.keys():
            level = cfg["threshold"]
            plt.hlines(
                y=level,
                xmin=0,
                xmax=t_data.index[-1],
                color="m",
                linestyles="dashed",
                linewidths=1,
            )
            plt.hlines(
                y=-level,
                xmin=0,
                xmax=t_data.index[-1],
                color="m",
                linestyles="dashed",
                linewidths=1,
            )

        plt.xlabel("Time[min]")
        if t_data.loc[:, "temp"].min() < 0:
            plt.ylabel("Δ Temperature [°C]")
        else:
            plt.ylabel("Temperature [°C]")

        plt.title(title)
        plt.legend()

        Plotter.save_and_display(fig_file, None, show)

    @staticmethod
    def input_process(data: pd.Series,
                      title: str,
                      cfg: dict,
                      v_lines: list,
                      h_lines: list,
                      fig_file: str,
                      show: bool = False,
                      log2f: Log = None) -> None:
        """
        Method plots input time series
        Args:
            data: normal time series representing process
            a_data: attacked time series representing process
            title: Name of the graph
            cfg: detection configuration
            fig_file: figure file name
            show: flag to show
            log2f: logger object
        Returns:
            <None>
        """

        # Aligning time to 0
        index_list = data.index.tolist()
        data.index = [i - index_list[0] for i in index_list]

        plt.grid(True, color="lightgrey", linestyle="-", linewidth=0.5)

        plt.xlim([0, cfg["process"]["cycle_len"] * cfg["process"]["samples"]])
        plt.ylim([0, cfg["process"]["max_value"]])

        # Normal trace - before attack
        plt.plot(
            data.index,
            data.loc[:, "tc"],
            "darkgreen",
            linestyle="--",
            label="Control input",
        )
        plt.plot(
            data.index,
            data.loc[:, "temp"],
            "blue",
            linestyle="-",
            label="Autoclave process",
        )
        for v_line in v_lines:
            plt.vlines(
                x=v_line,
                ymin=0,
                ymax=cfg["process"]["max_value"],
                color="black",
                linestyles="dashed",
                linewidths=1,
            )

        for h_line in h_lines:
            plt.hlines(
                y=h_line,
                xmin=0,
                xmax=data.index[-1],
                color="red",
                linestyles="dashed",
                linewidths=1,
            )

        plt.text(60, 35, "a) Heating")
        plt.text(130, 35, "b) Part\ntreatment")
        plt.text(200, 35, "c) Cooling")
        plt.text(12, 165, "Minimum heating", color="red")
        plt.text(12, 195, "Optimal heating", color="red")

        plt.xlabel("Time[min]")
        plt.ylabel("Temperature [°C]")

        plt.title(title)
        plt.legend()

        Plotter.save_and_display(fig_file, None, show)

    @staticmethod
    def input_ts_color_split(
        a_data: pd.DataFrame,
        sp_cycles: int,
        title: str,
        cfg: dict,
        fig_file: str,
        show: bool = False,
    ) -> None:
        """
        Method plots input time series in color coded fashion
        Args:
            a_data: time series used in model training
            sp_cycles: amount of split cycles that are used to split data sets into two parts
            title: name of the graph
            cfg: detection configuration
            fig_file: figure file name
            show: flag to show
        Returns:
            <None>
        """

        def plot_detection(data: pd.DataFrame,
                           ax: object,
                           x_label: str,
                           y_label:str,
                           y_lim: list,
                           title: str) -> None:
            """
            Method plots a single time series graph on multiple grid axis setup
            Args:
                data: input data that will be used to generate graph
                ax: axis object onto which graph is being plotted
                x_label: name of the X label
                y_label: name of the Y label
                y_lim: list of Y axis limits 0 - minimum, 1 - maximum
                title: graph title
            Returns:
                <None>
            """
            # Split data set into consecutive sets that
            # will be colored based on the label value
            split_data = []
            split_idx = data.loc[:, "label"].diff()
            split_idx = split_idx.where(split_idx != 0).dropna().index.tolist()

            prev_idx = data.index[0]
            for idx in split_idx:
                c_idx = data.loc[prev_idx:idx,].index[-2]
                split_data.append(data.loc[prev_idx:c_idx,])
                prev_idx = c_idx

            # Add last part of the trace
            split_data.append(data.loc[prev_idx : data.index[-1],])

            # Apply grid labels
            ax.grid(True, color="lightgrey", linestyle="-", linewidth=0.5)

            # Apply correct labels
            ax.set_xlabel(x_label)

            if y_label:
                ax.set_ylabel(y_label)

            ax.set_title(title)

            # Align X and Y axis correctly so there are no gaps in front and in the back
            ax.set_xlim(data.index[0], data.index[-1])
            ax.set_ylim(*y_lim)

            # Flags for legend labels that will be marked only once on the graph
            for data_set in split_data:

                # Data set is empty or Index incorrect
                if data_set is None or data_set.index.isna().all():
                    continue
                if data_set.index.isna().all():
                    continue

                # One incorrect sample allowed - it will not make a huge difference on the graph
                if data_set.loc[:, "label"].sum() < 2:
                    ax.plot(data_set.index,
                            data_set.loc[:, "temp"],
                            "b",
                            linestyle="-")
                else:
                    ax.plot(data_set.index,
                            data_set.loc[:, "temp"],
                            "r",
                            linestyle="-")

            # Plot threshold lines if detection type is THB
            if "thb" in fig_file:
                level = cfg["model"]["ad"]["thb"]["threshold"]
                plt.hlines(y=level,
                           xmin=0,
                           xmax=data.index[-1],
                           color="m",
                           linestyles="dashed",
                           linewidths=1)
                plt.hlines(y=-level,
                           xmin=0,
                           xmax=data.index[-1],
                           color="m",
                           linestyles="dashed",
                           linewidths=1)

        # Y axis limits
        ylim_min = a_data.loc[:, "temp"].min() * 1.1
        ylim_max = a_data.loc[:, "temp"].max() * 1.1

        split_idx = sp_cycles * cfg["process"]["cycle_len"] * cfg["process"]["samples"]

        data_tr = a_data.loc[: a_data.index[split_idx]-1, ]
        data_ev = a_data.loc[a_data.index[split_idx]:, ]

        ratio_tr = 1
        ratio_ev = 1

        # Precompute ratio
        if data_tr.shape[0] < data_ev.shape[0]:
            ratio_tr = data_tr.shape[0] / data_ev.shape[0]
        elif data_tr.shape[0] > data_ev.shape[0]:
            ratio_ev = data_tr.shape[0] / data_ev.shape[0]

        fig, ax = plt.subplots(
            1,
            2,
            figsize=(ratio_tr * 10 + ratio_ev * 10, 4),
            gridspec_kw={"width_ratios": [ratio_tr, ratio_ev]},
            layout="compressed"
        )

        dx = {"tr": data_tr, "ev": data_ev}
        idx = 0
        for k in dx.keys():
            dx[k] = dx[k].reset_index()
            if "index" in dx[k].columns:
                dx[k] = dx[k].drop("index", axis=1)
            if "time" in dx[k].columns:
                dx[k] = dx[k].set_index("time")

            if data_tr.loc[:, "temp"].min() < 0:
                y_label = "Δ Temperature [°C]"
            else:
                y_label = "Temperature [°C]"

            if 0 == idx:
                ax_title = "Training"
            else:
                ax_title = "Evaluation"

            plot_detection(dx[k],
                           ax[idx],
                           "Time [min]",
                           (y_label if idx == 0 else ""),
                           [ylim_min, ylim_max],
                           ax_title)
            idx += 1

        plt.suptitle(title)

        # Get handles and labels from the plot
        handles, labels = plt.gca().get_legend_handles_labels()

        legend_lines = [Line2D([], [], color="b", label="Normal operation"),
                        Line2D([], [], color="r", label="Attack")]
        if "thb" in fig_file:
            legend_lines.append(Line2D([], [], color="m", linestyles="dashed", label="Limit"))

        # Append manual items to the legend handles
        handles.extend(legend_lines)

        fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.0))
        Plotter.save_and_display(fig_file, "tight", show)

    @staticmethod
    def detection_ts(data: pd.Series,
                     title: str,
                     fig_file: str,
                     show: bool = False) -> None:
        """
        Method
        Args:
            data: pd.Series representing per cycle detection data set
            title: name of the graph
            fig_file: file where graph will be saved
            show: flag to enable direct display on the screen
        Returns:
            <None>
        """
        # Plot Detection
        fig, axs = plt.subplots(1, 1, figsize=(7, 5))

        plt.ylim(-1.1, 1.1)
        plt.yticks([-1, 0, 1])
        plt.grid(True, color="lightgrey", linestyle="-", linewidth=0.5)

        axs.stairs(data.loc[:, "label"], hatch="\\\\", label="Ground truth")
        axs.stairs(-data.loc[:, "result"], hatch="//", label="AD model detection")
        # for ax in axs:
        axs.legend()

        plt.xlabel("Time [min]")
        plt.ylabel("Detection class [0/1]")
        plt.title(title)

        Plotter.save_and_display(fig_file, None, show)

    @staticmethod
    def detection_cd(data: pd.Series,
                     title: str,
                     fig_file: str,
                     show: bool = False) -> None:
        """
        Method that creates cycle detector detection graph (per cycle) with
        ground truth above and detection result on the bottom of the graph.
        Args:
            data: pd.Series representing per cycle detection data set
            title: name of the graph
            fig_file: file where graph will be saved
            show: flag to enable direct display on the screen
        Returns:
            <None>
        """
        # data_cp = data.reset_index()
        # data_cp.set_index("cycle", inplace=True)

        fig, axs = plt.subplots(1, 1, figsize=(7, 5))

        plt.ylim(-1.1, 1.1)
        plt.yticks([-1, 0, 1])
        plt.grid(True, color="lightgrey", linestyle="-", linewidth=0.5)

        axs.stairs(data.loc[:, "cd_label"], hatch="\\\\", label="Ground truth")

        axs.stairs(-data.loc[:, "cd_result"], hatch="//", label="CD detection")
        axs.legend()

        plt.xlabel("Cycle #")
        plt.ylabel("Detection class [0/1]")
        plt.title(title)

        Plotter.save_and_display(fig_file, None, show)

    @staticmethod
    def plot_limit(plot: object,
                   limit: float,
                   label: str = None,
                   color: str = "darkgreen") -> None:
        """
        Method adds error allowable limit
        Args:
            plot: plotter object
            limit: ROC limit used in the algorithm evaluation
            label: Limit name on the graph
            color: Matplotlib color name for the limit
        Returns:
            <None>
        """
        plt.vlines(limit, 0, 1.0, color=color, linestyles="--", linewidth=1, label=label)

    @staticmethod
    def roc_2d(data: pd.DataFrame,
               cfg: dict,
               title: str,
               fig_file: str,
               limit: float,
               line_enabled: bool,
               show: bool = False) -> None:
        """
        Method plots roc curve for given AD model
        Args:
            data: Data set containing false positive rate and true positive rate
            cfg: Configuration for metrics - area under curve value
            fig_file: figure file name where plot will be saved,
                       if empty figure will not be saved
            title: Plot title
            limit: Optimal limit ROC parameters used in algorithm evaluation
            line_enabled: if True it connects lines on the ROC curve, otherwise it draws scatter plot
            show: test flag that determines whether to show
                  the figure on screen (True) or not.
        Returns:
            <None>
        """
        plt.grid(True, color="lightgrey", linestyle="-", linewidth=0.5)
        Plotter.plot_limit(plt, limit, f"Optimal detection at {limit*100:.2f}%")

        # Plot default 50% (random) model for comparison
        plt.plot([0, 1], [0, 1], color="black", linestyle="--", label="No skill")

        if line_enabled:
            # Plot ROC curve - line
            plt.plot(
                data.loc[:, data.columns[0]],
                data.loc[:, data.columns[1]],
                color="orange",
                label=f"AUC: {cfg['auc']:.2f}",
                linewidth=2,
                zorder=1
            )

        # Plot FPR/TPR points only
        plt.scatter(
            data.loc[:, data.columns[0]],
            data.loc[:, data.columns[1]],
            marker="x",
            color="red",
            linewidth=1,
            zorder=2,
        )
        if "fpr" in cfg.keys() and "tpr" in cfg.keys():
            plt.scatter(
                [cfg["fpr"]],
                [cfg["tpr"]],
                marker="+",
                color="red",
                linewidth=1,
                zorder=2,
            )

        plt.xlabel("FPR")
        plt.ylabel("TPR")

        plt.title(title)
        plt.legend()

        Plotter.save_and_display(fig_file, None, show)

    @staticmethod
    def roc_2d_anim(data: pd.DataFrame,
                    cfg: dict,
                    title: str,
                    fig_file: str,
                    limit: float,
                    line_enabled: bool,
                    show: bool = False) -> None:
        """
        Method plots animated roc curve for given AD model.
        Args:
            data: Data set containing false positive rate and true positive rate
            cfg: Metric configuration such as area under curve value
            fig_file: figure file name where plot will be saved,
                       if empty figure will not be saved
            title: Plot title
            limit: Optimal limit ROC parameters used in algorithm evaluation
            line_enabled: Flag to determine whether to plot line or scatter
            show: test flag that determines whether to show
                  the figure on screen (True) or not.
        Returns:
            <None>
        """
        cnt = 1
        frame_folder = []
        for l in range(2, data.shape[0]):
            if l in data.index:
                file_name = f"{fig_file}_s{cnt}.png"
                Plotter.roc_2d(data.loc[:l, :],
                               cfg,
                               title,
                               file_name,
                               limit,
                               line_enabled,
                               show)

                frame_folder.append(file_name)
            cnt += 1

        frames = [Image.open(image) for image in frame_folder]
        if not frames:
            return
        frame_one = frames[0]
        frame_one.save(fig_file, format="GIF", append_images=frames, save_all=True, duration=250, loop=0)

    @staticmethod
    def roc_3d(data: pd.DataFrame,
               title: str,
               fig_file: str,
               num_models: int,
               line_enabled: bool,
               model_selected: int = -1,
               show: bool = False) -> None:

        # Create a figure
        fig = plt.figure()

        # Create a 3D axis
        ax = fig.add_subplot(111, projection='3d')
        # Adjust the axes position
        ax.set_position([0, 0, 1, 1])

        model_count = len(data["model_no"].unique())

        # Create a colormap
        cmap = Plotter.cm_ryg # colormaps["viridis"]

        # Generate colors
        colors = [cmap(i) for i in np.linspace(0, 1, model_count+1)]

        for m in range(model_count, 0, -1):
            roc = data[data["model_no"] == m]

            fpr = roc.loc[:, "fpr"]
            model_no = roc.loc[:, "model_no"]

            tpr = roc.loc[:, "tpr"]

            if line_enabled:
                ax.plot(fpr,
                        model_no,
                        tpr,
                        color='r' if model_selected == m else colors[m],
                        linewidth=1.5 if model_selected == m else 0.7)

            ax.scatter(
                fpr,
                model_no,
                tpr,
                marker="x",
                color='r' if model_selected == m else colors[m],
            )
        # Set the limits
        ax.set_xlim([0, 1.05])
        ax.set_zlim([0, 1.05])
        ax.set_ylim([1, num_models])

        ax.set_xlabel('FPR')
        ax.set_ylabel('Model number')
        ax.set_zlabel('TPR')

        plt.title(title)
        Plotter.save_and_display(fig_file, None, show)

    @staticmethod
    def animate_image(frame_folder: str,
                      fig_file: str,
                      duration_delay: int = 250,
                      loop_count: int = 0) -> None:
        """
        Method creates animated png from multiple roc curves used in AD model training - showing
        how best model is created.
        Args:
            roc_files: list of ROC curve files generated by multiple models
            fig_file: output png file where animation will be written
            show: flag to enable displaying resulting file
        """
        frames = [Image.open(image) for image in glob.glob(frame_folder)]
        if not frames:
            return
        frame_one = frames[0]
        frame_one.save(fig_file,
                       format="GIF",
                       append_images=frames,
                       save_all=True,
                       duration=duration_delay,
                       optimize=True,
                       loop=loop_count)

    @staticmethod
    def d2_tau(
        delta1: pd.Series,
        delta2: pd.Series,
        title: str,
        fig_file: str,
        show: bool = False,
    ) -> None:
        """
        Method visualize delta2 over tau changes for cycle detector activity.
        Args:
            delta1: Cycle detector delay computed vs. AD model algorithm
            delta2: Cycle detector delay computed vs. AD model algorithm + delay
            title: graph title
            fig_file: name of the file where figure will be saved
            show: flag to enable (True) or disable (False) graph display on screen
        Returns:
            <None>
        """
        # Create aggregated figure
        fig = plt.figure()  # figsize=(14.0, 10.0))
        plt.grid(True, color="lightgrey", linestyle="-", linewidth=0.5)
        gs = fig.add_gridspec(1, 1, hspace=0.4, wspace=0.15)
        axs = gs.subplots(squeeze=False)

        # Detection delay delta 2 plot
        axs[0, 0].plot(
            delta2.index,
            delta2.loc[:, "avg"],
            color="orange",
            label="Delta 2 (Cycle detector) delay over tau",
            linewidth=2,
        )
        delta1_ext = pd.DataFrame(delta2)
        delta1_ext.loc[:, "d1_avg"] = delta1.loc[0, "avg"]

        axs[0, 0].plot(
            delta1_ext.index,
            delta1_ext.loc[:, "d1_avg"],
            color="blue",
            label="Delta 1 (Cycle detector) delay over tau",
            linewidth=2,
        )

        axs[0, 0].set_xlabel("tau")
        axs[0, 0].set_ylabel("Delta 1/Delta 2")
        axs[0, 0].set_title(title)

        # Add vertical lines in places where cycle detector is activated (CD)
        # and deactivated
        one_idx = pd.DataFrame([(delta2.loc[:, "avg"] > 0)]).index

        if one_idx.shape[0] > 0:
            tau_start = delta2.index[0]
            tau_stop = delta2.index[-1]
            y_axis_max_size = axs[0, 0].dataLim.max[-1]
            axs[0, 0].vlines(
                [tau_start, tau_stop],
                0,
                y_axis_max_size,
                linestyles="dashed",
                colors="c",
            )

        plt.legend()

        Plotter.save_and_display(fig_file, None, show)

    @staticmethod
    def cf_over_tau(
        fn: pd.Series,
        tp: pd.Series,
        fp: pd.Series,
        tn: pd.Series,
        label: str,
        fig_file: str,
        show: bool = False,
    ) -> None:
        """
        Method plots confusion matrix over tau
        Args:
            fn - false negative count in confusion matrix series
            tp - true negative count in confusion matrix series
            fp - false positive count in confusion matrix series
            tn - true negative count in confusion matrix series
            fig_file - name of the file where figure will be saved
            show - display on screen flag
        Returns:
            <None>
        """
        bar_width = 1
        plt.figure(figsize=(14.0, 10.0))

        label_done = False

        for row in fn.index:
            y_offset = 0

            if not label_done:
                plt.bar(
                    row,
                    fn[row],
                    bar_width,
                    bottom=y_offset,
                    color="blue",
                    label="False Negative",
                )
                y_offset += fn[row]
                plt.bar(
                    row,
                    tp[row],
                    bar_width,
                    bottom=y_offset,
                    color="green",
                    label="True Positive",
                )
                y_offset += tp[row]
                plt.bar(
                    row,
                    fp[row],
                    bar_width,
                    bottom=y_offset,
                    color="yellow",
                    label="False Positive",
                )
                y_offset += fp[row]
                plt.bar(
                    row,
                    tn[row],
                    bar_width,
                    bottom=y_offset,
                    color="orange",
                    label="True Negative",
                )
                label_done = True
            else:
                plt.bar(row, fn[row], bar_width, bottom=y_offset, color="blue")
                y_offset += fn[row]
                plt.bar(row, tp[row], bar_width, bottom=y_offset, color="green")
                y_offset += tp[row]
                plt.bar(row, fp[row], bar_width, bottom=y_offset, color="yellow")
                y_offset += fp[row]
                plt.bar(row, tn[row], bar_width, bottom=y_offset, color="orange")

        plt.xlabel("tau")
        plt.ylabel("FN/TP/FP/TN count")
        plt.title(label)

        plt.legend()

        Plotter.save_and_display(fig_file, None, show)

    @staticmethod
    def cf_insight(
        data: pd.Series, title: str, fig_file: str, show: bool = False
    ) -> None:
        """
        Method generates graph for component
        Args:
            data  - pd.Dataframe containing tn, tp, fn, fp CF components
            title - name of the graph
            fig_file - file where graph will be saved
            show - flag to enable direct display on the screen
        Returns:
            <None>
        """
        fig, axs = plt.subplots(5, 1, figsize=(7, 5))

        cnt = 0
        d_sum = None
        for key, color in {
            "tp": "green",
            "fp": "blue",
            "tn": "orange",
            "fn": "red",
        }.items():
            axs[cnt].plot(data.index, data.loc[:, key], color)
            axs[cnt].set_ylabel(key)

            if d_sum is None:
                d_sum = data.loc[:, key]
            else:
                d_sum += data.loc[:, key]

            cnt += 1

        axs[4].plot(data.index, d_sum, "black")
        axs[4].set_ylabel("sum")

        axs[0].set_title(title)
        plt.xlabel("Index")

        Plotter.save_and_display(fig_file, None, show)

    @staticmethod
    def tpr_fpr(fpr: pd.Series,
                tpr: pd.Series,
                label: str,
                fig_file: str,
                show: bool = False
    ) -> None:
        """
        Method prints TPD/FPR bar graph
        Args:
            fpr: False Positive Rate
            tpr: True Positive Rate
            label: Graph label
            fig_file: File name
            show: Graph display flag
        Returns:
            <None>
        """
        bar_width = 1
        # Create aggregated figure
        fig = plt.figure(figsize=(14.0, 10.0))

        label_done = False
        for row in fpr.index:
            y_offset = 0
            if not label_done:
                plt.bar(
                    row, fpr[row], bar_width, bottom=y_offset, color="red", label="FPR"
                )
                y_offset += fpr[row]
                plt.bar(
                    row,
                    tpr[row],
                    bar_width,
                    bottom=y_offset,
                    color="green",
                    label="TPR",
                )
                y_offset += tpr[row]
                label_done = True
            else:
                plt.bar(row, fpr[row], bar_width, bottom=y_offset, color="red")
                y_offset += fpr[row]
                plt.bar(row, tpr[row], bar_width, bottom=y_offset, color="green")
                y_offset += tpr[row]

        plt.xlabel("Tau")
        plt.ylabel("FPR/TPR")
        plt.title(f"{label}")

        plt.legend()

        Plotter.save_and_display(fig_file, None, show)

    @staticmethod
    def precision_recall(
        data: pd.DataFrame,
        title: str,
        fig_file: str,
        line_enabled: bool,
        show: bool = False,
    ) -> None:
        """
        Method generates precision/recall
        Args:
            data: data set containing recall: recall vector values (X axis)
                 and precision vector values (Y axis)
            title: title of the precision/recall graph
            fig_file: File name
            line_enabled: flag to determine how plot will be plotted
            show: graph display on screen flag
        Returns:
            <None>
        """

        # Plot default 50% (random) model for comparison
        plt.axhline(y=0.5, color="black", linestyle="--", label="No skill")
        plt.grid(True, color="lightgrey", linestyle="-", linewidth=0.5)

        data_s = data.copy(deep=False)

        data_s.drop_duplicates(
            subset=[data.columns[0], data.columns[1]], keep="first", inplace=True
        )
        data_s.sort_values(
            by=[data.columns[0], data.columns[1]], ascending=[True, False], inplace=True
        )

        # Plot FPR/TPR points
        plt.scatter(
            data_s.loc[:, data_s.columns[0]],
            data_s.loc[:, data_s.columns[1]],
            color="red",
            linewidth=1,
            zorder=2,
            marker="x"
        )

        # Plot precision-recall line
        if line_enabled:
            plt.plot(
                data_s.loc[:, data_s.columns[0]],
                data_s.loc[:, data_s.columns[1]],
                color="orange",
                linewidth=2,
                zorder=1,
                label="Algorithm",
            )  # label="Precision/Recall"

        plt.xlabel("Precision")
        plt.ylabel("Recall")

        plt.xlim(0.0, 1.05)
        plt.ylim(0.4, 1.05)

        plt.legend()

        plt.title("Precision/Recall" if "" == title else title)

        Plotter.save_and_display(fig_file, None, show)

    @staticmethod
    def aligned_plot(
        data_mir: pd.DataFrame,
        a_data_mir: pd.DataFrame,
        data_idx: int,
        a_data_idx: int,
        file_name: str,
        show: bool = False,
    ) -> None:
        """
        Method prints aligned plot for input data
        Args:
            data_mir: original mirrored data set input
            a_data_mir: mirrored data set with attack
            data_idx: original mirrored data index set input
            a_data_idx: mirrored data set index with attack
            file_name: name of the file
            show: figure show flag
        Returns:
            <None>
        """

        # Data engineering debug
        plt.annotate(
            f"data = {data_idx}",
            xy=(0.5, 0.5),
            xycoords="data",
            va="top",
            ha="right",
            bbox={"boxstyle": "round,pad=0.,rounding_size=0.2", "fc": "w"},
        )

        plt.annotate(
            f"a_data = {a_data_idx}",
            xy=(0.5, 0.5),
            xycoords="data",
            va="top",
            ha="right",
            bbox={"boxstyle": "round,pad=0.,rounding_size=0.2", "fc": "w"},
        )

        plt.plot(
            data_mir.index,
            data_mir.loc[:,],
            color="blue",
            label="Normal trace",
        )

        plt.plot(
            a_data_mir.index,
            a_data_mir.loc[:,],
            color="orange",
            label="Attack trace",
        )
        plt.grid(True, color="lightgrey", linestyle="-", linewidth=0.5)
        Plotter.save_and_display(f"{file_name}_dbg.png", None, show)

    @staticmethod
    class HandlerDoubleCircle(HandlerBase):
        """
        Custom legend handler
        """
        def __init__(self, color1, color2, **kwargs):
            self.color1 = color1
            self.color2 = color2
            super().__init__(**kwargs)

        def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
            # Create the first circle
            circle1 = Circle((width / 2, height / 2), width / 8, facecolor=self.color1, edgecolor='none')
            # Create the second circle on top
            circle2 = Circle((width / 2, height / 2), width / 12, facecolor=self.color2, edgecolor='none')
            return [circle1, circle2]

    @staticmethod
    def comp_roc(data: list,
                 fpr_label: str,
                 tpr_label: str,
                 optimal_point: list,
                 title: str,
                 fig_file: str,
                 limit: int,
                 show: bool = False) -> None:
        """
        Method generates graph with multiple ROC curves generated between
        training, detection and cycle detector detection.
        Args:
            data: list of pd.DataFrames responsible for training, detection, cd detection
                  tpr/fpr data sets
            fpr_label: False Positive Rate label name in all pd.DataFrames
            tpr_label: True Positive Rate label name in all pd.DataFrames
            optimal_point: List of optimal points that were selected to represent AD and CD models
            title: Plot title
            fig_file: figure file name where plot will be saved,
                      if empty figure will not be saved
            limit: cyber-security limit used in the algorithm evaluation
            show: test flag that determines whether to show
                  the figure on screen (True) or not.
        Returns:
            <None>
        """

        # Plot default 50% (random) model for comparison
        plt.plot([0, 1], [0, 1], color="black", linestyle="--", label="No skill")
        plt.grid(True, color="lightgrey", linestyle="-", linewidth=0.5)

        colors = ["blue", "red", "green"]
        dot_colors = ["xkcd:light green", "xkcd:light yellow", "xkcd:light blue"]
        label_limit = [f"Optimal detection at {limit*100:.2f}%", ""]
        label = ["AD", "CD"]

        idx = 0

        # Plot ROC curves
        for ds in data:
            if idx == 0:
                Plotter.plot_limit(plt, limit, label_limit[idx])
            fpr = ds.loc[:, fpr_label]
            tpr = ds.loc[:, tpr_label]
            auc_val = metrics.auc(fpr, tpr)
            plt.plot(
                fpr,
                tpr,
                color=colors[idx],
                label=f"{label[idx]} AUC: {auc_val:.2f}",
                linewidth=2,
                zorder=1,
            )

            # Plot FPR/TPR points
            plt.scatter(fpr, tpr, color=colors[idx], linewidth=1, zorder=2, marker="x")

            # Optimal point plotting
            plt.scatter([optimal_point[idx][0]],
                        [optimal_point[idx][1]],
                        color=colors[idx],
                        linewidth=4,
                        zorder=2,
                        marker=".")
            plt.scatter([optimal_point[idx][0]],
                        [optimal_point[idx][1]],
                        color=dot_colors[idx],
                        linewidth=2,
                        zorder=2,
                        marker=".")
                        # label=f"{label[idx]} optimal point")

            idx += 1
        # Create the custom legend handles
        custom_handle1 = Circle((0, 0), 1, facecolor=colors[0], edgecolor='none')
        custom_handle2 = Circle((0, 0), 1, facecolor=colors[1], edgecolor='none')

        # Routine that mixes default and custom-made handles and labels
        ax = plt.gca()

        handles, labels = ax.get_legend_handles_labels()

        # Combine custom and default handles and labels
        custom_handles = [custom_handle1, custom_handle2]
        custom_labels = ['Optimized AD model point', 'Optimized CD model point']
        combined_handles = handles + custom_handles
        combined_labels = labels + custom_labels

        # Add the combined legend to the plot
        ax.legend(combined_handles,
                  combined_labels,
                  handler_map={custom_handle1: Plotter.HandlerDoubleCircle(colors[0], dot_colors[0]),
                               custom_handle2: Plotter.HandlerDoubleCircle(colors[1], dot_colors[1])},
                  loc="lower right")

        plt.xlabel("FPR")
        plt.ylabel("TPR")

        plt.title(title)

        Plotter.save_and_display(fig_file, None, show)

    @staticmethod
    def comp_precision_recall(
        data: list,
        title: str,
        fig_file: str,
        show: bool = False,
    ) -> None:
        """
        Method generates graph with multiple precision curves generated between
        model anomaly detection and cycle detector detection.
        Args:
            data: list of pd.DataFrames responsible for training, detection, cd detection
                  tpr/fpr data sets
            title: Plot title
            fig_file: figure file name where plot will be saved,
                      if empty figure will not be saved
            show: test flag that determines whether to show
                  the figure on screen (True) or not.
        Returns:
            <None>
        """

        # Plot default 50% (random) model for comparison
        plt.axhline(y=0.5, color="black", linestyle="--", label="No skill")
        plt.grid(True, color="lightgrey", linestyle="-", linewidth=0.5)

        colors = ["blue", "red"]
        label = ["AD model", "CD model"]

        idx = 0

        # Plot ROC curves
        for ds in data:
            d = ds.sort_values(by=["recall", "prec"], ascending=[True, False])
            plt.plot(
                d.loc[:, "recall"],
                d.loc[:, "prec"],
                color=colors[idx],
                label=f"{label[idx]}",
                linewidth=2,
                zorder=1,
            )

            # Plot FPR/TPR points
            plt.scatter(
                d.loc[:, "recall"],
                d.loc[:, "prec"],
                color=colors[idx],
                linewidth=1,
                zorder=2,
                marker="x",
            )
            idx += 1

        plt.xlabel("Precision")
        plt.ylabel("Recall")

        plt.xlim(0.0, 1.05)
        plt.ylim(0.4, 1.05)

        plt.legend()

        plt.title("Precision/Recall" if "" == title else title)

        Plotter.save_and_display(fig_file, None, show)

    @staticmethod
    def metrics_over_tau(
        data: pd.Series, title: str, fig_file: str, max_tau: int, show: bool = False
    ) -> None:
        """
        Method plots metrics performance graph (per attack)
        over Tau computed for cycle detector (CD).
        Args:
            data  - pd.Series representing per cycle detection data set
            title - name of the graph
            fig_file - file where graph will be saved
            max_tau - maximum number of delay applied in CD model
                      (used to compute % of trace)
            show - flag to enable direct display on the screen
        Returns:
            <None>
        """

        def make_patch_spines_invisible(ax):
            ax.set_frame_on(True)
            ax.patch.set_visible(False)
            for sp in ax.spines.values():
                sp.set_visible(False)

        fig, ax1 = plt.subplots()
        ax1.grid(True, color="lightgrey", linestyle="-", linewidth=0.5)

        ax2 = ax1.twinx()

        eer_color = "crimson"
        ax2.tick_params(axis="y", labelcolor=eer_color)

        # ax2.spines["right"].set_position(("axes", -0.23))
        # make_patch_spines_invisible(ax2)
        # ax2.spines["right"].set_visible(True)

        ax1.set_ylim(0, 1)
        ax1.set_yticks(list(np.arange(0, 1.1, 0.1)))
        ax1.set_xlim(0, 100)
        ax1.set_xticks(list(np.arange(0, 110, 10)))

        ax2.set_ylim(1, 0)
        ax2.set_yticks(list(np.arange(1.0, -0.1, -0.1)))

        for column_name in ["index", "level_0"]:
            if column_name in data.columns:
                data.drop(columns=[column_name], inplace=True)

        n_data = data.copy(deep=False)
        for column_name in ["index", "level_0"]:
            if column_name in n_data.columns:
                n_data.drop(columns=[column_name], inplace=True)

        n_data.reset_index(inplace=True)
        n_data = n_data.iloc[1:-1, :]

        # Recompute % of tau value
        n_data = n_data.assign(x=(((n_data.loc[:, "tau"]) / max_tau) * 100).round(2))

        n_data.set_index("x", inplace=True)

        ax1.plot(
            n_data.index,
            n_data.loc[:, "acc"],
            color="green",
            label="Accuracy",
            linewidth=2,
        )

        ax1.plot(
            n_data.index,
            n_data.loc[:, "f1"],
            color="blue",
            label="F1 score",
            linewidth=2,
        )

        ax2.plot(
            n_data.index,
            n_data.loc[:, "eer"],
            color=eer_color,
            label="EER",
            linewidth=2,
        )

        ax2.hlines(
            0.5,
            0,
            100,
            color="black",
            linestyles="dashed",
            label="No skill",
            linewidth=1,
        )

        # Combine labels
        ax1_lbl, _ = ax1.get_legend_handles_labels()
        ax2_lbl, _ = ax2.get_legend_handles_labels()

        lns = ax1_lbl + ax2_lbl
        labs = [l.get_label() for l in lns]

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax1.set_xlabel("% Cycle")
        ax1.set_ylabel("Accuracy, F1 score")

        # ax2.yaxis.set_label_coords(-0.28, 0.5)
        ax2.set_ylabel(
            "EER", color=eer_color
        )  # we already handled the x-label with ax1

        plt.legend(lns, labs, loc=0)
        plt.title(title)
        plt.tight_layout()

        Plotter.save_and_display(fig_file, None, show)

    @staticmethod
    def eer_over_tau(
        data: pd.Series, title: str, fig_file: str, max_tau: int, show: bool = False
    ) -> None:
        """
        Method plots metrics eer performance graph (per attack)
        over Tau computed for cycle detector (CD).
        Args:
            data  - pd.Series representing per cycle detection data set
            title - name of the graph
            fig_file - file where graph will be saved
            max_tau - maximum number of delay applied in CD model
                      (used to compute % of trace)
            show - flag to enable direct display on the screen
        Returns:
            <None>
        """
        plt.ylim(
            1.1,
            -0.1,
        )
        plt.yticks(list(np.arange(0, 1.1, 0.1)))
        plt.xlim(0, 100)
        plt.xticks(list(np.arange(0, 110, 10)))
        plt.hlines(
            0.5,
            0,
            100,
            color="black",
            linestyles="dashed",
            label="No skill",
            linewidth=1,
        )
        plt.grid(True, color="lightgrey", linestyle="-", linewidth=0.5)

        n_data = data.copy(deep=False)
        for column_name in ["index", "level_0"]:
            if column_name in n_data.columns:
                n_data.drop(columns=[column_name], inplace=True)

        n_data.reset_index(inplace=True)
        n_data = n_data.iloc[1:-1, :]

        # Recompute % of tau value
        n_data = n_data.assign(x=(((n_data.loc[:, "tau"]) / max_tau) * 100).round(2))

        n_data.set_index("x", inplace=True)

        plt.plot(
            n_data.index, n_data.loc[:, "eer"], color="purple", label="EER", linewidth=2
        )

        # Add some text for labels, title and custom x-axis tick labels, etc.
        plt.ylabel("EER")
        plt.title(title)
        plt.xlabel("% Cycle")
        plt.legend(loc="best")

        plt.tight_layout()

        Plotter.save_and_display(fig_file, None, show)

    @staticmethod
    def d2_over_tau(
        data: pd.Series, title: str, fig_file: str, max_tau: int, show: bool = False
    ) -> None:
        """
        Method plots metrics Δ2 performance graph (per attack)
        over Tau computed for cycle detector (CD).
        Args:
            data  - pd.Series representing per cycle detection data set
            title - name of the graph
            fig_file - file where graph will be saved
            max_tau - maximum number of delay applied in CD model
                      (used to compute % of trace)
            show - flag to enable direct display on the screen
        Returns:
            <None>
        """
        plt.xlim(0, 100)
        plt.xticks(list(np.arange(0, 110, 10)))
        plt.grid(True, color="lightgrey", linestyle="-", linewidth=0.5)

        n_data = data.copy(deep=False)
        for column_name in ["index", "level_0"]:
            if column_name in n_data.columns:
                n_data.drop(columns=[column_name], inplace=True)

        n_data.reset_index(inplace=True)
        n_data = n_data.iloc[1:-1, :]

        # Recompute % of tau value
        n_data = n_data.assign(x=(((n_data.loc[:, "tau"]) / max_tau) * 100).round(2))

        n_data.set_index("x", inplace=True)

        plt.plot(
            n_data.index,
            n_data.loc[:, "d2_min"],
            color="blue",
            label="Δ2 min",
            linewidth=2,
        )

        plt.plot(
            n_data.index,
            n_data.loc[:, "d2_avg"],
            color="purple",
            label="Δ2 avg",
            linewidth=2,
            linestyle="--",
        )

        plt.plot(
            n_data.index,
            n_data.loc[:, "d2_max"],
            color="red",
            label="Δ2 max",
            linewidth=2,
        )

        # Add some text for labels, title and custom x-axis tick labels, etc.
        plt.ylabel("Δ2 delay")
        plt.title(title)
        plt.xlabel("% Cycle")
        plt.legend(loc="best")
        plt.grid(True, color="lightgrey", linestyle="-", linewidth=0.5)
        plt.tight_layout()

        Plotter.save_and_display(fig_file, None, show)

    @staticmethod
    def d1_over_tau(data: pd.Series,
                    title: str,
                    fig_file: str,
                    line_enabled: bool,
                    show: bool = False
    ) -> None:
        """
        Method plots metrics Δ2 performance graph (per attack)
        over Tau computed for cycle detector (CD).
        Args:
            data: pd.Series representing per cycle detection data set
            title: name of the graph
            fig_file: file where graph will be saved
            line_enabled: bool flag to determine way to display the points
            show: flag to enable direct display on the screen
        Returns:
            <None>
        """
        plt.grid(True, color="lightgrey", linestyle="-", linewidth=0.5)

        data_cp = data.loc[:, ["fpr", "tpr", "d1_min", "d1_avg", "d1_max"]].copy(deep=True)
        data_cp = data_cp.iloc[1:-1, :]

        data_cp.reset_index(inplace=True)
        data_cp.sort_values(by=["fpr", "tpr"], inplace=True)

        plot_func = plt.scatter
        if line_enabled:
            plot_func = plt.plot

        plot_func(data_cp.index,
                  data_cp.loc[:, "d1_min"],
                  color="blue",
                  label="Δ1 min",
                  linewidth=2)

        plot_func(data_cp.index,
                  data_cp.loc[:, "d1_avg"],
                  color="purple",
                  label="Δ1 avg",
                  linewidth=2,
                  linestyle="--")

        plot_func(data_cp.index,
                  data_cp.loc[:, "d1_max"],
                  color="red",
                  label="Δ1 max",
                  linewidth=2)

        # Add some text for labels, title and custom x-axis tick labels, etc.
        plt.ylabel("Δ1 delay")
        plt.ylim([0, 2 if data_cp.loc[:, "d1_max"].sum() == 0
                       else data_cp.loc[:, "d1_max"].max() * 1.05])

        plt.grid(True, color="lightgrey", linestyle="-", linewidth=0.5)

        plt.title(title)
        plt.xlabel("Parameter set/threshold")
        plt.legend(loc="best")

        plt.tight_layout()

        Plotter.save_and_display(fig_file, None, show)

    @staticmethod
    def annotate_heatmap(
        im,
        im_color: list,
        valfmt: str = "{x:.2f}",
        textcolors: list = ("white", "black"),
        thresholds: list = [-5, 30],
        font_size: str = "large",
        log=None,
        **textkw,
    ) -> None:
        """
        A function to annotate a heatmap.
        Args:
            im: The AxesImage to be labeled.
            im_color: image color for each element
            valfmt: The format of the annotations inside the heatmap.  This should either
                    use the string format method, e.g. "$ {x:.2f}", or be a
                    `matplotlib.ticker.Formatter`.  Optional.
            textcolors: A pair of colors.  The first is used for values below a threshold,
                        the second for those above.  Optional.
            thresholds: Value in data units according to which the colors from text colors are
                        applied.  If None (the default) uses the middle of the colormap as
                        separation.  Optional.
            font_size: size of the font (small, medium, large).
            log: logger object.
            **textkw: All other arguments are forwarded to each call to `text` used to create
                      the text labels.
        Returns:
            <list> - list of texts added to given heatmap
        """
        # From example:
        # https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
        data = im.get_array()

        # Set default alignment to center, but allow it to be
        # overwritten by textkw.
        kw = {"horizontalalignment": "center", "verticalalignment": "center"}
        kw.update(textkw)

        # Get the formatter in case a string is supplied
        fmt = matplotlib.ticker.StrMethodFormatter(valfmt)

        # TODO: Automatic threshold adjustments
        # Loop over the data and create a `Text` for each "pixel".
        # Change the text's color depending on the data.
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):

                text_color = textcolors[1]

                if len(thresholds) == 1 and abs(data[i, j]) > abs(thresholds[0]):
                    text_color = textcolors[0]
                if len(thresholds) == 2:
                    if data[i, j] < thresholds[0] or data[i, j] > abs(thresholds[1]):
                        text_color = textcolors[0]

                kw.update(color=text_color)

                im.axes.text(
                    j,
                    i,
                    fmt(data[i, j], None),
                    fontsize=font_size,
                    fontweight="bold",
                    **kw,
                )

    @staticmethod
    def get_algo_names(data: pd.Series) -> list:
        """
        Method prunes names of algorithms
        Args:
            data: input data set containing algorithm column
        Returns:
            <list>: list of unique and pruned algorithm names
        """
        algorithm_names = data.unique().tolist()

        return [x.split("_")[1:] if "_" in x else x for x in algorithm_names]

    @staticmethod
    def sum_heatmap(algo_names_map: dict,
                    data: pd.DataFrame,
                    title: str,
                    fig_file: str,
                    log: object,
                    label: str = "auc",
                    show: bool = False) -> None:
        """
        Detection performance heatmap that is summarizing
        effectiveness of attacks vs detection algorithms (AD model) and de-noiser CD
        Args:
            data  - pd.Series representing per cycle detection data set
            title - name of the graph
            fig_file - file where graph will be saved
            log - logger object
            label - name of the given label to be used in plotting
            show - flag to enable direct display on the screen
        Returns:
            <None>
        """
        try:
            # X and Y labels
            attack_names = data["attack"].unique().tolist()
            algo_names = data["algo"].unique().tolist()

            algorithm_names = Plotter.get_algo_names(data["algo"])

            # Image selection
            im_ad = data[data["type"] == "ad"].loc[:, ["algo", "attack", label]]
            im_ad = im_ad.loc[:, ["algo", "attack", label]]
            im_ad.set_index("attack", inplace=True)

            im_cd = data[data["type"] == "cd"].loc[:, ["algo", "attack", label]]
            im_cd = im_cd.loc[:, ["algo", "attack", label]]
            im_cd.set_index("attack", inplace=True)

            # Setup maximum size to specific fit on the slide
            max_height = len(attack_names)
            max_height = max_height if max_height < 7 else 7

            # Figure size
            fig = plt.figure(figsize=((len(algorithm_names) * 2) + 1, max_height))

            grid = ImageGrid(
                fig,
                111,
                nrows_ncols=(len(attack_names), len(algorithm_names)),
            )

            scale_grid = ImageGrid(fig, 111, nrows_ncols=(1, 1))

            # Set the same color map across detection bars and scalar map
            c_map = Plotter.cm_ryg if "eer" != label else Plotter.cm_ryg_r

            norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
            sm = plt.cm.ScalarMappable(cmap=c_map, norm=norm)
            sm.set_array([])

            grid_cnt = 0
            for attack_name in attack_names:
                attack_cnt = 0
                for algo_name in algo_names:
                    ax = grid[grid_cnt]
                    if attack_cnt == 0:
                        ax.set_yticks([0], [attack_name.upper()])

                    # X axis
                    ax.set_xlabel(algo_names_map[algo_name.upper()].upper())
                    ax.set_xticks([0, 1], ["AD", "CD"])

                    # Combination of AD(first) + CD (last) for given
                    # algo + attack
                    # TODO: Fix the problem in case when data coming from ad or cd section of the ADEF is missing
                    #  In that case replace it with 0 value.

                    ad_val = im_ad[im_ad["algo"] == algo_name]
                    cd_val = im_cd[im_cd["algo"] == algo_name]

                    def_val = [1.0] if label == "eer" else [0.5]

                    if ad_val.empty:
                        for at_name in attack_names:
                            ad_val = pd.concat([ad_val,
                                                pd.DataFrame({"algo": [algo_name],
                                                                    label: def_val},
                                                             index=[at_name])])
                    if cd_val.empty:
                        for at_name in attack_names:
                            cd_val = pd.concat([cd_val,
                                                pd.DataFrame({"algo": [algo_name],
                                                                    label: def_val},
                                                             index=[at_name])])

                    im = pd.concat([ad_val, cd_val]).loc[attack_name, label]
                    if isinstance(im, pd.Series):
                        im = im.to_frame()
                        im.reset_index(inplace=True)
                        column_name = "attack"
                        if "attack" not in im.columns.tolist():
                            column_name = "index"
                        im.set_index(column_name, inplace=True)
                        im = im.T
                    elif (isinstance(im, numpy.float64)
                          or isinstance(im, numpy.float32)
                          or isinstance(im, numpy.float)
                          or isinstance(im, float)):
                        im = pd.DataFrame([im])

                    imx = ax.imshow(im, cmap=c_map, norm=norm)

                    # Add all numeric labels to the graph
                    Plotter.annotate_heatmap(imx,
                                             im,
                                             font_size="large",
                                             textcolors=["white", "black"],
                                             thresholds=[0.3, 0.7],
                                             log=log)
                    grid_cnt += 1
                    attack_cnt += 1

            # ------------------------ color bar ----------------------------
            ax = scale_grid[0]

            ax.axis("off")
            y_ticks = np.linspace(0, 1, 6).tolist()

            # fig.subplots_adjust(wspace=0.05)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size=15/fig.dpi, pad=-5/fig.dpi)

            fig.colorbar(sm,
                         cax,
                         orientation="vertical",
                         ticks=y_ticks,
                         drawedges=False,
                         spacing="uniform")

            # Final title and Y axis annotation
            fig.suptitle(title, y=1.05 if len(attack_names) == 1 else 0.97)
            fig.supylabel("Attack type", x=0.05)
            #, ha="right")  # , x=0.15

            Plotter.save_and_display(fig_file, "tight", show)

        except Exception as ex:
            log.exception(ex)

    @staticmethod
    def c2gl(data: pd.DataFrame) -> pd.DataFrame:
        """
        Method converts experiment data results into delta gain/loss
        dataframe to display.
        Args:
            data: data frame containing experimental results
        Returns:
            <pd.DataFrame>
        """
        sel_columns = ["auc", "acc", "f1", "eer"]
        data_diff = pd.DataFrame()

        for algo in data["algo"].unique().tolist():
            data_sc = data[(data["type"] == "cd") & (data["algo"] == algo)].copy(
                deep=True
            )
            data_sc = data_sc.loc[:, ["attack"] + sel_columns]
            data_sc.reset_index(inplace=True)
            data_ad = data[(data["type"] == "ad") & (data["algo"] == algo)].copy(
                deep=True
            )
            data_ad = data_ad.loc[:, ["attack"] + sel_columns]
            data_ad.reset_index(inplace=True)

            data_sc.loc[:, sel_columns] = (
                data_sc.loc[:, sel_columns]
                .astype(float)
                .subtract(data_ad.loc[:, sel_columns].astype(float))
            )
            data_sc.loc[:, sel_columns] = data_sc.loc[:, sel_columns].round(2)
            data_sc["algo"] = algo

            data_sc.fillna(0.0, inplace=True)
            data_diff = pd.concat([data_diff, data_sc])

        # data_diff.set_index("attack", inplace=True)
        data_diff.drop("index", axis=1, inplace=True)

        return data_diff

    @staticmethod
    def get_color_indices(pos_len: int, ratio: int, is_negative: bool = False) -> list:
        """
        Method fetch color indices
        Args:
            pos_len:
            ratio:
            is_negative:
        Returns:
            <list>
        """
        idxs = []
        if ratio < 1:
            return idxs
        incr = int(pos_len / ratio)
        for i in range(0, ratio):
            idx = incr * (i + 1)
            idx = idx if idx < pos_len else pos_len
            idxs.append(idx)
        if is_negative:
            idxs.reverse()
        return idxs

    @staticmethod
    def gain_loss_heatmap(algo_names_map: dict,
                          data: pd.DataFrame,
                          title: str,
                          fig_file: str,
                          log: object,
                          label: str = "auc",
                          reverse_colors: bool = False,
                          show: bool = False) -> None:
        """
        Heatmap that is summarizing gain and loss between AD model and de-noiser CD
        algorithm within N% limit.
        Args:
            data: pd.Series representing per cycle detection data set
            title: name of the graph
            fig_file: file where graph will be saved
            log: logger object
            label: name of the given label to be used in plotting
            reverse_colors: invert colors to help interpret the graph result
            show: flag to enable direct display on the screen
        Returns:
            <None>
        """
        # X and Y labels
        attack_names = data["attack"].unique().tolist()

        # Convert data to delta format
        data_sc = Plotter.c2gl(data)

        # Image selection
        im_gain_loss = data_sc.loc[:, ["attack", "algo", label]]
        im_gain_loss.set_index("attack", inplace=True)

        # Setup maximum size to specific fit on the slide
        max_height = (len(attack_names)) * 1
        max_height = max_height if max_height < 7 else 7

        # Figure size
        fig = plt.figure(figsize=((len(algo_names_map.keys())) + 2, max_height))
        grid = ImageGrid(fig,
                    111,
                         nrows_ncols=(len(attack_names), 1))
        scale_grid = ImageGrid(fig, 111, nrows_ncols=(1, 1))

        const_scale = 1
        v_min = data_sc[label].min().min()
        v_min = const_scale * (round(v_min // const_scale))
        v_max = data_sc[label].max().max()
        v_max = const_scale * (math.ceil(v_max / const_scale))

        if v_min == 0 and v_max == 0:
            v_max = const_scale
            v_min = -const_scale

        norm = matplotlib.colors.Normalize(vmin=v_min, vmax=v_max)

        # Color map selection
        c_map = Plotter.cm_ryg if "eer" != label else Plotter.cm_ryg_r
        if v_min < 0 and v_max == 0:
            c_map = Plotter.cm_ry if "eer" != label else Plotter.cm_yg_r
        elif v_min == 0 and v_max > 0:
            c_map = Plotter.cm_yg if "eer" != label else Plotter.cm_ry_r

        sm = plt.cm.ScalarMappable(cmap=c_map, norm=norm)
        ticks = np.linspace(v_min, v_max, 5)

        grid_cnt = 0
        for attack_name in attack_names:
            attack_cnt = 0

            ax = grid[grid_cnt]
            ax.set_yticks([0],
                          [attack_name.upper()])

            # X axis
            ax.set_xticks(range(0,
                          len(algo_names_map.keys())),
                          [x for x in algo_names_map.values()])

            # Plot difference for all algorithms at once
            im = im_gain_loss.loc[attack_name, label]
            if isinstance(im, pd.Series):
                im = im.to_frame()
            elif isinstance(im, float):
                im = im_gain_loss.loc[:, label].to_frame()
                im.reset_index(inplace=True)
                im = im[im["attack"] == attack_name]
                im.set_index("attack", inplace=True)

            im.reset_index(inplace=True)
            im.set_index("attack", inplace=True)
            im = im.T
            imx = ax.imshow(im, cmap=c_map, norm=norm)

            # An example of the threshold
            thresholds = []

            if min(ticks) < 0:
                thresholds.append(-0.7)

            if max(ticks) > 0:
                thresholds.append(0.7)

            # Add all numeric labels to the graph
            Plotter.annotate_heatmap(
                imx,
                im,
                font_size="large",
                textcolors=["white", "black"],
                thresholds=thresholds,
                log=log,
            )
            grid_cnt += 1
            attack_cnt += 1

        # -------------------------- Color bar --------------------------
        ax = scale_grid[0]
        ax.axis("off")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size=15/fig.dpi, pad=-5/fig.dpi) # "5%")

        fig.colorbar(sm,
                     cax,
                     pad=0,
                     orientation="vertical",
                     ticks=ticks,
                     drawedges=False,
                     spacing="uniform")

        fig.supylabel("Attack type")
        fig.suptitle(title, y=1.05 if len(attack_names) == 1 else 0.97)

        Plotter.save_and_display(fig_file, "tight", show)

    @staticmethod
    def meta_per_cycle(data: pd.DataFrame,
                       title: str,
                       x_label: str,
                       fig_file: str,
                       log: object,
                       label: str = "auc",
                       show: bool = False) -> None:
        """
        Method generates combined graphs of the best AUC
        performance for given attack scenario depending
        on number of production cycles used (amount of data).
        There are two graph series for AD model and CD parts for each
        probability set that was set in the experiment.
        Args:
            data: pd.Series representing per cycle detection data set
            title: name of the graph
            x_label: X label
            fig_file: file where graph will be saved
            log: logger object
            label: data label used in plotting
            show: flag to enable direct display on the screen
        Returns:
            <None>
        """
        attack_names = data["attack"].unique().tolist()
        algo_names = data["algo"].dropna().unique().tolist()
        number_of_cycles = data["cycle"].dropna().unique().tolist()

        for algo_name in algo_names:
            scale = "linear"
            y = np.arange(0, 1.1, 0.1).round(2)
            # AD model/CD parts
            for attack_name in attack_names:
                fig = plt.figure()
                plt.xlabel(x_label)
                plt.xticks(number_of_cycles, number_of_cycles)
                plt.xscale(scale)
                plt.ylim(0, 1.1)
                plt.yticks(y, y)
                plt.ylabel(label.upper())
                plt.grid(True, color="lightgrey", linestyle="-", linewidth=0.5)
                lbl_track = []

                for subset in ["ad", "cd"]:
                    ds = data[
                        (data["type"] == subset)
                        & (data["algo"] == algo_name)
                        & (data["attack"] == attack_name)
                    ]
                    lbl = f"{subset.upper()} {algo_name.upper()}"
                    if lbl not in lbl_track:
                        lbl_track.append(lbl)
                    else:
                        lbl = ""

                    plt.plot(
                        ds.loc[:, "cycle"].tolist(),
                        ds.loc[:, label].tolist(),
                        label=lbl,
                        color="blue" if "ad" == subset else "red"
                    )

                    # Plot points
                    plt.scatter(
                        ds.loc[:, "cycle"].tolist(),
                        ds.loc[:, label].tolist(),
                        linewidth=1,
                        zorder=2,
                        marker="x",
                        color="blue" if "ad" == subset else "red"
                    )

                plt.legend(loc="lower right")
                plt.title(title)

                fig.tight_layout()
                Plotter.save_and_display(
                    f"{fig_file[:-4]}_{algo_name}_{scale[:3]}_{attack_name}_perf.png",
                    None,
                    show,
                )

    @staticmethod
    def meta_per_attack(data: pd.DataFrame,
                        title: str,
                        fig_file: str,
                        log: object,
                        label: str = "auc",
                        show: bool = False) -> None:
        """
        Per attack number graph comparison between AD model and CD
        (model knowledge - known vs. unknown attacks).
        Args:
            data  - pd.Series representing per cycle detection data set
            title - name of the graph
            fig_file - file where graph will be saved
            log - logger object
            label - data label used in plotting
            show - flag to enable direct display on the screen
        Returns:
            <None>
        """
        attack_names = data[label].unique().tolist()
        algorithm_names = Plotter.get_algo_names(data["algo"])
        number_of_attacks = data["attack_no"].dropna().unique().tolist()

        for algo_name in algorithm_names:
            scale = "linear"
            y = np.arange(0, 1.1, 0.1).round(2)
            # AD model/CD parts
            # TODO: Assign the same color to the same attack and algorithm combination
            fig = plt.figure()
            plt.xlabel("Number of attacks")
            plt.xticks(number_of_attacks, number_of_attacks)
            plt.xscale(scale)
            plt.ylim(0, 1.1)
            plt.yticks(y, y)
            plt.ylabel(label.upper())
            plt.grid(True, color="lightgrey", linestyle="-", linewidth=0.5)
            lbl_track = []
            for subset in ["ad", "cd"]:
                for attack_name in attack_names:
                    ds = data[
                        (data["type"] == subset)
                        & (data["algo"] == algo_name)
                        & (data["attack"] == attack_name)
                    ]
                    lbl = f"{subset.upper()} {algo_name.upper()}"
                    if lbl not in lbl_track:
                        lbl_track.append(lbl)
                    else:
                        lbl = ""
                    plt.plot(
                        ds.loc[:, "attack_no"].tolist(),
                        ds.loc[:, label].tolist(),
                        label=lbl,
                        color="blue" if "ad" == subset else "red"
                    )

                    # Plot points
                    plt.scatter(
                        ds.loc[:, "attack_no"].tolist(),
                        ds.loc[:, label].tolist(),
                        linewidth=1,
                        zorder=2,
                        marker="x",
                        color="blue" if "ad" == subset else "red"
                    )

            plt.legend(loc="lower right")
            plt.title(title)

            fig.tight_layout()
            Plotter.save_and_display(
                f"{fig_file[:-4]}_{algo_name}_{scale[:3]}_perf.png",
                None,
                show,
            )

    @staticmethod
    def generate_graphs(cfg: dict,
                        out_ts: pd.DataFrame,
                        roc_cd: pd.DataFrame,
                        roc_data: pd.DataFrame,
                        auc_tab: pd.DataFrame,
                        optimal_point: list,
                        metric_vals: dict,
                        file_name: str,
                        limit: int,
                        max_tau: int,
                        log2f: Log) -> None:
        """
        Method generates all the graphs for the cycle detector
        experiment.
        Args:
            cfg: Configuration of the experiment
            out_ts: Detection per point data set generated as detection algorithm output
            roc_cd: Cycle detector data used generated during Cycle Detector
                    component activity
            roc_data: Receiver Operational Characteristics' data frame
            auc_tab: AUC over Tau performance for each of detection algorithms
            optimal_point: Optimal points that are representing AD and CD models
            metric_vals: Given anomaly detection precomputed metrics
            file_name: Name of the file where all graphs should be written (pattern_name)
            limit: Manufacturing anomaly detection safety limit
            max_tau: maximum number of delay applied in CD model
                      (used to compute % of trace)
            log2f: logger object
        Returns:
            <None>
        """
        # THB has additional element in file name so this is accounted for
        # when extracting detector name from file name

        names = IterHelper.extract_names(file_name)
        detector_name = names["algo_name"].upper()
        attack_name = names["attack_name"].upper()
        prefix_name = f"Model: {detector_name} - Attack:{attack_name} -"

        # Apply correct filtering to make sure that the graphs are not "wavy"
        Plotter.cf_insight(
            roc_data.loc[roc_data.index[1] : roc_data.index[-1],].sort_values(["tau"])[
                1:
            ],
            f" - CD - CF insight",
            f"{file_name[:-8]}cfi.png",
        )

        Plotter.tpr_fpr(
            roc_data.loc[:, "fpr"],
            roc_data.loc[:, "tpr"],
            f"{prefix_name} CD TPR/FPR",
            f"{file_name[:-4]}_tpr_fpr_cd.png",
        )

        # Per time point detection
        if out_ts is not None:
            Plotter.input_ts_color(
                None,
                out_ts,
                f"{prefix_name} Input attack time series",
                cfg,
                f"{file_name[:-4]}_ts.png",
                False,
                log2f,
            )
            Plotter.detection_ts(
                out_ts,
                f"{prefix_name} Detection over time",
                f"{file_name[:-4]}_det_ts.png",
                False,
            )

        # Per cycle detection
        if roc_cd is not None:
            Plotter.detection_cd(
                roc_cd,
                f"{prefix_name} Detection per cycle",
                f"{file_name[:-4]}_det_cd.png",
                False,
            )

        if auc_tab is not None:
            graph_data = auc_tab.sort_values(by=["tau"])
            Plotter.d2_over_tau(graph_data,
                           f"{prefix_name} Δ2 over Tau",
                         f"{file_name[:-4]}_d2.png",
                                max_tau,
                          False)
            Plotter.metrics_over_tau(graph_data,
                                f"{prefix_name} Metrics over Tau",
                              f"{file_name[:-4]}_tau_c.png",
                                     max_tau,
                               False)

        if auc_tab is not None:
            roc_cd = auc_tab.loc[:, ["fpr", "tpr"]].copy(deep=False)
            roc_cd.sort_values(by=["fpr", "tpr"], inplace=True)
            # ROC
            Plotter.roc_2d(
                roc_cd,
                metric_vals,
                f"{prefix_name} ROC",
                f"{file_name[:-4]}_roc.png",
                limit,
                "search_param" in roc_data[roc_data["cfg"] != 0.0]["cfg"].keys(),
            )

            # Extract previous AD model experiment results for comparison
            fpath = file_name[: file_name.rfind("/")]
            fpath = fpath[: fpath.rfind("/")]
            f_name = file_name[file_name.rfind("/"):]

            rcm_file = f"{fpath}/detection{f_name[:-4]}_roc.csv"

            if os.path.isfile(rcm_file):
                ad_roc_data = pd.read_csv(rcm_file)
                if "fpr" in ad_roc_data.columns and "tpr" in ad_roc_data.columns:
                    # ROC comparison with between AD and CD algorithms
                    Plotter.comp_roc(
                        [roc_data, ad_roc_data],
                        "fpr",
                        "tpr",
                        optimal_point,
                        f"{prefix_name} ROC comparison",
                        f"{file_name[:-4]}_auc_diff.png",
                        limit,
                    )
                else:
                    log2f.error(f"Problem with AD model data: |{rcm_file}| =>{ad_roc_data.columns}")

                if "prec" in ad_roc_data.columns and "recall" in ad_roc_data.columns:
                    # ROC comparison with between AD and CD algorithms
                    Plotter.comp_precision_recall(
                        [roc_data, ad_roc_data],
                        f"{prefix_name} Precision/Recall comparison",
                        f"{file_name[:-4]}_prec_recall_diff.png",
                    )
                else:
                    log2f.error(
                        f"Problem with AD model data: |{rcm_file}| =>{ad_roc_data.columns}"
                    )

            else:
                log2f.error(f"RCM file: {rcm_file} doesn't exists.")

    @staticmethod
    def box_plot(data: pd.DataFrame,
                 title: str,
                 fig_file: str,
                 log: object,
                 show: bool = False) -> None:
        """
        Method generates box plot.
        Args:
            data  - pd.Series representing per cycle detection data set
            title - name of the graph
            fig_file - file where graph will be saved
            log - logger object
            show - flag to enable direct display on the screen
        Returns:
            <None>
        """
        # TODO: Further examples can be found at: https://matplotlib.org/3.1.1/gallery/statistics/boxplot.html
        red_diamond = dict(markerfacecolor="r", marker="D")
        medianlineprops = dict(linestyle="-", linewidth=1.5, color="purple")
        medianprops = dict(linestyle="-", linewidth=1.5, color="firebrick")

        fig, ax = plt.subplots()
        positions = np.arange(1, 1 + (len(data.columns)) * 0.1, 0.1).round(1)
        positions = (
            positions[: len(data.columns) - len(positions)]
            if len(positions) > len(data.columns)
            else positions
        )
        ax.boxplot(
            data,
            vert=False,
            showmeans=True,
            meanline=True,
            medianprops=medianprops,
            flierprops=red_diamond,
            positions=positions,
            widths=0.05,
        )
        ax.set_xlim(0, 1.05)
        ax.set_ylim(positions[0] - 0.05, positions[-1] + 0.05)
        ax.set_xticks(np.arange(0, 1.1, 0.1).round(1))
        ax.set_yticklabels(list(data.columns))
        plt.title(title)

        Plotter.save_and_display(
            fig_file,
            None,
            show,
        )
