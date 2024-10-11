""" LaTeX generator for experiment information """

from __future__ import annotations

import json

import pandas as pd

from tools.log import Log


class ACMTableHelper:
    """
    ACM table format helper
    """
    @staticmethod
    def header(caption: str, header_names: list, label: str) -> str:
        """
        Method creates customized LaTeX ACM formatted table header.
        Args:
            caption:
            header_names:
            label:
        Returns:
            <str>: LaTeX header
        """
        column_flag = 'c |'*len(header_names)
        column_flag = column_flag[:-1]
        column_names = " &".join(header_names)
        column_names = column_names[: -1]

        header = [
            "\\begin{table}\n",
            f"  \\caption{{{caption}}}\n",
            f"  \\label{{tab:{label}}}\n",
            f"  \\begin{{tabular}}{{{column_flag}}}\n",
            "    \\toprule\n",
            f"    {column_names}\\\\\n",
            "    \\midrule\n",
        ]

        return "".join(header)

    @staticmethod
    def footer() -> str:
        """
        Method creates customized LaTeX ACM formatted table footer.
        Args:
            <None>
        Returns:
            <str>: LaTeX footer
        """
        footer = ["   \\bottomrule\n",
                  " \\end{tabular}\n",
                  "\\end{table}\n"]

        return "".join(footer)


class LaTeXGenerator:
    """
    LaTeX Generator
    """

    def __init__(self, cfg: dict, logger: Log) -> None:
        """
        CTOR
        Args:

        Returns:
            <None>
        """
        self.cfg = cfg
        self.log = logger

        self.timing_metrics = ["Training time", "Evaluation time", "Cycle detector time"]
        self.metric_types = ["auc", "acc", "f1", "eer", "d1", "d2"]

    def get_metric_lines(self, data: dict) -> str:
        lines = ""
        for metric_type in self.metric_types:
            metric_line = f"\t & {metric_type} &"

            if metric_type not in data.keys():
                continue
            val = data[metric_type]
            if val is None:
                continue
            if metric_type == "auc" or metric_type == "eer":
                if isinstance(val, pd.Series):
                    metric_line += f"{val.tolist()[0]:.2f} & "
                else:
                    metric_line += f"{val:.2f} & "
            else:
                if isinstance(val, str) or isinstance(val, list):
                    metric_line += f"{val} & "
                elif isinstance(val, float):
                    metric_line += f"{val:.2f} & "

            '''
            elif metric_type == "d1":
                metric_line += (
                    f"[{data[metric_type + '_min']:.2f}, "
                    # f" {val.loc[val.index[0], 'avg']:.2f},"
                    f" {data[metric_type + '_max']:.2f}] & "
                )
            elif metric_type == "d2":
                # Precompute max, min and average across all column
                # that was executed from whole experiment.
                metric_line += (
                    f"[{data[metric_type + '_min']:.2f},"
                    # f" {avg_val:.2f},"
                    f" {data[metric_type + '_max']:.2f}] & "
                )
            '''

            lines += metric_line[:-2] + "\\\\\n"

        lines += "\t\\hline\n"
        return lines

    def gen_metric_table(self, data: dict, caption: str, file_name: str) -> None:
        """
        Method generates LaTeX formatted metrics table
        Args:
            data: dictionary of pd.DataFrames with experimental results
            caption: table caption
            file_name: name of the LaTeX output file
        Returns:
            <None>
        """
        # TODO: Rethink design of the table
        #  would it be better to split it onto two tables (metrics and timing?)

        methods = list(data.keys())

        # Table formatting routine
        lines = ACMTableHelper.header(caption, methods, caption)

        # Iterate through correctly aligned data set to generate rows inside LaTEX table
        for algo_type in data.keys():
            for attack_name in list(data[algo_type].keys()):
                lines += self.get_metric_lines(data[algo_type][attack_name])

        lines += ACMTableHelper.footer()

        # Write LaTeX table into file
        with open(file_name, "w", encoding="utf-8") as f2w:
            f2w.writelines(lines)
        self.log.debug(f"TeX saved to: {file_name}")

    def gen_time_table(self, data: dict, caption: str, file_name: str) -> None:
        """
        Method generates LaTeX formatted time table
        Args:
            data: dictionary of pd.DataFrames with experimental results
            caption: table caption
            file_name: name of the LaTeX output file
        Returns:
            <None>
        """
        lines = ACMTableHelper.header("test", ['test'], "time_table_x")

        # TODO: Add each line for the middle section of the table

        lines += ACMTableHelper.footer()

        # Write LaTeX table into file
        with open(file_name, "w", encoding="utf-8") as f2w:
            f2w.writelines(lines)
        self.log.debug(f"TeX saved to: {file_name}")

    def run(self, datas: list) -> None:
        """
        Main entry to LaTeX generator
        Args:
            datas:
        Returns:
            <None>
        """
        # with open("latex_test.json", "w") as f2w:
        #    json.dump(datas, f2w)

        # Extract column names for table creation - each column name will be presenting single
        # algorithm results.
        for data in datas:
            # TODO: Adjust file name for each table content - so it will be unique
            algo_types = list(set(data[0].keys()) - set(self.timing_metrics))

            metric_file = f"{data[2][:-4]}_metric.tex"
            time_file = f"{data[2][:-4]}_time.tex"

            self.gen_metric_table({key: data[0][key] for key in data[0].keys() & algo_types}, data[1], metric_file)
            self.gen_time_table({key: data[0][key] for key in data[0].keys() & self.timing_metrics}, data[1], time_file)


# ACM example tables
"""
# Centered across all paper columns table example
\begin{table*}
\caption{Comparison of Security Requirements with Existing Multireceiver Schemes}
 \label{tab:my-table14}
  \begin{tabular}{cccccc}
  \toprule
    Schemes&Confidentiality&Unforgeability&Anonymity&Non-repudiation&Forward Security\\
  \midrule
    Niu et al. \cite{niu2017heterogeneous} &$\checkmark$ &$\checkmark$ &$\checkmark$ &$\checkmark$ &$\checkmark$\\
    Peng et al.  \cite{DBLP:journals/iotj/PengCOVH20} &$\checkmark$ &$\checkmark$ &$\checkmark$ &$\times$ &$\times$\\
    Niu et al. \cite{DBLP:journals/cn/NiuZFHW22} &$\checkmark$ &$\checkmark$ &$\checkmark$ &$\times$ &$\times$\\
    Our scheme &$\checkmark$ &$\checkmark$ &$\checkmark$ &$\checkmark$ &$\checkmark$\\
  \bottomrule
\end{tabular}
\end{table*}


# 
# Single column rule
\begin{table}
  \caption{Comparison of Communication cost with Existing Multireceiver Schemes}
  \label{tab:my-table007}
  \begin{tabular}{cccc}
    \toprule
    Schemes&Ciphertext Length& Signcryption & Unsigncryption\\
    \midrule
    Niu et al. \cite{niu2017heterogeneous} &$n| m | + | \mathbb{G} | + 2n| \mathbb{G} |$ &$\mathcal{O}(n^2)$ &$\mathcal{O}(n)$\\
    Peng et al.  \cite{DBLP:journals/iotj/PengCOVH20} &$n| m | + (n+2)| \mathbb{Z}_q^* |$&$\mathcal{O}(n^2)$&$\mathcal{O}(n)$\\
    Niu et al. \cite{DBLP:journals/cn/NiuZFHW22} &$n| (m+2) | + 2| \mathbb{G} | + 2| \mathbb{Z}_q^* |$ &$\mathcal{O}(n)$&$\mathcal{O}(n)$\\
    Our scheme & $n| m | + | \mathbb{Z}_q^* | + | \mathbb{G} | + | \mathsf{K} |$& $\mathcal{O}(n)$ &$\mathcal{O}(1)$\\
  \bottomrule
\end{tabular}
\end{table}

"""
