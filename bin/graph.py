import argparse
from typing import cast

import matplotlib.pyplot as plt
import pandas as pd

DEFAULT_COLUMNS = [
    "capture",
    "instantiation",
    "first_launch_total",
    "repeat_launch_total",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot mean/std graphs from a CSV file, grouped by length."
    )
    parser.add_argument(
        "filenames",
        nargs="+",
        help="Path(s) to the CSV file(s). Provide more than one to compare them on the same graphs.",
    )
    parser.add_argument(
        "-t",
        "--transpose",
        action="store_true",
        help="Plot one graph per file instead of per column",
    )
    parser.add_argument(
        "-c",
        "--columns",
        nargs="+",
        help="Columns to plot (space separated). If omitted, all default columns are plotted.",
        default=DEFAULT_COLUMNS,
    )
    parser.add_argument(
        "-x",
        "--x-axis",
        help="Column to use for the x axis.",
        default="length",
    )

    return parser.parse_args()


def load_means_stds(filename: str, columns: list[str], x_axis: str):
    df = pd.read_csv(filename, skipinitialspace=True).dropna(axis=1)
    df = df.drop(columns=["width", "pattern"])

    unknown = [c for c in columns if c not in df.columns]
    if unknown:
        print(f"{filename}: unknown column(s): {', '.join(unknown)}")
        return None, None

    df = df[columns + [x_axis]]
    vals = df.groupby([x_axis]).agg(["mean", "std"])
    means = vals.xs("mean", axis=1, level=1)
    stds = vals.xs("std", axis=1, level=1)

    return means, stds


def main():
    args = parse_args()
    columns = cast(list[str], args.columns)
    x_axis = cast(str, args.x_axis)
    if x_axis in columns:
        print(f'Column "{x_axis}" cannot be plotted because it used as the x_axis.')
        return

    data = {}
    for filename in args.filenames:
        means, stds = load_means_stds(filename, columns, x_axis)
        data[filename] = (means, stds)

    axes = {key: plt.subplots()[1] for key in (data if args.transpose else columns)}
    for col in columns:
        for filename, (means, stds) in data.items():
            ax = axes[filename if args.transpose else col]
            ax.plot(
                means.index,
                means[col],
                "+-",
                label=col if args.transpose else filename,
            )
            ax.fill_between(
                means.index,
                means[col] - stds[col],
                means[col] + stds[col],
                alpha=0.2,
                edgecolor="k",
                linestyle="--",
            )

    for key, ax in axes.items():
        ax.set_title(key)
        ax.set_xlabel(x_axis)
        ax.set_ylabel("microseconds")
        ax.legend()

    plt.show()


if __name__ == "__main__":
    main()
