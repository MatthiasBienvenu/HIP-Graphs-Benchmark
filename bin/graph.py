import argparse
import os

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
        "-c",
        "--columns",
        nargs="+",
        default=None,
        help="Columns to plot (space separated). If omitted, all default columns are plotted.",
    )
    return parser.parse_args()


def load_means_stds(filename, columns):
    df = pd.read_csv(filename, skipinitialspace=True).dropna(axis=1)
    df = df.drop(columns=["width", "pattern"])

    unknown = [c for c in columns if c not in df.columns]
    if unknown:
        print(f"{filename}: unknown column(s): {', '.join(unknown)}")
        return None, None

    df = df[["length"] + columns]
    vals = df.groupby(["length"]).agg(["mean", "std"])
    means = vals.xs("mean", axis=1, level=1)
    stds = vals.xs("std", axis=1, level=1)
    return means, stds


def main():
    args = parse_args()
    columns = args.columns if args.columns is not None else DEFAULT_COLUMNS

    data = {}
    for filename in args.filenames:
        means, stds = load_means_stds(filename, columns)
        if means is None:
            continue
        data[filename] = (means, stds)

    if not data:
        print("No valid data to plot.")
        return

    # one figure per column, one line per file so they can be compared directly
    for col in columns:
        fig, ax = plt.subplots()
        for filename, (means, stds) in data.items():
            if col not in means.columns:
                continue
            ax.errorbar(
                means.index,
                means[col],
                yerr=stds[col],
                marker="x",
                capsize=3,
                label=os.path.basename(filename),
            )
        ax.set_title(col)
        ax.set_xlabel("length")
        ax.set_ylabel(col)
        if len(data) > 1:
            ax.legend()

    plt.show()


if __name__ == "__main__":
    main()
