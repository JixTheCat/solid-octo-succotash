import numpy as np
from math import ceil, sqrt
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats


def multihist(df: pd.DataFrame, bins=10, color="c") -> None:
    """Plot a graph for each column in the given dataframe

    Args:
        df: a Pandas dataframe whose columns you want to graph.
        bins: Number of bins for the histogram
    """
    square = ceil(sqrt(len(df.columns)))
    fig, ax = plt.subplots(
        ceil(len(df.columns) / square)
        , square
        , figsize=(20, 10))
    i = 0
    for col in df.columns:
        if df[col].dtype != object:
            ax[i // square][i % square].hist(
                df[~np.isnan(df[col])][col]
                , bins=bins
                , color=color)

            ax[i // square][i % square].set_title(col)
            i += 1
    return fig, ax


def qq(df: pd.DataFrame) -> None:
    sw_tot = 0

    for col in df.columns:
        if df[col].dtype != object:
            if len(df[~np.isnan(df[col])][col]) > (
                    10 + ceil(0.02 * len(df[col]))):
                # df[col] = df[col].apply(lambda x: x/df[col].max())
                df[col] = np.log(df[col], where=(df[col] != 0))
                sw = stats.shapiro(df[~np.isnan(df[col])][col])
                print("\n{}\n".format(col))
                stats.probplot(df[col], dist="norm", plot=plt)
                plt.show()
                if sw[1] < 0.05:
                    sw_tot += 1
                    print("{}: {}".format(col, sw))
                print('---')
            else:
                print(
                    "{} of length {} from {} has insufficient entries!!".format(
                        col, len(df[~np.isnan(df[col])][col]),
                        len(df[col])))
    print("sw =: {}".format(sw_tot))
