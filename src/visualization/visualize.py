from typing import Sequence, List

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch

from src.data.load_processed_data import get_df_results


def plot_load_duration_curve(df, country, load_name, years: Sequence[int] = None):
    """
    Args:
        df: data frame containing total (predicted or not) load
        country:
        load_name: e.g. load_predicted, but could also be temperature
        years:
    """
    if years is None:
        years = df.year.unique()
    normalize = mcolors.Normalize(vmin=min(years), vmax=max(years))
    colormap = cm.magma

    plt.figure()
    for y in years:
        d = df[(df.country == country) & (df.year == y)]
        plt.plot(d[load_name].sort_values(ascending=False).values, color=colormap(normalize(y)))

    # setup the colorbar
    scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=colormap)
    scalarmappaple.set_array(years)
    plt.colorbar(scalarmappaple)
    plt.show()


def get_profiles_figures(y: torch.Tensor, df: pd.DataFrame, country: str):
    """

    Args:
        y: Output of the model (on country-specific data, or all data)
        df: Original dataframe with all data
        country:

    Returns:

    """
    df_results = get_df_results(df, y, country)

    # Plot and save figures
    fig_month = sns.relplot(
        x='hour_of_day', y='load_predicted', col='subsector', hue='month', kind='line', col_wrap=3,
        data=df_results
    ).fig
    fig_is_we = sns.relplot(
        x='hour_of_day', y='load_predicted', col='subsector', hue='is_weekend', kind='line', col_wrap=3,
        data=df_results
    ).fig

    return fig_month, fig_is_we


def write_metrics(criterion, other_metrics, y_predict, y_true, country, writer, epoch):
    writer.add_scalar(f'{country}/epoch_loss', criterion(y_predict, y_true).item(), epoch)
    for metric_name, metric in other_metrics.items():
        writer.add_scalar(f'{country}/epoch_{metric_name}', metric(y_predict, y_true).item(), epoch)
