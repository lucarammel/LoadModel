import logging
from pathlib import Path

import cvxpy as cp
import numpy as np
import pandas as pd


def load_weo_hourly_data(project_dir: Path, file_name: str):
    # load data
    df_weo = pd.read_csv(project_dir.joinpath(f'data/external/weo_results/{file_name}.csv'))
    df_weo.rename(columns={'Hour': 'hour_of_day', 'Hour in Year': 'hour_of_year'}, inplace=True)

    # process dates
    df_weo['timestamp'] = pd.DatetimeIndex(
        df_weo.agg(lambda x: f"{x['Day']} {(x['hour_of_day'] - 1):02d}:00:00", axis=1), dayfirst=True)
    df_weo['month'] = df_weo.timestamp.dt.month
    df_weo['year'] = df_weo.timestamp.dt.year
    df_weo['day_of_week'] = df_weo.timestamp.dt.dayofweek
    df_weo['is_weekend'] = (df_weo.day_of_week == 5) | (df_weo.day_of_week == 6)
    df_weo.drop(columns=['Day', 'hour_of_year', 'day_of_week'], inplace=True)
    return df_weo


def separate_res_non_res_single_year(annual_demand_res_sc: float, annual_demand_ser_sc: float,
                                     hourly_load_clu_sc: np.ndarray,
                                     weo_load_res_sc: np.ndarray, weo_load_ser_sc: np.ndarray):
    """

    Args:
        annual_demand_res_sc: Annual demand (MWh) for residential space cooling, ETP estimate
        annual_demand_ser_sc: idem, for non-residential space cooling
        hourly_load_clu_sc: hourly load data for the space cooling cluster, for a whole year (8760 data points), in MW.
            This is the output of the model to separate into a residential (res) and a non-residential (ser) load curve.
        weo_load_res_sc: hourly load data for residential space cooling according to the WEO model, for a whole year.
            Unit does not matter (can be MW, GW, or even unit-less) as it will be normalized
        weo_load_ser_sc: idem, for non-residential space cooling

    Returns:
        (hourly_load_res_sc, hourly_load_ser_sc) : Hourly load data after residential/non-residential separation, MW

    """
    D_r = annual_demand_res_sc
    D_s = annual_demand_ser_sc
    z = hourly_load_clu_sc
    # Normalize WEO loads
    y_r = weo_load_res_sc / weo_load_res_sc.sum()
    y_s = weo_load_ser_sc / weo_load_ser_sc.sum()

    # Let's have, for t in [1, 8760]
    #   y_t = D_r * y_r,t + D_s * y_s,t
    #   z_t = D_r * z_r,t + D_s * z_s,t
    #
    # where
    #   y and z are WEO and our model's total cooling loads (MW)
    #   D_r, D_s are annual demand estimates for residential and non-residential cooling (MWh)
    #   y_r, y_s, z_r, z_s are residential and non-residential cooling load, from WEO and our model,
    #       but normalized (sum over the year = 1)
    #
    # We want to separate z into z_r and z_s, with
    #   z_r and z_s as close as possible to y_r and y_s (WEO profiles)
    #   under the constraint z = D_r * z_r + D_s * z_s (we want to keep the exact same total load)
    #
    # We can choose to write the objective as minimizing the residual sum of squares :
    #   sum((y_r,t - z_r,t)**2 + (y_s,t - z_s,t)**2)    (sum over t in [1, 8760])
    # under the constraint.
    #
    # We can rewrite the constraint as
    #   z_r,t = (z_t - D_s * z_s,t) / D_r
    # We can use that to transform the objective into minimizing
    #   sum((y_r,t - (z_t - D_s * z_s,t) / D_r)**2 + (y_s,t - z_s,t)**2)
    # and get rid of the constraint.

    # The variable that we want to find (normalized non-residential space cooling hourly load) 8760 = number of hours in a year
    z_s = cp.Variable(8760)
    # The cost function we want to minimize
    cost = cp.sum_squares(y_r - (z - D_s * z_s) / D_r) + cp.sum_squares(y_s - z_s)
    # Add some constraints nonetheless to ensure a positive and normalized load (sum over the year = 1)
    prob = cp.Problem(cp.Minimize(cost), constraints=[z_s >= 0, np.ones(8760) @ z_s == 1])

    # Solve the optimization problem. Constraints are not strictly enforced, so to avoid loads that are too negative,
    # we increase the required precision (using options eps_abs, eps_rel from the OSQP solver,
    # cf https://www.cvxpy.org/tutorial/advanced/index.html#setting-solver-options)
    prob.solve(eps_abs=1e-8, eps_rel=1e-8)

    if prob.status != cp.OPTIMAL:
        raise ValueError('Could not solve optimisation problem and separate residential/non-residential cooling loads')

    # Resulting hourly loads after separation
    hourly_load_ser_sc = D_s * np.clip(z_s.value, a_min=0, a_max=None)  # transform any negative load to zero
    hourly_load_res_sc = z - hourly_load_ser_sc

    return hourly_load_res_sc, hourly_load_ser_sc


def separate_res_non_res(df_results: pd.DataFrame, df_original: pd.DataFrame, weo_data_file_name: str):
    """

    Args:
        df_results: Data frame obtained after model predictions, containing a column 'load_predicted' for subsectors
            loads
        df_original: Original data frame, before clustering
        weo_data_file_name: Name of the hourly dataset to use as a target

    Returns:
        df_results
    """
    logger = logging.getLogger(__name__)
    logger.info(f'Using WEO data {weo_data_file_name} to separate CLU_SC load into RES_SC and SER_SC loads')

    # Get WEO hourly load curves
    proj_dir = Path(__file__).resolve().parents[2]
    df_weo = load_weo_hourly_data(proj_dir, file_name=weo_data_file_name)
    weo_load_res_sc = df_weo['RES.CL'].values
    weo_load_ser_sc = df_weo['SER.CL'].values
    assert weo_load_res_sc.shape == weo_load_ser_sc.shape == (8760,)

    separated_loads = []
    countries = df_results.country.unique()
    for c in countries:
        df_country = df_results[df_results.country == c]
        for y in df_country.year.unique():
            logger.info(f'Applying optimisation algorithm to country {c}, year {y}')

            # Get annual demand estimates
            df_o = df_original[(df_original.country == c) & (df_original.year == y)]
            # The annual demand columns in the original df should be 8760 times the same element
            assert len(df_o.RES_SC.unique()) == 1
            assert len(df_o.SER_SC.unique()) == 1
            annual_demand_res_sc = df_o.RES_SC.unique()[0]
            annual_demand_ser_sc = df_o.SER_SC.unique()[0]

            # Get predicted hourly load values
            df_clu_sc = df_country.loc[(df_country.year == y) & (df_country.subsector == 'CLU_SC')]
            hourly_load_clu_sc = df_clu_sc.load_predicted.values
            assert hourly_load_clu_sc.shape == (8760,)

            # Apply optimisation algorithm
            hourly_load_res_sc, hourly_load_ser_sc = separate_res_non_res_single_year(
                annual_demand_res_sc=annual_demand_res_sc, annual_demand_ser_sc=annual_demand_ser_sc,
                hourly_load_clu_sc=hourly_load_clu_sc, weo_load_res_sc=weo_load_res_sc, weo_load_ser_sc=weo_load_ser_sc)

            df_res_sc = df_clu_sc.copy()
            df_res_sc.subsector = 'RES_SC'
            df_res_sc.load_predicted = hourly_load_res_sc
            df_ser_sc = df_clu_sc.copy()
            df_ser_sc.subsector = 'SER_SC'
            df_ser_sc.load_predicted = hourly_load_ser_sc
            separated_loads += [df_res_sc, df_ser_sc]

    df_check = df_results.pivot_table(index=['utc_timestamp', 'country'], columns='subsector', values='load_predicted')

    df_results = pd.concat([df_results[df_results.subsector!='CLU_SC']] + separated_loads, ignore_index=True)

    # Safety check
    df_results_pivoted = df_results.pivot_table(index=['utc_timestamp', 'country'], columns='subsector',
                                                values='load_predicted')
    assert np.allclose(df_check.CLU_SC.values, (df_results_pivoted.RES_SC + df_results_pivoted.SER_SC).values)
    assert not df_results_pivoted[['RES_SC', 'SER_SC']].isna().any().any()

    return df_results
