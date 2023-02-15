import logging
from pathlib import Path

import pandas as pd

from src.data.load_processed_data import get_subsectors

PJ_to_MWh = 1e9 / 3600
TJ_to_MWh = 1e6 / 3600


def load_historical_etp_data(data_root: Path):
    etp_data_path = data_root.joinpath('raw/annual_demand/etp_inputs.xlsx')
    wem_inputs_india_path = data_root.joinpath('raw/annual_demand/wem_inputs_india_2019.xlsx')
    
    # Load ETP data
    df_raw = pd.read_excel(etp_data_path)
    df_raw_india = pd.read_excel(wem_inputs_india_path)
    df_raw['demand_MWh'] = df_raw.demand_PJ * PJ_to_MWh
    df_raw_india['demand_MWh'] = df_raw_india.demand_PJ * PJ_to_MWh
    df_fusion = pd.concat([df_raw, df_raw_india])

    return df_fusion.loc[:, ['country', 'year', 'subsector', 'sector', 'demand_MWh']]


def load_edc_data(data_root: Path):
    edc_data_path = data_root.joinpath('raw/annual_demand/edc_historical_data.csv')
    country_mapping_path = data_root.joinpath('mapping/country_mapping.csv')
    sector_mapping_path = data_root.joinpath('mapping/sector_mapping.csv')

    # Adding the option encoding='latin' may help if they are errors with encoding
    df_raw = pd.read_csv(edc_data_path, na_values=['..', 'c'])
    # remove white spaces at the beginning of columns
    df_raw.columns = df_raw.columns.str.lstrip()
    # Replace NaN values for Road in Sweden by 0 (as EV demand is very small), to avoid dropping valuable years
    df_raw.loc[df_raw.COUNTRY == 'Sweden', 'Road'] = df_raw.loc[df_raw.COUNTRY == 'Sweden', 'Road'].fillna(0)
    # Drop rows where some data is missing
    df_raw.dropna(inplace=True)
    # Drop too old data, as we don't have hourly load data anyway
    df_raw.drop(df_raw[df_raw.TIME < 2003].index, axis=0, inplace=True)

    # Get the mapping from country codes to EDC country names, load demand data, weather data, etc
    country_mapping = pd.read_csv(country_mapping_path)
    edc_names = country_mapping.set_index('country').to_dict()['edc_country_name']
    # Get the mapping from subsector codes to EDC subsector names
    sector_mapping = pd.read_csv(sector_mapping_path)

    # Build the data frame
    data_frames = []
    for country_code in edc_names.keys():
        country_name = edc_names[country_code]

        # Go through every sector
        for i in range(sector_mapping.shape[0]):
            subsector = sector_mapping.loc[i, 'subsector']
            sector = sector_mapping.loc[i, 'sector']
            edc_mapping = sector_mapping.loc[i, 'edc_mapping']

            data_frames.append(
                pd.DataFrame({
                    'country': country_code,
                    'year': df_raw.loc[df_raw.COUNTRY == country_name, 'TIME'],
                    'subsector': subsector,
                    'sector': sector,
                    'demand_TJ': pd.Series(df_raw.loc[df_raw.COUNTRY == country_name, edc_mapping], dtype='float'),
                })
            )
    df = pd.concat(data_frames, ignore_index=True)
    df['demand_MWh'] = df.demand_TJ * TJ_to_MWh

    return df.drop(columns='demand_TJ')


def make_custom_adjustments(df):
    """
    Transform etp/edc data to get custom data points, e.g. France demand data for some subsectors only

    Args:
        df:

    Returns:

    """
    # Print a warning if we are using singapore annual data (not correctly estimated for all subsectors)
    if 'SGP' in df.country.unique():
        logger = logging.getLogger(__name__)
        logger.warning('⚠️ - Data for SGP is not 100% correct : residential and commercial '
                       'demand data for singapore only works with non-temperature'
                       ' dependant uses aggregated together')

    # For Enedis non national data, copy FRA data and set to zero some of it
    # Residential demand
    df_fra_res = df.loc[df.country == 'FRA'].replace({'country': {'FRA', 'FRA_RES'}})
    df_fra_res.loc[df_fra_res.sector != 'RES', 'demand_MWh'] = 0
    # Non-residential demand : industries + commerces & services
    df_fra_pro = df.loc[df.country == 'FRA'].replace({'country': {'FRA', 'FRA_PRO'}})
    df_fra_res.loc[df_fra_pro.sector == 'RES', 'demand_MWh'] = 0

    # and for 2019 italy test year
    # df = df.append(
    #     df.loc[(df.country == 'ITA') & (df.year == 2017)].replace(
    #         {'year': 2017}, 2019))

    return pd.concat([df, df_fra_res, df_fra_pro], ignore_index=True)


def process_historical_annual_data(data_root: Path):
    """

    Args:
        data_root:

    Returns:
        A DataFrame of annual demand indexed by country and year, each subsector in a column

    """
    logger = logging.getLogger(__name__)
    logger.info('Processing historical annual demand data')

    df_etp = load_historical_etp_data(data_root)
    logger.info('Loaded data from ETP')
    df_edc = load_edc_data(data_root)
    logger.info('Loaded data from EDC')

    
    df = pd.concat([df_etp, df_edc])
    df = make_custom_adjustments(df)

    # Reshape data frame and drop years where we are missing data
    df = df.drop(columns='sector').pivot_table(index=['country', 'year'], columns='subsector', values='demand_MWh')
    df.dropna(inplace=True)

    return df


def process_scenarios_annual_data(data_root: Path):
    etp_data_path = data_root.joinpath('raw/annual_demand/etp_inputs_scenarios.xlsx')

    # Load ETP data
    df_raw = pd.read_excel(etp_data_path)
    df_raw = df_raw.melt(id_vars=['country', 'year', 'scenario'], value_vars=get_subsectors(df_raw), value_name='demand_PJ', var_name='subsector')
    df_raw['demand_MWh'] = df_raw.demand_PJ * PJ_to_MWh

    # Get separate dataframes for different scenarios/year tuples
    data_frames = {}
    scenario_years = df_raw.set_index(['scenario', 'year']).index.tolist()
    for (s, y) in scenario_years:
        df = df_raw.loc[(df_raw.scenario == s) & (df_raw.year == y),
                        ['country', 'subsector', 'demand_MWh']]

        # Reshape data frame and fill years where we are missing data with 0
        df = df.pivot_table(index='country', columns='subsector', values='demand_MWh').fillna(0)
        data_frames[(s, y)] = df

    return data_frames
