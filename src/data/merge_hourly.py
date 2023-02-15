import logging
from pathlib import Path
from typing import Mapping
import yaml

import numpy as np
import pandas as pd

from src.features.build_features import compute_features


def merge_hourly_weather_and_load(data_root: Path, hourly_weather_dict: Mapping[str, pd.DataFrame],
                                  hourly_load_dict: Mapping[str, pd.DataFrame],
                                  not_use_wem_inputs : bool,split_week_end : bool,hourly_load_dict_wem = None):
    """
    Merge hourly weather data and hourly load data for each country.
    Data processing happening :
        1. Interpolate and fill missing load (when there is a few data points missing).
        2. Compute features (e.g. cos(hour_of_day), etc)
        3. Drop any remaining missing data (and 1 day for 366-day years) to only have full 8760 hours years.

    Args:
        data_root:
        hourly_weather_dict:
        hourly_load_dict:

    Returns:

    """
    country_mapping_path = data_root.joinpath('mapping/country_mapping.csv')

    logger = logging.getLogger(__name__)
    logger.info('Merging hourly weather data, hourly load data, and annual demand data together')

    # Get the mapping from country codes to EDC country names, load demand data, weather data, etc
    country_mapping = pd.read_csv(country_mapping_path)
    # Drop countries where we are missing weather datasets
    country_mapping.dropna(subset=['ISO_2_digits'], inplace=True)
    # Get the names of the dataset to use, for every countries
    timezones = country_mapping.set_index('country').to_dict()['timezone']

    data_frames = []
    for country in hourly_weather_dict.keys():
        # We need at least weather data (as it's a required input), but it's not a problem if we are
        # missing load data (e.g. to use the model, in scenarios predictions)
        df_weather = hourly_weather_dict[country]
        if not_use_wem_inputs : 
            df_load_actual = hourly_load_dict.get(country, pd.DataFrame(columns=['utc_timestamp', 'load']))
            df_country = pd.merge(df_weather, df_load_actual, how='left', on='utc_timestamp')
        else : 
            with open(data_root.parent.absolute().joinpath('models/logs/subsector_mapping.yaml'),'r') as file : 
                subsector_mapping = yaml.safe_load(file)
                sector = list(subsector_mapping.keys())
            df_load = hourly_load_dict_wem.get(country, pd.DataFrame(columns=['utc_timestamp','load'] + sector))
            df_load_actual = hourly_load_dict.get(country, pd.DataFrame(columns=['utc_timestamp', 'load']))
            df_country = pd.merge(df_weather, df_load, how='left', on='utc_timestamp')
            df_country.drop(columns = 'load',inplace = True) #Don't use the total load of the WEM model but the total actual load
            df_country = pd.merge(df_country,df_load_actual, how = 'left', on = 'utc_timestamp')   
        df_country = interpolate_missing_timestamps(df_country, timezone=timezones[country], country=country, not_use_wem_inputs =not_use_wem_inputs)   
        df_country = compute_features(df_country, country = country,split_week_end = split_week_end)
        df_country['country'] = country
        data_frames.append(df_country)
        
    df = pd.concat(data_frames, ignore_index=True, sort=False)
    df = get_full_year_data_only(data_root = data_root, df = df ,not_use_wem_inputs = not_use_wem_inputs)
    
    #Match the load of subsector to total actual load
    df_copy = df.copy()
    for s in sector :
        df[s] = df[s]*df.load/df_copy[sector].sum(axis = 1)
    
    return df

def interpolate_missing_timestamps(df_country: pd.DataFrame, timezone: str, country: str, not_use_wem_inputs : bool):
    # Get the local timestamp
    df_country['local_timestamp'] = df_country.utc_timestamp.dt.tz_convert(timezone)
    # Outlier (outage ?) : almost 0 load, better to interpolate as if it was a missing value
    if country == 'NOR':
        df_country.loc[df_country.utc_timestamp == '2010-12-09 19:00:00+00:00', 'load'] = np.nan
    
    # Locate missing data : complete the DateTimeIndex so that we always have full years (8760 or 8784 hours)
    # in local time.
    # This will mark missing load data (e.g. at the beginning/end of the year, or during blackouts) as NaN
    # instead of simply having a missing row, in order to interpolate and fill this NaN data later.
    full_index = pd.date_range(start=f'{int(df_country.local_timestamp.dt.year.min())}-01-01 00:00:00',
                               end=f'{int(df_country.local_timestamp.dt.year.max())}-12-31 23:00:00',
                               tz=timezone, freq='1H')
        
    try :
        df_country = df_country.set_index('local_timestamp').reindex(index=full_index).reset_index().rename(
            columns={'index': 'local_timestamp'})  
    except ValueError : #ValueError due to duplicated indexes for no reason
        df_country = df_country.set_index('local_timestamp')
        df_country = df_country.loc[~df_country.index.duplicated(),:]
        df_country = df_country.reindex(index=full_index).reset_index().rename(columns={'index': 'local_timestamp'})   
    # Fill the gaps in utc timestamps with the right values
    df_country.utc_timestamp = df_country.local_timestamp.dt.tz_convert('UTC')
    # Fill the gaps in load data
    
    df_country.loc[:, 'load'] = df_country.loc[:, 'load'].interpolate(limit_direction='both', method='cubic',
                                                                      limit=24).fillna(method='bfill', limit=2)
    df_country.loc[:,'load'] = df_country.loc[:,'load'].fillna(method='bfill', limit=5).fillna(method='ffill', limit=5)
    if not_use_wem_inputs : 
        df_country.dropna(inplace=True)
    return df_country


def get_full_year_data_only(data_root : Path, df: pd.DataFrame , not_use_wem_inputs  = True):
    logger = logging.getLogger(__name__)

    # Use only 365 days by removing the last day of the year for leap years
    df = df[df.day_of_year < 366]

    # Locate the years with missing load data
    df_count = df.groupby(['country', 'year']).load.count()
    df_incomplete_years = df_count[df_count != 8760]
    logger.info(f'The following years do not have 8760 hours of load data and will be dropped :'
                f"\n{df_incomplete_years}")
    # List of tuples (country, year) to drop
    drop_years = df_incomplete_years.reset_index().loc[:, ['country', 'year']].apply(tuple, axis=1).to_list()

    # Keep only years with 8760 hours of load data
    df = df[~df[['country', 'year']].apply(tuple, axis=1).isin(drop_years)]
    #assert not df.isna().any().any()
    
    if not not_use_wem_inputs : 
        with open(data_root.parent.absolute().joinpath('models/logs/subsector_mapping.yaml'),'r') as file : 
                subsector_mapping = yaml.safe_load(file)
                sector = list(subsector_mapping.keys())
        #Here we asses that every cluster has same number of NaN at same index. 
        df_count = df.groupby(['country', 'year'])[sector[0]].count()
        df_incomplete_years = df_count[(df_count > 8750)&(df_count != 8760)]
        drop_years = df_incomplete_years.reset_index().loc[:, ['country', 'year']].apply(tuple, axis=1).to_list()
        for c,y in drop_years : 
            df.loc[(df['country'] == c) & (df['year'] == y)] = df.loc[(df['country'] == c) & (df['year'] == y)].fillna(method = 'bfill').fillna(method = 'ffill')       
    return df
