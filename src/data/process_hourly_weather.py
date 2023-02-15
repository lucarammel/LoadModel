import logging
from pathlib import Path

import pandas as pd


def process_historical_hourly_weather(data_root: Path, drop_before_year: int):
    logger = logging.getLogger(__name__)
    logger.info(f'Processing hourly weather data (dropping years up to {drop_before_year})')

    country_mapping_path = data_root.joinpath('mapping/country_mapping.csv')
    weather_data_path = data_root.joinpath('raw/explanatory_variables')

    # Get the mapping from country codes to EDC country names, load demand data, weather data, etc
    country_mapping = pd.read_csv(country_mapping_path)
    # Drop countries where we are missing weather datasets
    country_mapping.dropna(subset=['ISO_2_digits'], inplace=True)

    # Get the names of the dataset to use, for every countries
    weather_datasets = {
        c: f'ninja_weather_country_{iso_code}_merra-2_population_weighted.csv' for c, iso_code in
        country_mapping.set_index('country').to_dict()['ISO_2_digits'].items()
    }

    # Build the hourly dataframe
    data_frames = {}
    for country in weather_datasets.keys():
        country_weather_data_name = weather_datasets[country]
        df_country_raw = pd.read_csv(weather_data_path.joinpath(country_weather_data_name), skiprows=2, index_col=0,
                                     parse_dates=True, date_parser=lambda col: pd.to_datetime(col, utc=True))
        df_country_raw = df_country_raw.reset_index().rename(columns={'time': 'utc_timestamp'})
        df_country = df_country_raw.loc[df_country_raw.utc_timestamp.dt.year > drop_before_year,
                                        ['utc_timestamp', 'temperature', 'irradiance_surface', 'air_density']]
        data_frames[country] = df_country
        logger.info('Added weather data for {} - {} to {}'.format(country, df_country.utc_timestamp.min(),
                                                               df_country.utc_timestamp.max()))

    return data_frames
