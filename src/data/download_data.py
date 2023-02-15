import logging
import shutil
from pathlib import Path

import requests


def save_to_disk(url, path):
    response = requests.get(url, stream=True, verify=False)
    with open(path, 'wb') as f:
        shutil.copyfileobj(response.raw, f)
    del response


def download_weather_data(project_dir: Path):
    """
        code taken from : https://data.open-power-system-data.org/weather_data
        Download hourly temperature data aggregated on a country/state/region level, from the renewables.ninja website.
    """
    logger = logging.getLogger(__name__)

    dir_countries = project_dir.joinpath('data/raw/explanatory_variables')
    base_url = 'https://www.renewables.ninja/country_downloads/'
    country_url_template = '{country}/ninja_weather_country_{country}_merra-2_population_weighted.csv'
    countries = [
        'DK', 'DE', 'FR', 'IT', 'NO', 'FI', 'SE', 'GB', 'SG', 'IN'
    ]
    logger.info(f'Downloading weather data for countries {countries}...')

    country_urls = [base_url + country_url_template.format(country=i) for i in countries]
    for u in country_urls:
        save_to_disk(u, dir_countries.joinpath(u.split('/')[-1]))
    logger.info('Done downloading weather data')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    project_dir = Path(__file__).resolve().parents[2]
    download_weather_data(project_dir)
