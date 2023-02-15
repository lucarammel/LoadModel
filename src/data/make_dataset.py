import logging
from argparse import ArgumentParser
from pathlib import Path


import sys

import yaml

import pandas as pd

from src.data.merge_hourly import merge_hourly_weather_and_load
from src.data.process_annual_demand import process_historical_annual_data, process_scenarios_annual_data
from src.data.process_hourly_load import process_historical_hourly_load, data_augmentation_CL
from src.data.process_hourly_weather import process_historical_hourly_weather
from src.data.process_scenarios_anomalies import process_scenarios_anomalies
from src.data.process_hourly_load import load_and_process_wem_inputs_hourly
from src.data.data_split import make_train_test_set, split_train_wem


def process_historical_data(data_root: Path, not_use_wem_inputs : bool,pop_weighted : None, missing_features = None, split_week_end = False): 
    df_annual_demand = process_historical_annual_data(data_root)
    hourly_weather_dict = process_historical_hourly_weather(data_root, drop_before_year=2002)
    if not_use_wem_inputs :
        hourly_load_dict = process_historical_hourly_load(data_root)
        df_hourly = merge_hourly_weather_and_load(data_root=data_root, hourly_weather_dict=hourly_weather_dict,
                                              hourly_load_dict=hourly_load_dict, not_use_wem_inputs = True,split_week_end = split_week_end)
    else : 
        hourly_load_dict = process_historical_hourly_load(data_root)
        df_hourly_load = load_and_process_wem_inputs_hourly(data_root = data_root,df_hist_annual = df_annual_demand, pop_weighted = pop_weighted)
        hourly_load_dict_wem = data_augmentation_CL(df_hourly_load = df_hourly_load, missing_features = missing_features, get_dict = True)
        df_hourly = merge_hourly_weather_and_load(data_root=data_root, hourly_weather_dict=hourly_weather_dict, hourly_load_dict_wem=hourly_load_dict_wem,hourly_load_dict=hourly_load_dict,not_use_wem_inputs = False,split_week_end = split_week_end)
    # Reformat annual demand dataframe so that we can merge it with the hourly load df
    df_annual_demand = df_annual_demand.reset_index()
    # Join hourly and annual dataframes, so that each ow also has annual subsector demand data
    df = pd.merge(df_hourly, df_annual_demand, on=['country', 'year'])
    return df

def process_scenarios_data(data_root: Path, df_hourly_historical: pd.DataFrame, load_sectors : list):
    logger = logging.getLogger(__name__)
    annual_data_frames = process_scenarios_annual_data(data_root)
    processed_data_frames = {}

    for (s, y) in annual_data_frames.keys():
        logger.info(f'Processing data for scenario {s}, year {y}')
        df_annual = annual_data_frames[(s, y)].reset_index()

        # Use hourly data from base year 2017
        if y != 2050:
            logger.warning(f'Currently using base year 2017 and scaling the temperature up, '
                        f'this may not be what you want for year {y}')
        # Drop columns that we don't need (load) or we will replace (subsectors annual demands)
        subsectors = df_annual.columns.to_list()
        subsectors.remove('country')

        df_hourly_country_list = []
        for c in df_annual.country.unique():
            df_hourly_c = df_hourly_historical.loc[df_hourly_historical.country==c
                                            ].drop(columns=subsectors + load_sectors)
            
            if df_hourly_c.temperature.count() < 8760:
                logger.warning(f'Missing historical temperature data for country {c}, '
                                'no scenario data can be made')
                pass
            
            # Get hourly data (especially temperature) for a baseline historical year
            # Preferably 2017, as this is the year that was used for temperature anomaly data by Chiara
            if 2017 in df_hourly_c.year.unique():
                df_hourly_c = df_hourly_c[df_hourly_c.year==2017]
            else:
                baseline_y = df_hourly_c.year.max()
                logger.warning(f'No historical temperature data for country {c}, '
                                f'year 2017, year {baseline_y} will be used instead as a baseline')
                df_hourly_c = df_hourly_c[df_hourly_c.year==baseline_y]
            
            # Scale up historical temperature data by adding the monthly temperature anomalies
            # todo : it would be nice to adapt the variability of the weather data as well,
            #   and/or to adapt scale up multiple years for a more representative analysis
            df_anomalies = process_scenarios_anomalies(data_root, etp_scenario=s, year=y)
            df_anomalies = df_anomalies.loc[df_anomalies.country==c]
            df_hourly_c.set_index(['month'], inplace=True)
            df_hourly_c.temperature += df_anomalies.set_index(['month']).temperature_anomaly
            df_hourly_c.reset_index(inplace=True)
            
            df_hourly_country_list.append(df_hourly_c)

        df_hourly = pd.concat(df_hourly_country_list, ignore_index=True)
        # Join hourly and annual dataframes, so that each row also has annual subsector demand data
        df = pd.merge(df_hourly, df_annual.reset_index(), on='country')
        df['year'] = y
        processed_data_frames[(s, y)] = df

    return processed_data_frames


def main(data_root: Path):
    logger = logging.getLogger(__name__)
    
    split_train_test = True
    split_week_end = True
    
    logger.info('Processing and combining raw data')
    logger.info('Mother folder: iea_load_curve_modelling')
    #Choose whether you want to use wem inputs for training or not. 
    inp = input('Not use wem inputs to train the model ? True or False') #True or False
    if inp == 'True':
        not_use_wem_inputs = True 
    elif inp == 'False' : 
        not_use_wem_inputs = False
    if not not_use_wem_inputs : 
        inp = input('Population weighted ? True or False')
        if inp == 'True':
            pop_weighted = True 
        elif inp == 'False' : 
            pop_weighted = False
    elif not_use_wem_inputs: 
        pop_weighted = None
    print(f'You have chosen to use the following parameters to make dataset :\n split week end : {split_week_end} \n WEM inputs : {not not_use_wem_inputs} \n Population weighted decomposition of WEM region hourly load : {pop_weighted}')    
        
    scenario = 'historical'
    with open(data_root.parent.absolute().joinpath('models/logs/subsector_mapping.yaml'),'r') as file : 
        subsector_mapping = yaml.safe_load(file)
        sector = list(subsector_mapping.keys())
    missing_features = ['load'] + sector
    
    df_historical = process_historical_data(data_root,not_use_wem_inputs = not_use_wem_inputs,missing_features = missing_features,split_week_end = split_week_end,pop_weighted = pop_weighted)
    #Merge hourly load and weather + data augmentation to fillna years due to mismatch in dataset for ETP and WEO
    
    
    if split_train_test : 
        logger.info('Splitting test set and training set...')
        if not not_use_wem_inputs : 
            data_train,data_test = make_train_test_set(data_processed = df_historical,scenario = scenario)
            data_train,data_train_wem = split_train_wem(data_root = data_root ,df = data_train)
            logger.info('Saving data to disk...')
            #df_historical.to_csv(data_root.joinpath('processed/processed_historical.csv'), index=False)
            data_train_wem.to_csv(data_root.joinpath('processed/processed_historical_train_set_wem.csv'),index = False)
            data_train.to_csv(data_root.joinpath('processed/processed_historical_train_set.csv'), index=False)
            data_test.to_csv(data_root.joinpath('processed/processed_historical_test_set.csv'), index=False)
    
        elif not_use_wem_inputs:
            data_train,data_test = make_train_test_set(data_processed = df_historical,scenario = scenario)
            logger.info('Saving data to disk...')
            df_historical.to_csv(data_root.joinpath('processed/processed_historical.csv'), index=False)
            data_train.to_csv(data_root.joinpath('processed/processed_historical_train_set.csv'), index=False)
            data_test.to_csv(data_root.joinpath('processed/processed_historical_test_set.csv'), index=False)
    else : 
        if not not_use_wem_inputs : 
            logger.info('Saving data to disk...')
            df_historical.to_csv(data_root.joinpath('processed/processed_historical.csv'), index=False)
        elif not_use_wem_inputs:
            logger.info('Saving data to disk...')
            df_historical.to_csv(data_root.joinpath('processed/processed_historical.csv'), index=False)
        
    scenarios_data_frames = process_scenarios_data(data_root, df_hourly_historical=df_historical, load_sectors = missing_features)
    for (s, y) in scenarios_data_frames.keys():
        df_scenario = scenarios_data_frames[(s, y)]
        df_scenario.to_csv(data_root.joinpath(f'processed/processed_{s}{y}.csv'), index=False)
    logger.info('All done !')

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    ch = logging.StreamHandler(sys.stdout)
    project_dir = Path(__file__).resolve().parents[2]

    parser = ArgumentParser()
    args = parser.parse_args()

    main(project_dir.joinpath('data'))

