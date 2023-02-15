# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 14:13:19 2021

@author: PEREIRA_LU
"""

import logging
from argparse import ArgumentParser
from pathlib import Path
from typing import Mapping
import pandas as pd
import random
import numpy as np
import math
import yaml

def make_train_test_set(data_processed : pd.DataFrame, scenario : str):
    logger = logging.getLogger(__name__)
    df = data_processed
    countries =  df.country.unique()
    test_ratio_size = 0.35
    logger.info(f'Spliting test and train set based on year with size ratio of {test_ratio_size*100}%')
    year_count = {}
    nbr_year_test = {}
    rdm = []
    l = []
    year_to_drop = {}
    for c in countries :
        year_count[c] = len(df[df.country == c].year.unique())
        nbr_year_test[c] = round(test_ratio_size*year_count[c])
        rdm = np.unique([random.randint(0,year_count[c]) for i in range(nbr_year_test[c])])
        while len(rdm) != nbr_year_test[c] : 
            rdm = np.unique([random.randint(0,year_count[c]) for i in range(nbr_year_test[c])])
        for i in rdm:
            l.append(min(df[df.country == c].year.unique())+i)            
        year_to_drop[c] = l  
        l = []
    df_test = pd.DataFrame()
    df_train = pd.DataFrame()
    for c in countries : 
        logger.info(f'Splitting train and test set for country {c}..')
        if len(year_to_drop[c]) == 0:
            logger.warning('⚠️ - Not enough load data to split the dataset. Train set  = test set to bypass the problem'' ')
            df_test = pd.concat([df_test,df[(df.country == c)]])
        else : 
            for y in year_to_drop[c]:
                df_test = pd.concat([df_test,df[(df.country == c) & (df.year == y)]])
    for c in countries :
        for y in df[df.country == c].year.unique():
            if y not in year_to_drop[c]:
                df_train = pd.concat([df_train,df[(df.country == c) & (df.year == y)]]) 
        if nbr_year_test[c] == 0: 
           year_to_drop[c] = df[(df.country == c)].year.unique().tolist()       
    logger.info(f"Test set is composed of the following years : {pd.DataFrame.from_dict(year_to_drop,orient = 'index')}")
    return (df_train, df_test)

def split_train_wem(data_root : Path, df: pd.DataFrame):
    with open(data_root.parent.absolute().joinpath('models/logs/subsector_mapping.yaml'),'r') as file : 
                subsector_mapping = yaml.safe_load(file)
                sector = list(subsector_mapping.keys())
    logger = logging.getLogger(__name__)
    logger.info('Splitting the train set into a dataset with WEM hourly load by end-use and the one with the total actual load')
    df_train_wem = df[df[sector[0]].notna()]
    df_train = df.drop(columns = sector)
    return(df_train,df_train_wem)
    
    

def main(data_root: Path):
    #Function to split a total dataset into a training set and a test set. 
    logger = logging.getLogger(__name__)
    logger.info('Mother folder: iea_load_curve_modelling')
    scenario = 'historical'
    processed_data_path = data_root.joinpath(f'processed/processed_{scenario}.csv')
    data_processed = pd.read_csv(processed_data_path)
    logger.info('Splitting ...')
    data_train,data_test = make_train_test_set(data_processed = data_processed,scenario = scenario)
    
    logger.info('Saving data to disk...')
    data_train.to_csv(data_root.joinpath('processed/processed_historical_train_set.csv'), index=False)
    data_test.to_csv(data_root.joinpath('processed/processed_historical_test_set.csv'), index=False)
    logger.info('All done !')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    project_dir = Path(__file__).resolve().parents[2]

    parser = ArgumentParser()
    args = parser.parse_args()
    main(project_dir.joinpath('data'))