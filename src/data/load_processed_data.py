import logging
import re
from pathlib import Path
from typing import Mapping, Sequence, List, Tuple

import pandas as pd
import numpy as np
import torch



def get_subsectors(df: pd.DataFrame):
    """

    Args:
        df:

    Returns:
        The list of column names in df matching IND_**, RES_**, SER_**, TRA_**.
        With ** being 2 (or any number) of capital letters or digits.

    """
    return sorted([c for c in df.columns if re.match('((IND)|(RES)|(SER)|(TRA)|(CLU)|(ser)|(res)|(clu))_[A-Z0-9]*', c)])


def get_features(subsector_mapping: Mapping[str, Mapping]):
    return sorted(list(set([f for c in subsector_mapping.keys() for f in subsector_mapping[c]['features']])))


def combine_subsectors(df: pd.DataFrame, subsector_mapping: Mapping[str, Mapping],not_use_wem_inputs : bool, prediction = False):
    """
    Cluster subsectors together according to subsector_mapping, by summing annual demands columns in the dataframe.

    Args:
        df:
        subsector_mapping:

    Returns:

    """
    
    # Make copies to check for mistakes
    if not not_use_wem_inputs : 
        df.rename(columns = {'clu_01' : 'load_01','res_LI' : 'load_res_li','ser_LI' : 'load_ser_LI', 'res_SH' : 'load_res_SH','ser_SH':'load_ser_SH','res_SC':'load_res_SC','ser_SC':'load_ser_SC'},inplace = True)
    if prediction : 
        df.rename(columns = {'clu_01' : 'load_01','res_LI' : 'load_res_LI','ser_LI' : 'load_ser_LI', 'res_SH' : 'load_res_SH','ser_SH':'load_ser_SH','res_SC':'load_res_SC','ser_SC':'load_ser_SC'},inplace = True)
    initial_subsectors = get_subsectors(df)
    df_combined = df.copy()
    #Sum subsectors annual demand for every cluster
    for cluster in subsector_mapping.keys():
        subsectors = subsector_mapping[cluster]['subsectors']
        df_combined[cluster] = df_combined[subsectors].sum(axis=1)
        df_combined.drop(columns=subsectors, inplace=True)
    # Check that no subsectors was forgotten
    if not get_subsectors(df_combined) == sorted(subsector_mapping.keys()):
        raise Exception(f'Mismatch between subsectors specified in the mapping and subsectors after clustering :'
                        f'{sorted(subsector_mapping.keys())}\n'
                        f'{get_subsectors(df_combined)}')

    # Check that we still have the same total annual demand
    assert np.allclose(df_combined[get_subsectors(df_combined)].sum(axis=1), df[initial_subsectors].sum(axis=1))

    return df_combined


def get_df_results(df: pd.DataFrame, y: torch.Tensor, country: str, not_use_wem_inputs : bool):
    """
    Merge results from the model into the initial pandas dataframe, for a single country or for all countries

    Args:
        df: The original data frame from which the input tensors were computed
        y: the output tensor from the model, of shape (N, T, nb_subsectors), for a single country or for all countries
            if it comes from the generic model
        country: country code

    Returns:
        df_results, which is similar to df, but reshaped to have annual demands in a single column
        (instead of a column for each subsector), and an added column for the predicted load

    """
    subsectors = get_subsectors(df)

    # Reshape data frames
    subsector_loads = y.view(-1, len(subsectors)).detach().numpy()
    df_subsector_loads = pd.DataFrame(data=subsector_loads, columns=subsectors).melt(var_name='subsector',
                                                                                     value_name='load_predicted')

    if country != 'all':
        df = df[df.country == country]
    
    keep_variables = ['utc_timestamp', 'is_weekend','weekday','day_of_week','hour_of_day', 'month', 'year', 'country', 'temperature','irradiance_surface']
    if not not_use_wem_inputs : 
        loads = []
        for l in df.columns.tolist() : 
            if 'load' in l : 
                loads.append(l)
        for load in loads : 
            if load in df.columns: 
                keep_variables +=[load]
    else : 
        if 'load' in df.columns:
            # If we have a historical df, with load data
            keep_variables += ['load']

    df = df.melt(id_vars=keep_variables, value_vars=subsectors, var_name='subsector', value_name='annual_demand')

    # Make sure the output and input data frames have the same length (nb of hours)
    assert len(df_subsector_loads) == len(df)

    # Rename indexes
    df_subsector_loads.rename_axis('index', inplace=True)
    df.rename_axis('index', inplace=True)

    # Results dataframe
    return df.merge(df_subsector_loads, on=['index', 'subsector'])


def get_df_results_all_countries(df: pd.DataFrame, y_dict: Mapping[str, torch.Tensor],not_use_wem_inputs :bool):
    """

    Args:
        df: The original data frame from which the input tensors were computed
        y_dict: a dictionary of outputs from the models, for a number of countries

    Returns:
        df_results, which is similar to df, but reshaped to have annual demands in a single column
        (instead of a column for each subsector), and an added column for the predicted load

    """
    return pd.concat([get_df_results(df, y_dict[c], c,not_use_wem_inputs = not_use_wem_inputs) for c in y_dict.keys()])


def load_processed_df(project_dir: Path, scenario: str, countries: List[str] = 'all', years: List[int] = 'all',prediction = bool, not_use_wem_inputs = True):
    """
    Load the processed DataFrame (only with specified countries)

    Args:
        project_dir:
        scenario: 'historical', 'SDS2050', 'STEPS2050', etc
        countries:

    Returns:
        The DataFrame

    """
    logger = logging.getLogger(__name__)
    if prediction :
        if scenario == 'historical' : 
            df = pd.read_csv(project_dir.joinpath(f'data/processed/processed_{scenario}_test_set.csv'))
        else : 
            df =  pd.read_csv(project_dir.joinpath(f'data/processed/processed_{scenario}.csv'))
    else:
        if not_use_wem_inputs : 
            df = pd.read_csv(project_dir.joinpath(f'data/processed/processed_{scenario}_train_set.csv'))
        else : 
            df = pd.read_csv(project_dir.joinpath(f'data/processed/processed_{scenario}_train_set_wem.csv'))

    if countries != 'all':
        logger.info(f'Ignoring countries {[c for c in df.country.unique() if c not in countries]}')
        # Only keep countries specified
        df = df.loc[df.country.isin(countries), :]

    if years != 'all':
        logger.info(f'Loading only years {years}')
        # Only keep years specified
        df = df.loc[df.year.isin(years), :]

    # Check there is no 'NaN' data
    #assert not df.isna().any().any()

    return df


def compute_training_tensors(project_dir : Path, df: pd.DataFrame, subsector_mapping: Mapping[str, Mapping], match_annual_demand: bool,not_use_wem_inputs =  True):
    """
    Given a DataFrame with complete data (8760 hours/year), compute input and output tensors of each country
    for training.
    Continuous input features and total load will be scaled down to have unit variance, in order to facilitate
    training.

    Args:
        df:
        subsector_mapping:
        match_annual_demand: Whether or not annual demand estimates should be scaled to match the sum of total load

    Returns:

    """
    logger = logging.getLogger(__name__)
    countries = df.country.unique().tolist()
    subsectors = get_subsectors(df)
    subsectors_count = len(subsectors)
    features = get_features(subsector_mapping)
    features_count = len(features)

    if match_annual_demand:
        # Scale up annual demand estimates per subsector so that total annual demand matches the sum of total load
        df_total_load = df.groupby(['country', 'year']).load.sum().reset_index()
        df = df.merge(df_total_load, how='left', on=['country', 'year'], suffixes=('', '_total'))
        df['sum_annual_demands'] = df[subsectors].sum(axis=1)
        for s in subsectors:
            df[s] = df[s] * (df.load_total / df.sum_annual_demands)
        df.drop(columns=['load_total', 'sum_annual_demands'], inplace=True)
        logger.info('Scaled subsector annual demands (ETP) to match the total historical load over the year')

    inputs = {}
    outputs = {}
    T = 8760  # Number of hours per sample (year : 8760 hours)

    # Scale down total load to unit variance (this makes learning easier, as the numbers are not too big)
    if not not_use_wem_inputs : 
        load_features = []
        for l in df.columns.tolist() : 
            if 'load' in l : 
                load_features.append(l)
        #load_std = df[load_features].std()
        load_std = df.load.std()
    else : 
        load_std = df.load.std()
    logger.info(f'Scaled load data to MW/{load_std}')
        
    country_file_path = project_dir.joinpath('data/mapping/country_mapping.csv')
    country_mapping = pd.read_csv(country_file_path)
    economies = country_mapping[['country','economy']]
    advanced_economies = economies[economies.economy == 'advanced_economies'].country.tolist()
    developping_economies = economies[economies.economy == 'developping_economies'].country.tolist()
    #for c in countries + ['all']:
        #indexes = df.country.isin(countries) if c == 'all' else (df.country == c)
    for c in countries + ['advanced_economies','developping_economies'] :
        indexes = df.country.isin(advanced_economies) if c == 'advanced_economies' else(df.country.isin(developping_economies) if c == 'developping_economies' else df.country == c)
        if not_use_wem_inputs : 
            load = torch.tensor(df.loc[indexes, 'load'].values.reshape(-1, 1) / load_std).float().view(-1, T, 1)
            temperature = torch.tensor(df.loc[indexes, 'temperature'].values).float().view(-1, T, 1)
            lighting = torch.tensor(df.loc[indexes, 'irradiance_surface'].values).float().view(-1, T, 1)
            services = torch.tensor(df.loc[indexes, 'services_rate'].values).float().view(-1,T,1)
            activity = torch.tensor(df.loc[indexes, 'activity_rate'].values).float().view(-1,T,1)
            # shape : (N, T, 5), we use temperature compute the cooling/heating penalty and lighting for lighting penalty
            y = torch.cat([load, temperature,lighting,services,activity], dim=2)
        else : 
            for idx,load_ in enumerate(load_features) : 
                #load = torch.tensor(df.loc[indexes, load_].values.reshape(-1, 1) / load_std[load_]).float().view(-1, T, 1) 
                load = torch.tensor(df.loc[indexes, load_].values.reshape(-1, 1) / load_std).float().view(-1, T, 1)
                if idx == 0 : 
                    y = load
                else : 
                    y = torch.cat([y, load], dim=2)
            temperature = torch.tensor(df.loc[indexes, 'temperature'].values).float().view(-1, T, 1)
            lighting = torch.tensor(df.loc[indexes, 'irradiance_surface'].values).float().view(-1, T, 1)
            services = torch.tensor(df.loc[indexes, 'services_rate'].values).float().view(-1,T,1)
            activity = torch.tensor(df.loc[indexes, 'activity_rate'].values).float().view(-1,T,1)
            y = torch.cat([y,temperature,lighting,services,activity], dim=2)
            #order of y columns : total load, load_01, load_li, load_sh, load_sc, temperature, irradiance

        # Get torch tensor for inputs
        # Note : scaling (some) inputs could help as well (e.g temperature ?), especially if there are some big
        # numbers, but we would need to save the scaler, and re-use it on the predict data
        x_features = df.loc[indexes, features].values
        if not_use_wem_inputs : 
            x_annual_demands = df.loc[indexes, subsectors].values / load_std
        else : 
            x_annual_demands = df.loc[indexes, subsectors].values / load_std
            #x_annual_demands = df.loc[indexes, subsectors].values/load_std.drop(index = 'load').values
        x = np.concatenate([x_features, x_annual_demands], axis=1).astype('float')
        x = torch.tensor(x).float().view(-1, T, features_count + subsectors_count)

        logger.info(f'{c} - created input and output tensors ({x.shape[0]} samples)')
        inputs[c] = x
    #Inputs is finally of size (N years for n countries : SUM(N_c * n_c) , T hours, features  + subsectors clustered)
    #Outputs is finally of size (N years for n countries : SUM(N_c * n_c) , T hours, hourly load + temperature)
        outputs[c] = y
    #For 'all' every countries are concatenated
    #if not not_use_wem_inputs : 
     #   load_std = load_std.to_dict()
    return inputs, outputs, features, load_std


def compute_predict_tensors(project_dir : Path, df: pd.DataFrame, subsector_mapping: Mapping[str, Mapping], load_std = None ,not_use_wem_inputs = True):
    
    logger = logging.getLogger(__name__)
    countries = df.country.unique().tolist()
    subsectors = get_subsectors(df)
    subsectors_count = len(subsectors)
    features = get_features(subsector_mapping)
    features_count = len(features)
        
    T = 8760
    inputs = {}
    
    # Scale down total load to unit variance (this makes learning easier, as the numbers are not too big)
    logger.info(f'Scaled load data to MW/{load_std}')
    
    country_file_path = project_dir.joinpath('data/mapping/country_mapping.csv')
    country_mapping = pd.read_csv(country_file_path)
    economies = country_mapping[['country','economy']]
    advanced_economies = economies[economies.economy == 'advanced_economies'].country.tolist()
    developping_economies = economies[economies.economy == 'developping_economies'].country.tolist()
    
   # for c in countries + ['all']:
    #    indexes = df.country.isin(countries) if c == 'all' else (df.country == c)
    for c in countries + ['advanced_economies','developping_economies'] :
        indexes = df.country.isin(advanced_economies) if c == 'advanced_economies' else(df.country.isin(developping_economies) if c == 'developping_economies' else df.country == c)
        # Get torch tensor for inputs
        # Note : scaling (some) inputs could help as well (e.g temperature ?), especially if there are some big
        # numbers, but we would need to save the scaler, and re-use it on the predict data
        x_features = df.loc[indexes, features].values
        if not_use_wem_inputs : 
            x_annual_demands = df.loc[indexes, subsectors].values / load_std
        else : 
            x_annual_demands = df.loc[indexes, subsectors].values / load_std
            #x_annual_demands = df.loc[indexes, subsectors].values/ load_std['load']
        x = np.concatenate([x_features, x_annual_demands], axis=1).astype('float')
        x = torch.tensor(x).float().view(-1, T, features_count + subsectors_count)

        logger.info(f'{c} - created input and output tensors ({x.shape[0]} samples)')
        inputs[c] = x

    return inputs, features
