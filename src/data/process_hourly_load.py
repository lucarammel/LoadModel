import logging
import re
from pathlib import Path
from functools import reduce

import pandas as pd
import numpy as np
import random

import yaml 

def load_singapore_hourly_load_data(data_root: Path):
    singapore_data_root = data_root.joinpath('raw/hourly_demand/Singapore')

    # Combine files with data for a whole week into a list of df
    df_week_list = []
    for file_path in singapore_data_root.iterdir():
        # Filter all files of week-long half-hourly data, format is '20160606.xls'
        if re.match('[0-9]{8}\.xls', file_path.name):
            # (48, 3*7) array, 3 columns for each day of the week, the first being actual load
            # (demand of customers including their own generators)
            raw_data = pd.read_excel(singapore_data_root.joinpath(file_path),
                                     index_col=0, skiprows=range(5), skipfooter=6, header=None).values

            # Concatenate week long actual load data in a single column
            half_hourly_data = np.concatenate([raw_data[:, 3 * i] for i in range(7)])

            # Create dataframe for this week's data
            date_time_index = pd.date_range(start=file_path.name[:8], periods=7 * 48, freq='30min')
            df_week_list.append(pd.DataFrame({'SG_actual_load': half_hourly_data}, index=date_time_index))

    # Load the df with already combined data for 2012 to 2016
    df_a = pd.read_csv(singapore_data_root.joinpath('half-hourly-system-demand-data-from-2-feb-2012-onwards.csv'))
    date_time_index = pd.DatetimeIndex(df_a.date + ' ' + df_a.period_ending_time)
    df_a = df_a.set_index(date_time_index
                          ).drop(['date', 'period_ending_time', 'nem_demand_actual', 'nem_demand_forecast'], axis=1
                                 ).rename(columns={'system_demand_actual': 'SG_actual_load'})

    # Concat everything in a single dataframe
    df = pd.concat([df_a] + df_week_list).sort_index()

    # Check there are no missing timestamps
    assert len(pd.date_range(start=df.index.min(), end=df.index.max(), freq='30min').difference(df.index)) == 0

    # Convert to the standard format we use (1h interval, UTC timestamps)
    # The resampling takes the average of the 2 half-hourly loads (at ending times,
    # e.g. 2:30 and 3:00), and indexes it at beginning time, like European loads (e.g. 2:00)
    df = df.resample('1H').mean().tz_localize('Asia/Singapore').tz_convert('UTC')
    df = df.reset_index().rename(columns={'index': 'utc_timestamp'})
    return df


def load_france_enedis_data(data_root: Path):
    enedis_data_path = data_root.joinpath('raw/hourly_demand/enedis_france/bilan-electrique-demi-heure.csv')

    # Load the raw dataset
    df = pd.read_csv(enedis_data_path, sep=';', parse_dates=['Horodate'],
                     date_parser=lambda col: pd.to_datetime(col, utc=True))

    # Combine clients profile together to get residential load and services/commerces + industry load
    df.rename(columns={
        'Horodate': 'utc_timestamp',
        'Consommation HTA profilée (W)': 'HTA',
        'Consommation PME-PMI profilée (W)': 'PME_PMI',
        'Consommation professionnelle profilée (W)': 'PRO',
        'Consommation résidentielle profilée (W)': 'FR_RES_load',
    }, inplace=True)
    df['FR_PRO_load'] = df[['HTA', 'PME_PMI', 'PRO']].sum(axis=1)

    # Convert load in W to MW
    df.loc[:, ['FR_RES_load', 'FR_PRO_load']] /= 1e6
    # Resample half-hourly load to hourly load
    df = df.resample('1H', on='utc_timestamp').mean().reset_index()

    # Return columns we care about
    return df.loc[:, ['utc_timestamp', 'FR_RES_load', 'FR_PRO_load']]


def load_entsoe_data(data_root: Path):
    entsoe_data_path = data_root.joinpath('raw/hourly_demand/time_series_60min_singleindex.csv')
    # Load the raw dataset, and simply return it (already contains hourly data with the format we want)
    return pd.read_csv(entsoe_data_path, parse_dates=['utc_timestamp'])

def load_and_process_wem_inputs_hourly(data_root : Path,df_hist_annual :pd.DataFrame,pop_weighted : bool):
    
    """
    This function process the WEM data on hourly load which are aggregated by region. The goal is to disaggregate this hourly load by country
    using ETP annual demand for each country. We then apply the shared part to the hourly load. Then data are processed to be usable by the model. 
    """
    logger = logging.getLogger(__name__)
    wem_US_path = data_root.joinpath('raw/WEM_inputs/US_Load_Model_preprocessed.xlsx')
    df_raw_US = pd.read_excel(wem_US_path)
    logger.info('WEM US hourly load data loaded ')
    wem_eu_path = data_root.joinpath('raw/WEM_inputs/EU_LoadModel_preprocessed.xlsx')
    df_raw_EU = pd.read_excel(wem_eu_path)
    logger.info('WEM EU hourly load data loaded ')
    df_raw_US['utc_timestamp'] = pd.to_datetime(df_raw_US.Date)
    df_raw_EU['utc_timestamp'] = pd.to_datetime(df_raw_EU.Date)
    
    #Define clustering of subsectors in wem and etp
    with open(data_root.parent.absolute().joinpath('models/logs/subsector_mapping.yaml'),'r') as file : 
        subsector_mapping = yaml.safe_load(file)
    country_mapping = pd.read_csv(data_root.joinpath('mapping/country_mapping.csv'))
    subsector_mapping_wem = {'clu_01' : ['RES.CK','SER.AP','IND.CE','IND.IS','IND.NS','IND.CH','IND.PA','IND.TOT','AGR.AG','TRA.RO','TRA.RA'],
    'res_LI' : ['RES.LI'],
    'ser_LI' :['SER.LI'],
    'res_SH': ['RES.SH','RES.WH'],
    'ser_SH': ['SER.SH','SER.WH'],                        
    'res_SC':['RES.CL'], 'ser_SC': ['SER.CL']}
    
    #Compute the cluster load
    sector = list(subsector_mapping_wem.keys())
    for s in sector:
        if s == 'CLU_01' : 
            duplicated_sector = ['IND.CE','IND.IS','IND.NS','IND.CH','IND.PA']
            df_raw_US[s] = df_raw_US[[x for x in subsector_mapping_wem[s] if x not in duplicated_sector]].sum(axis = 1)
            df_raw_EU[s] = df_raw_EU[[x for x in subsector_mapping_wem[s] if x not in duplicated_sector]].sum(axis = 1)
        else :    
            df_raw_US[s] = df_raw_US[subsector_mapping_wem[s]].sum(axis = 1)
            df_raw_EU[s] = df_raw_EU[subsector_mapping_wem[s]].sum(axis = 1)
    #Extract clustered sector 
    columns_interesting = ['utc_timestamp','Year','Month','Hour','TOTAL'] + sector 
    df_US = df_raw_US[columns_interesting]
    df_EU = df_raw_EU[columns_interesting]
    
    #Population weighted demand to compute the load of each subsector for each country
    if pop_weighted : 
        country_mapping = pd.read_csv(data_root.joinpath('mapping/country_mapping.csv'))

        df_hourly_load = df_US[['utc_timestamp','Year']].copy()
        for features in  ['country'] + ['load'] + sector:
            df_hourly_load[features] = np.nan
        df_structure = df_hourly_load.copy()
 
        shares = country_mapping.set_index('country')
        countries = shares.index.tolist()
        for c in countries:
            df_aux = df_structure.copy()
            if c in ['US_NY', 'US_FL', 'US_CA']:
                df_aux.loc[:,sector] = df_US.loc[:,sector]*shares.loc[c,'pop_percentage']
                df_aux.loc[:,'load'] = df_US.loc[:,sector].sum(axis = 1)
            else : 
                df_aux.loc[:,sector] = df_EU.loc[:,sector]*shares.loc[c,'pop_percentage']
                df_aux.loc[:,'load'] = df_EU.loc[:,sector].sum(axis = 1)
            
            df_aux.country.fillna(c,inplace = True)
            df_hourly_load = pd.concat([df_hourly_load, df_aux])
        
        df_hourly_load.rename(columns = {'Year': 'year'},inplace = True)
        df_hourly_load = df_hourly_load[df_hourly_load.country.isin(countries)]
        df_hourly_load['utc_timestamp'] = pd.to_datetime(df_hourly_load.utc_timestamp,utc = True)
        return(df_hourly_load)
    
    #Compute the annual demand for 3 states of the USA for each subsector from 2014-2020
    annual_demand_US = {}
    res = []
    for y in list(df_US.Year.unique()):
        for s in sector : 
            res.append(np.sum(df_US[df_US.Year == y][s]))
        annual_demand_US[y] = res
        res = []

    #Compute the annual demand of Europe for each subsector from 2014-2020
    annual_demand_EU = {}
    res = []
    for y in list(df_EU.Year.unique()):
        for s in sector : 
            res.append(np.sum(df_EU[df_EU.Year == y][s]))
        annual_demand_EU[y] = res
        res = []

    #Compute the annual_demand per subsector per year per country using etp data
    cluster_sum_etp = {}
    l = []
    for c,y in df_hist_annual.index : 
        for s in sector:
            l.append([s,df_hist_annual.loc[c].loc[y].loc[subsector_mapping[s]['subsectors']].sum()])
        cluster_sum_etp[(c,y)] = l
        l = []

    #Compute the shared part of EU & US countries in annual demand for subsectors using 
    #ETP data country demand and WEM region aggregated annual demand 

    l = []
    shared_part = {}
    for (c,y) in cluster_sum_etp.keys() : 
        if c in ['US_NY', 'US_FL', 'US_CA']:
            if y in annual_demand_US.keys():
                for idx,s in enumerate(cluster_sum_etp[(c,y)]):
                    l.append([s[0],s[1]/annual_demand_US[y][idx]])
                shared_part[(c,y)] = l
                l = []
        else : 
            if y in annual_demand_EU.keys():
                for idx,s in enumerate(cluster_sum_etp[(c,y)]):
                    l.append([s[0],s[1]/annual_demand_EU[y][idx]])
                shared_part[(c,y)] = l
                l = []
            
    #Compute the hourly load for each subsector with respect to the shared part 
    year_dict= {}
    year = []
    res = []
    countries = []

    #Compute the couple year and countries where we do have values. Put it in a usable format
    idx_countries = []
    for c,y in shared_part:
        countries.append(c)
        year.append(y)
    
    old_country = countries[0]    
    for idx,c in enumerate(countries) : 
        if idx != len(countries)-1 :
            if countries[idx+1] != old_country : 
                idx_countries.append(idx)
                old_country = countries[idx+1]
        else : 
            idx_countries.append(idx)

    old_idx = 0    
    for i in idx_countries : 
        year_dict[countries[i]] = year[old_idx :i+1]
        old_idx = i+1

    countries = np.unique(countries)

    df_hourly_load = df_US[['utc_timestamp','Year']].copy()
    df_hourly_load.rename(columns = {'Year' : 'year'},inplace = True)
    for features in  ['country'] + ['load'] + sector:
        df_hourly_load[features] = np.nan
    df_structure = df_hourly_load.copy()
  
    for c in countries:
        df_aux = df_structure.copy()
        for y in year_dict[c]:
            for share in shared_part[(c,y)] : 
                if 'US' in c:
                    df_aux.loc[df_aux.year == y,share[0]] = df_US[df_US.Year == y][share[0]]*share[1]
                else : 
                    df_aux.loc[df_aux.year == y,share[0]] = df_EU[df_EU.Year == y][share[0]]*share[1]
            if 'US' in c :
                mean_ratio = np.mean(np.array(shared_part[(c,y)],dtype = 'O')[:,1])
                df_aux.loc[df_aux.year == y,'load'] = df_US[df_US.Year == y].TOTAL*mean_ratio
            else : 
                mean_ratio = np.mean(np.array(shared_part[(c,y)],dtype = 'O')[:,1]) 
                df_aux.loc[df_aux.year == y,'load'] = df_EU[df_EU.Year == y].TOTAL*mean_ratio
        df_aux.country.fillna(c,inplace = True)
        df_hourly_load = pd.concat([df_hourly_load, df_aux])

    logger.info('WEM data disaggregation completed using ETP annual demands per subsector for countries')
    df_hourly_load = df_hourly_load[df_hourly_load.country.isin(countries)]
    df_hourly_load['utc_timestamp'] = pd.to_datetime(df_hourly_load.utc_timestamp,utc = True)
    return(df_hourly_load)

def data_augmentation_CL(df_hourly_load : pd.DataFrame, missing_features : list ,get_dict : bool) :
    """
    Data augmentation of missing features using a linear combination of available year data. 
    """
    logger = logging.getLogger(__name__)
    logger.info('Processing to data augmentation using linear combination of other historical years')
    countries = list(df_hourly_load.country.unique())
    year = {}
    rdm = {}
    shares = {}
    cl = 0
    year_na = {}
    for c in countries : 
        year[c] = list(df_hourly_load.loc[df_hourly_load.country == c].dropna().year.unique())
        year_na[c] = list(df_hourly_load[(df_hourly_load.load.isna()) & (df_hourly_load.country == c)].year.unique())             
        for y in year_na[c]:
            if df_hourly_load.loc[(df_hourly_load.country == c) & (df_hourly_load.year == y)].shape[0] != 8760 : 
                logger.info(f"Year {y} of country {c} won't be take into account for data augmentation because of a year - length different of 8760 hours")
                year_na[c].remove(y)
        for y in year_na[c]:
            rdm[(c,y)] = [random.random() for i in range(len(year[c]))]
            shares[(c,y)] = (rdm[(c,y)]/np.sum(rdm[(c,y)]))
            for f in missing_features: 
                for i in range(len(shares[(c,y)])): 
                    cl = cl + shares[(c,y)][i]*df_hourly_load[(df_hourly_load.year == year[c][i]) & (df_hourly_load.country == c)][f].values
                df_hourly_load.loc[(df_hourly_load.country == c) & (df_hourly_load.year == y),f] = cl
                cl = 0 
    if get_dict : 
        #Reshape into dict format the dataframe of hourly load 
        df_hourly_load.reset_index(drop = True ,inplace = True)
        df_out = {}
        for c in countries : 
            df_out[c] = df_hourly_load[df_hourly_load.country == c].drop('country',axis = 1)
        return(df_out)
    else :
        return(df_hourly_load)

def load_us_data(data_root: Path):
    df_state_list = []
    for file_path in data_root.joinpath('raw/hourly_demand/US').iterdir():
        s = file_path.name
        state_name = s[11: s.index('_(region)_')]
        
        # Load the .csv file
        df_state = pd.read_csv(file_path, skiprows=4)
        df_state = df_state.rename(columns={'Category': 'utc_timestamp'})
        df_state.utc_timestamp = pd.to_datetime(df_state.utc_timestamp, utc=True)
        
        # Rename the load data column
        df_state = df_state.rename(columns={df_state.columns[1]: f'load_{state_name}'})
        df_state[f'load_{state_name}'] = df_state[f'load_{state_name}'].astype('float')
        
        df_state_list.append(df_state)
    
    df_merged = reduce(lambda  left,right: pd.merge(left,right,on='utc_timestamp', how='outer'), df_state_list)
    return df_merged

def load_india_data(data_root : Path) :
    india_data_root = data_root.joinpath('raw/hourly_demand/Carbontracker_india.xlsx')
    df = pd.read_excel(india_data_root)
    df.rename(columns = {'timestamps' : 'utc_timestamp'} ,inplace = True)
    df.utc_timestamp = pd.to_datetime(df.utc_timestamp, utc = True)
    df = df[df.utc_timestamp.dt.minute == 0]
    return df

def process_historical_hourly_load(data_root: Path):
    logger = logging.getLogger(__name__)
    logger.info('Processing hourly load data')

    country_mapping_path = data_root.joinpath('mapping/country_mapping.csv')

    # Get the mapping from country codes to EDC country names, load demand data, weather data, etc
    country_mapping = pd.read_csv(country_mapping_path)
    # Drop any missing data (e.g. load or temperature)
    country_mapping.dropna(inplace=True)

    # Get the names of the datasets to use, for every countries
    load_data_names = country_mapping.set_index('country').to_dict()['load_data_name']

    # Load the raw datasets
    # Each function loads a different dataset, and should return a pandas DataFrame with (at least) a DateTimeIndex
    # column named 'utc_timestamp', and corresponding hourly loads in other columns, with their names being
    # registered in the 'country_mapping.csv' file
    df_entsoe = load_entsoe_data(data_root)
    logger.info('Loaded ENTSO-E load data')
    df_sgp = load_singapore_hourly_load_data(data_root)
    logger.info('Loaded Singapore load data')
    df_enedis = load_france_enedis_data(data_root)
    logger.info('Loaded Enedis load data for France')
    df_us = load_us_data(data_root)
    logger.info('Loaded US load data')
    df_india = load_india_data(data_root)
    logger.info('Loaded India load data')

    # Merge the datasets together
    df_raw = df_entsoe.merge(df_sgp, how='outer', on='utc_timestamp') \
        .merge(df_enedis, how='outer', on='utc_timestamp').merge(df_us, how='outer', on='utc_timestamp')\
        .merge(df_india, how = 'outer', on = 'utc_timestamp')

    # Build the hourly load DataFrame
    data_frames = {}
    for country in load_data_names.keys():
        # Get the country load data from the raw df
        load_dataset_name = load_data_names[country]
        df_country = df_raw.dropna(subset=[load_dataset_name]).loc[:, ['utc_timestamp', load_dataset_name]]
        df_country.rename(columns={load_dataset_name: 'load'}, inplace=True)

        data_frames[country] = df_country
        logger.info('Added load data for {} - {} to {}'.format(country, df_country.utc_timestamp.min(),
                                                               df_country.utc_timestamp.max()))

    return data_frames
