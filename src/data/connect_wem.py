# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 11:29:51 2021

@author: PEREIRA_LU
"""

import pandas as pd
from pathlib import Path
import logging


def main(data_root : Path, df : pd.DataFrame) : 

    df.utc_timestamp = pd.DatetimeIndex(df.utc_timestamp)
    df.weekday = df.weekday + 1
    
    
    ## Share hour : Data
    df_td_hour = df.groupby(['month','weekday','hour_of_day'])[['load_predicted']].sum()
    df_td_1 = df.groupby(['month','weekday'])[['load_predicted']].sum()
    df_td_hour['shares_in_%'] = (df_td_hour/df_td_1).values
    df_td_hour.drop(columns = 'load_predicted',inplace  = True)
    df_td_hour.reset_index(inplace  = True)
    df_td_hour['typical_day'] = df_td_hour.apply(lambda row : 'TD'+f'{int(row.month):02d}'+str(int(row.weekday)), axis = 1)
    df_td_hour = df_td_hour.set_index(['typical_day','month','weekday','hour_of_day'])['shares_in_%'].unstack().reset_index()
    df_td_hour = df_td_hour.rename_axis(None, axis=1)
    
    ## Share Months
    df_month_sub = df.groupby(['subsector','month'])[['load_predicted']].sum()
    df_month = df.groupby(['subsector'])[['load_predicted']].sum()
    df_month_sub['shares_in_%'] = (df_month_sub/df_month).values
    df_month_sub.drop(columns = 'load_predicted',inplace  = True)
    df_month_sub.reset_index(inplace = True)
    df_month_sub = df_month_sub.set_index(['subsector','month'])['shares_in_%'].unstack().reset_index()
    df_month_sub = df_month_sub.rename_axis(None, axis=1)
    df_month_sub.rename(columns = {1 : 'January', 2 : 'February', 3 : 'March', 4 : 'April', 5 : 'May', 6 : 'June', 7 : 'July', 8 : 'August',
                                  9 : 'September', 10 : 'October', 11 : 'November', 12 : 'December'}, inplace = True)
    
    
    winter = ['December','January','February']
    summer = ['June','July','August']
    spring = ['March', 'April', 'May']
    autumn = ['September','October','November']
    
    df_month_sub['spring'] = df_month_sub[spring].sum(axis = 1)
    df_month_sub['summer'] = df_month_sub[summer].sum(axis = 1)
    df_month_sub['autumn'] = df_month_sub[autumn].sum(axis = 1)
    df_month_sub['winter'] = df_month_sub[winter].sum(axis = 1)
    
    ## Share Days 
    
    df_day_sub = {}
    df_day = {}
        
    for i in df.month.unique().tolist() : 
        filter = df.month == i
        df_filter = df[filter].copy()
        df_filter.day_of_week = df_filter.day_of_week + 1
            
        df_day_sub[i] = df_filter.groupby(['subsector','day_of_week'])[['load_predicted']].sum()
        df_day[i] = df_filter.groupby(['subsector'])[['load_predicted']].sum()
        df_day_sub[i]['shares_in_%'] = (df_day_sub[i]/df_day[i]).values
        df_day_sub[i].drop(columns = 'load_predicted',inplace  = True)
        df_day_sub[i].reset_index(inplace = True)
        df_day_sub[i] = df_day_sub[i].set_index(['subsector','day_of_week'])['shares_in_%'].unstack().reset_index()
        df_day_sub[i] = df_day_sub[i].rename_axis(None, axis=1)
    
        df_day_sub[i].rename(columns = {1 : 'Monday', 2 : 'Tuesday', 3 : 'Wednesday', 4 : 'Thursday', 5 : 'Friday', 6 : 'Saturday', 7 : 'Sunday'}, 
                          inplace = True)
        
    ## Share Hours
    
    winter_filter = [12,1,2]
    summer_filter = [6,7,8]
    weekday_filter = [0,1,2,3,4]
    saturday = [5]
    sunday = [6]
    df_hour_sub = {}
    df_hour = {}
    
    for idx_season,season in enumerate([summer_filter,winter_filter]) : 
        for idx_day,day in enumerate([weekday_filter,saturday,sunday]) : 
            
            filter = (df.month.isin(season) & df.day_of_week.isin(day))
            df_filter = df[filter].copy()                
            df_hour_sub[(idx_season,idx_day)] = df_filter.groupby(['subsector','hour_of_day'])[['load_predicted']].sum()
            df_hour[(idx_season,idx_day)] = df_filter.groupby(['subsector'])[['load_predicted']].sum()
            df_hour_sub[(idx_season,idx_day)]['shares_in_%'] = (df_hour_sub[(idx_season,idx_day)]/df_hour[(idx_season,idx_day)]).values
            df_hour_sub[(idx_season,idx_day)].drop(columns = 'load_predicted',inplace  = True)
            df_hour_sub[(idx_season,idx_day)].reset_index(inplace = True)
            df_hour_sub[(idx_season,idx_day)] = df_hour_sub[(idx_season,idx_day)].set_index(['subsector','hour_of_day'])['shares_in_%'].unstack().reset_index()
            df_hour_sub[(idx_season,idx_day)] = df_hour_sub[(idx_season,idx_day)].rename_axis(None, axis=1)
     
    return(df_td_hour,df_month_sub,df_day_sub, df_hour_sub)

    


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)
    project_dir = Path(__file__).resolve().parents[2]
    data_root = project_dir.joinpath('data')
    
    #version_number = 34
    version_number = int(input('Choose the version to extract the outputs'))
    scenario = 'historical'
    path_file = data_root.parent.absolute().joinpath(f'models/logs/version_{version_number}/output_{scenario}.csv')
    df = pd.read_csv(path_file)
    region_mapping = {#'EUa' : ['FRA','DEU','ITA','GBR'],
                      #'Eub' : ['DNK','NOR','FIN','SWE'],
                      #'EUc': ['DNK','NOR','FIN','SWE','DEU'],
                      #'US' : ['US_NY','US_FL','US_CA'],
                      'INDIA' : ['IND']}
    
    for region in region_mapping.keys() : 
        filter = df.country.isin(region_mapping[region])
        df_filter = df[filter].copy()   
        logger.info(f'Computing data and shares for typical days, month etc.. for region {region}')
        df_td_hour, df_month_sub, df_day_sub,df_hour_sub = main(data_root = data_root, df = df_filter)
        day_correspondance = ['WD','SAT','SUN']
        season_correspondance = ['Summer','Winter']
    
    ## Save data
        final_path = data_root.parent.absolute().joinpath(f'models/logs/version_{version_number}/{region}_Load_Model_LC.xlsx')
        logger.info(f'Saving data of region {region} here: {final_path}')
        with pd.ExcelWriter(final_path) as writer : 
            df_td_hour.to_excel(writer, sheet_name = 'Data')
            df_month_sub.to_excel(writer, sheet_name = 'ShareMonths')
            for i in df.month.unique().tolist() : 
                df_day_sub[i].to_excel(writer, sheet_name = f'ShareDays_{i:02d}')
            for i in range(2) : 
                for j in range(3) :
                    df_hour_sub[(i,j)].to_excel(writer , sheet_name = f'ShareHours_{day_correspondance[j]}_{season_correspondance[i]}')
    logger.info('All Done ! ')