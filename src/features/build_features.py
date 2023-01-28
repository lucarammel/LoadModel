import logging

import holidays
import pandas as pd
import numpy as np


def compute_features(df: pd.DataFrame, country: str, split_week_end : bool):
    """

    Args:
        df: DataFrame containing hourly load and weather data for a country. Needs to have
            a `local_timestamp` DateTimeIndex column,
            a `temperature` column
        country:

    Returns:
        The same df, with additional computed features (e.g. hour of day, year, etc)

    """
    logger = logging.getLogger(__name__)

    # Extract useful features from timestamp
    
    df['year'] = df.local_timestamp.dt.year
    df['month'] = df.local_timestamp.dt.month
    df['week_of_year'] = df.local_timestamp.dt.weekofyear
    df['day_of_year'] = df.local_timestamp.dt.dayofyear
    df['day_of_week'] = df.local_timestamp.dt.dayofweek
    df['hour_of_year'] = (df.day_of_year - 1) * 24 + df.local_timestamp.dt.hour
    df['hour_of_day'] = df.local_timestamp.dt.hour
    df['is_weekend'] = (df.day_of_week == 5) | (df.day_of_week == 6)

    # Define activity and occupation profiles for RES, SER and RES LI
    #pic_hour = [0,8,20,26.5]
    #sigma = [1.5,4.5,3,1.5]
    
    #def activity(x : list,sigma : int,pic_hour : list) : 
     #   from scipy.stats import norm 
      #  pdf = 0
      #  for idx,m in enumerate(pic_hour) : 
      #      if idx in [0,3] : 
      #          pdf += norm.pdf(x,loc = m, scale = sigma[idx])/4
      #      else : 
      #          pdf += norm.pdf(x,loc = m, scale = sigma[idx])
      #  return pdf /max(pdf)
    #df['activity_rate'] = activity(df.hour_of_day,sigma,pic_hour)
    
    #pic_hour_occ = [10,13,16,20]
    #sigma_occ = [2,2,2,3]

    #def occupation(x : list,pic_hour_occ : list, sigma_occ : list) : 
     #   from scipy.stats import norm 
      #  pdf = 0
       # for idx, m in enumerate(pic_hour_occ) : 
        #    pdf += norm.pdf(x,loc = m, scale = sigma_occ[idx])*1.2
       # return (pdf + 0.1)
    
   # df['services_rate'] = occupation(df.hour_of_day,pic_hour_occ,sigma_occ) #services occupation
   # df['occupation_rate'] = 1 - df['services_rate'] #home occcupation
    
    activity = [5,6,7,8,9,10,18,19,20,21,22,22]
    services = [i for i in range(9,23)]
    occupation = [0,1,2,3,4,5,6,7,8,19,20,21,22,23]
    df['activity_rate'] = df.apply(lambda row :row.hour_of_day in activity , axis = 1)
    df['services_rate'] = df.apply(lambda row :row.hour_of_day in services , axis = 1)
    df['occupation_rate'] = df.apply(lambda row :row.hour_of_day in occupation , axis = 1)
    
    if split_week_end : 
        df['is_saturday'] = df.day_of_week == 5
        df['is_sunday'] = df.day_of_week == 6
        df['weekday'] = np.nan 
        df['weekday'] = df.apply(lambda row : '0' if row.is_weekend == False else ('1' if row.is_saturday == True else '2'),axis = 1)
    
    # Transform hours, month, etc in cyclical features for continuity between 23:00 and 00:00, dec. and jan., etc
    df['cos_h'] = np.cos(df.hour_of_day * 2 * np.pi / 24)
    df['sin_h'] = np.sin(df.hour_of_day * 2 * np.pi / 24)
    df['cos_m'] = np.cos(df.month * 2 * np.pi / 12)
    df['sin_m'] = np.sin(df.month * 2 * np.pi / 12)
    df['cos_w'] = np.cos(df.week_of_year * 2 * np.pi / 52)
    df['sin_w'] = np.sin(df.week_of_year * 2 * np.pi / 52)

    # Compute higher harmonics hoping to extract more complex patterns : has not proven useful
    # with neural networks (as they already are able to model non-linear relations), adding noise instead
    # df['cos_2h'] = np.cos(df.hour_of_day * 2 * 2 * np.pi / 24)
    # df['sin_2h'] = np.sin(df.hour_of_day * 2 * 2 * np.pi / 24)
    # df['cos_3h'] = np.cos(df.hour_of_day * 3 * 2 * np.pi / 24)
    # df['sin_3h'] = np.sin(df.hour_of_day * 3 * 2 * np.pi / 24)

    # Temperature related features
    df['temperature_1d_mean'] = df.temperature.rolling(window=24, min_periods=1).mean()
    # todo : add temperatures at previous timestamps in the features, as a way to account for buildings thermal lag ?
    # for delta_h in range(1, temperature_max_lag):
    #     df['temperature_t-{}'.format(delta_h)] = df.temperature.shift(periods=delta_h, fill_value=15.)

    # Holidays
    if country in holidays.list_supported_countries():
        # Classify national holidays as week-ends
        h = holidays.CountryHoliday(country, years=df.year.unique())
        df.loc[df.local_timestamp.dt.date.isin(h), 'is_weekend'] = True
    else:
        logger.warning(f'Could not find holidays for country {country}')

    logger.info(f'Features computed for country {country}')
    return df
