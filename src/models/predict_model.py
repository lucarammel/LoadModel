import logging
from pathlib import Path
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
import yaml

from src.data.load_processed_data import load_processed_df, combine_subsectors, compute_predict_tensors, \
    get_df_results_all_countries
from src.models.model import LoadSeparator
#from src.models.res_non_res_separation import separate_res_non_res


def load_trained_model(project_dir: Path, version_name: str, country: str):
    if country == 'all':
        path = project_dir.joinpath(f'models/logs/{version_name}/trained_model.ckpt')
    else:
        path = project_dir.joinpath(f'models/logs/{version_name}/{country}/trained_model.ckpt')

    model = LoadSeparator.load_from_checkpoint(str(path), country=country)
    # not sure if it's necessary, but it's best practice to freeze model weights when using it for predictions only
    model.freeze()

    return model


def predict_results(project_dir: Path, scenario: str, version_number: int, separate_res_non_res_cooling: bool,not_use_wem_inputs : bool):
    logger = logging.getLogger(__name__)
    # Load experiment hyperparameters (what clustering to use, features, etc)
    if not_use_wem_inputs : 
        conf = OmegaConf.load(project_dir.joinpath(f'models/logs/version_{version_number}/hparams.yaml'))
    else : 
        conf = OmegaConf.load(project_dir.joinpath(f'models/logs/version_{version_number}/hparams_wem.yaml'))
    #logger.info(f'Loaded experiment {version_number}, using subsector mapping : \n'
     #           f'{conf.subsector_mapping.pretty()}')
    logger.info(f'Loaded experiment {version_number}, using subsector mapping : \n'
                f'{OmegaConf.to_yaml(conf.subsector_mapping)}')
    prediction = True
    # Load scenario data and compute tensors
    df_original = load_processed_df(project_dir, prediction = prediction, scenario=scenario, countries=[
        'FRA', 'DEU', 'ITA', 'FIN', 'NOR',
        'SWE', 'GBR', 'DNK',
        'US_CA', 'US_FL', 'US_NY', 
        'SGP', 'IND'
        # 'FRA_RES', 'FRA_IND_SER',
    ] , years='all',not_use_wem_inputs = not_use_wem_inputs) #years='all' or years=[2017,2050]
    df = combine_subsectors(df_original, subsector_mapping=conf.subsector_mapping,not_use_wem_inputs = not_use_wem_inputs,prediction = True) 
    inputs, features = compute_predict_tensors(project_dir,df, subsector_mapping=conf.subsector_mapping, load_std=conf.load_std,not_use_wem_inputs = not_use_wem_inputs)

    # Load trained models
    countries = list(df.country.unique())
    logger.info(f'Loading trained models for countries {countries}')
    models = {
        c: load_trained_model(project_dir, version_name=f'version_{version_number}', country=c) for c in countries
    }

    # Get results in a single data frame
    #if not_use_wem_inputs : 
    y_dict = {}
    for c in countries:
        country_model = models[c]
        y_dict[c] = country_model(x=inputs[c], normalized_load=False) * conf.load_std  # Outputs are scaled up to MW
    logger.info('Combining predict outputs from all models into a result data frame')
    df_results = get_df_results_all_countries(df=df, y_dict=y_dict,not_use_wem_inputs = not_use_wem_inputs)
    #else : 
     #   y_dict = {}
      #  for c in countries : 
       #     country_model = models[c]
        #    y_dict[c] = country_model(x = inputs[c],normalized_load = False) 
         #   df_load = conf.load_std.copy()
          #  del df_load['load']
           # for idx, key in enumerate(df_load.keys()):
            #    y_dict[c][:,:,idx] =  y_dict[c][:,:,idx] * df_load[key] 
    df_results = get_df_results_all_countries(df=df, y_dict=y_dict,not_use_wem_inputs = not_use_wem_inputs)
            
    # Separate res/non res cooling loads using weo data if we want to (and if it's not already separated)
    if separate_res_non_res_cooling and ('CLU_SC' in df_results.subsector.unique()): pass
        #df_results = separate_res_non_res(df_results, df_original, weo_data_file_name='EUa2017_hourly')
    else : 
        df_results.reset_index(inplace = True,drop = True)
    #Rescaling the annual sum of load predicted to the annual sum of actual load curve. (initially scaled to annual demand)
    years = {}
    df_hist = pd.DataFrame()
    for c in countries : 
        years[c] = list(df_results[df_results.country == c].year.unique())
    for c in countries:
        for y in years[c]:
            df_aux = df_results[(df_results.country == c) & (df_results.year == y)]
            load_sum = np.sum(df_aux.load)/len(df_results.subsector.unique().tolist())
            annual_demand = np.sum(df_aux.annual_demand.unique())
            scale = load_sum/annual_demand
            df_hist = pd.concat([df_hist , df_aux['load_predicted']*scale])
    df_results['load_predicted'] = df_hist
    return df_results


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    proj_dir = Path(__file__).resolve().parents[2]
    logger = logging.getLogger(__name__)

    version_number = int(input('Version number'))
    scenario  = str(input('Enter the scenario you want to predict : historical, SDS2050, STEPS2050'))
    separate_res_non_res_cooling = False
    with open(proj_dir.joinpath(f'models/logs/version_{version_number}/hparams.yaml'),'r') as file : 
        hparams = yaml.safe_load(file)
        not_use_wem_inputs = not hparams['WEM_inputs']
    df_res = predict_results(proj_dir, scenario=scenario, version_number=version_number,
                             separate_res_non_res_cooling=separate_res_non_res_cooling, not_use_wem_inputs = not_use_wem_inputs)

    # Save results to the disk
    output_path = proj_dir.joinpath(f'models/logs/version_{version_number}/output_{scenario}.csv')
    df_res.to_csv(output_path, index=False)
    logger.info(f"Results saved in "
                f"{proj_dir.joinpath(f'models/logs/version_{version_number}/output_{scenario}.csv')}")
