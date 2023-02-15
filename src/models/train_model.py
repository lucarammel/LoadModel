import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Mapping

import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import seed_everything
import os

from src.data.load_processed_data import load_processed_df ,compute_training_tensors,combine_subsectors
from src.models.model import LoadSeparator

import yaml

def main(project_dir: Path, args: Namespace,not_use_wem_inputs :bool,subsector_mapping : Mapping[str,Mapping],ci_ratio : Mapping[str, float],hparams : Mapping):
    logger_info = logging.getLogger(__name__)
    
    
    df_train = load_processed_df(project_dir, prediction = False, 
                                 scenario='historical', countries=[
        'FRA', 'DEU', 'ITA', 'FIN', 'NOR',
        'SGP', 'SWE', 'GBR', 'DNK',
        'US_CA', 'US_FL', 'US_NY','IND'
       # 'FRA_RES', 'FRA_IND_SER',
    ],not_use_wem_inputs = True)
    
    country_file_path = project_dir.joinpath('data/mapping/country_mapping.csv')
    country_mapping = pd.read_csv(country_file_path)
    economies = country_mapping[['country','economy']]
    economy_mapping = {key:value for (key,value) in zip(economies.country,economies.economy)}                                
                                                         
    df_train = combine_subsectors(df_train, subsector_mapping=subsector_mapping,not_use_wem_inputs = True)
    inputs_train, outputs_train, features, load_std = compute_training_tensors(project_dir, df_train, subsector_mapping=subsector_mapping,
                                                                   match_annual_demand=True,not_use_wem_inputs = True)
    countries = list(df_train.country.unique())
    
    hparams['countries'] =  countries
    hparams['features'] =  features
    hparams['load_std'] =  load_std
    hparams['WEM_inputs'] = not not_use_wem_inputs
                                                                   
    if hparams['WEM_inputs'] : 
        df_train_wem = load_processed_df(project_dir, prediction = False, scenario='historical', countries=[
            'FRA', 'DEU', 'ITA', 'FIN', 'NOR',
            'SGP', 'SWE', 'GBR', 'DNK',
            'US_CA', 'US_FL', 'US_NY','IND'
            # 'FRA_RES', 'FRA_IND_SER',
           ],not_use_wem_inputs = False)   
                                                         
        df_train_wem = combine_subsectors(df_train_wem, subsector_mapping=subsector_mapping,not_use_wem_inputs = False)
        inputs_train_wem, outputs_train_wem, features_wem, load_std_wem = compute_training_tensors(project_dir, df_train_wem, subsector_mapping=subsector_mapping,
                                                                   match_annual_demand=True,not_use_wem_inputs = False)
        hparams['load_std'] = load_std_wem
    
    # Create torch datasets and loaders
    loaders_train = {}
    loaders_train_wem = {}
    #for c in countries + ['all']:
    for c in countries + ['advanced_economies','developping_economies'] : 
        dataset_train = torch.utils.data.TensorDataset(inputs_train[c], outputs_train[c])
        loaders_train[c] = torch.utils.data.DataLoader(dataset_train, batch_size=hparams['batch_size'], shuffle=True)
        if hparams['WEM_inputs'] : 
            if c == 'DNK' : pass
            else : 
                dataset_train_wem = torch.utils.data.TensorDataset(inputs_train_wem[c],outputs_train_wem[c])
                loaders_train_wem[c] = torch.utils.data.DataLoader(dataset_train_wem,batch_size = hparams['batch_size'],shuffle = True)
        logger_info.info(f'Data of country {c} prepared for training ')
        
        
    # Train the generic model
    #callbacks = []
    #logger_info.info('Training the generic model on all countries')
    #shared_logger = TensorBoardLogger(str(project_dir.joinpath('models/logs')), name='')
    #log_dir = project_dir.joinpath(f'models/logs/version_{shared_logger.version}')
    #trainer = pl.Trainer.from_argparse_args(args, logger=shared_logger,callbacks = callbacks,max_epochs = hparams['max_epochs'])
    #shared_model = LoadSeparator(hparams=hparams, country='all')
    #trainer.fit(shared_model, loaders_train['all'])
    #trainer.save_checkpoint(log_dir.joinpath('trained_model.ckpt'))
    
    # Define the number of the experiment version
    liste = os.listdir(str(project_dir.joinpath('models/logs')))
    maximum = 'version_0'
    for l in liste : 
        if 'version' in l : 
            if int(l.split(sep = '_')[1]) > int(maximum.split(sep = '_')[1]) : 
                maximum = l
    version_number = int(maximum.split(sep = '_')[1]) + 1
    
    seed_everything(42, workers = True)
    
    #Train the model on the different types of economies (advanced/developping)
    for country_economy in ['developping_economies','advanced_economies'] :
        callbacks = []
        logger_info.info(f'Training the generic model on {country_economy} countries')
        shared_logger = TensorBoardLogger(str(project_dir.joinpath('models/logs')), name='',version = version_number)
        log_dir = project_dir.joinpath(f'models/logs/version_{shared_logger.version}')
        trainer = pl.Trainer.from_argparse_args(args, logger=shared_logger,callbacks = callbacks,max_epochs = hparams['max_epochs'],deterministic = True)
        country_economy_model = LoadSeparator(hparams=hparams, country=country_economy)
        trainer.fit(country_economy_model, loaders_train[country_economy])
        trainer.save_checkpoint(log_dir.joinpath(f'trained_model_{country_economy}.ckpt'))

    # Train country-specific models, starting with the generic model's weights
    for c in countries:
        country_economy = economy_mapping[c]
        logger_info.info(f'Training the model on country {c}')
        logger = TensorBoardLogger(str(log_dir), name='', version=c)
        trainer = pl.Trainer.from_argparse_args(args, logger=logger, callbacks = callbacks,max_epochs = hparams['max_epochs'],deterministic = True)
        country_model = LoadSeparator.load_from_checkpoint(str(log_dir.joinpath(f'trained_model_{country_economy}.ckpt')), country=c)
        #trainer.fit(country_model, train_dataloader=loaders[c])
        trainer.fit(country_model, train_dataloader = loaders_train[c])
        trainer.save_checkpoint(log_dir.joinpath(f'{c}/trained_model.ckpt'))
    
    #Train the model on hourly load by end-use starting from the country specific weights
    if hparams['WEM_inputs'] : 
        logger_info.info('Moving on to train the model on WEM hourly load data ')
        for c in countries : 
            if c == 'DNK' : pass
            else : 
                hparams['not_use_wem_inputs'] = False
                hparams['max_epochs_wem'] = int(hparams['max_epochs']*hparams['epochs_wem'])
                logger_info.info(f'Training country {c} on WEM hourly load data')
                logger = TensorBoardLogger(str(log_dir),name ='',version = c)
                trainer = pl.Trainer.from_argparse_args(args,logger = logger, callbacks = callbacks, max_epochs = hparams['max_epochs_wem'])
                with open(str(log_dir.joinpath('hparams_wem.yaml')), 'w') as yamlfile:
                    yaml.dump(hparams, yamlfile)
                country_model = LoadSeparator.load_from_checkpoint(str(log_dir.joinpath('trained_model.ckpt')),hparams_file = str(log_dir.joinpath('hparams_wem.yaml')),country = c)
                trainer.fit(country_model, train_dataloader = loaders_train_wem[c])
                trainer.save_checkpoint(log_dir.joinpath(f'{c}/trained_model.ckpt'))

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger_info = logging.getLogger(__name__)
    inp = input('ETP inputs ? True or False  -- If False, will use WEM inputs')
    if inp == 'True':
        not_use_wem_inputs = True   
    elif inp == 'False' : 
        not_use_wem_inputs = False
    logger_info.info(f'The algorithm will train using WEM inputs set on : {not not_use_wem_inputs}')
    #not_use_wem_inputs = False
    project_dir = Path(__file__).resolve().parents[2]
    parameters_path = project_dir.joinpath('models/logs')
    # cf https://pytorch-lightning.readthedocs.io/en/latest/introduction_guide.html#hyperparameters
    parser = ArgumentParser()
    # program level args
    parser.add_argument('--skip_plot', help="Don't plot anything after training to speed things up",
                        action='store_true')
    # add all the available trainer options to argparse (e.g --fast_dev_run)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    with open(parameters_path.joinpath('hparams_init.yaml'), 'r') as f:
        hparams = yaml.safe_load(f)
    with open(parameters_path.joinpath('ci_ratio.yaml'), 'r') as f:
        ci_ratio = yaml.safe_load(f)
    with open(parameters_path.joinpath('subsector_mapping.yaml'), 'r') as f:
        subsector_mapping = yaml.safe_load(f)
        
    main(project_dir = project_dir, args = args, not_use_wem_inputs = not_use_wem_inputs,ci_ratio = ci_ratio, subsector_mapping = subsector_mapping, hparams = hparams)
