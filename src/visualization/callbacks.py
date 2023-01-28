import logging

import pytorch_lightning as pl
from pytorch_lightning import Callback
import pandas as pd

from src.visualization.visualize import get_profiles_figures


class PlottingCallback(Callback):
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        logger = logging.getLogger(__name__)
        logger.info(f'Plotting figures for country {pl_module.country}')

        x, _ = trainer.train_dataloader.dataset.tensors  # (N, T, f+k)
        outputs = {'real_scale': pl_module(x),
                   'normalized': pl_module(x, normalized_load=True)}

        for scale in ['real_scale', 'normalized']:
            y = outputs[scale]
            fig_month, fig_is_we = get_profiles_figures(y, self.df, pl_module.country)

            # Save figures to TensorBoard
            # - removed as it doesn't work on IEA machines, without the torchvision package
            # trainer.logger.experiment.add_figure(f'real_scale/month/{pl_module.country}', fig_month)
            # trainer.logger.experiment.add_figure(f'real_scale/is_we/{pl_module.country}', fig_is_we)

            # Save figures as pdf files
            fig_month.savefig(f'{trainer.logger.experiment.log_dir}/profile_month_{scale}.pdf')
            fig_is_we.savefig(f'{trainer.logger.experiment.log_dir}/profile_is_weekend_{scale}.pdf')


        
    
    