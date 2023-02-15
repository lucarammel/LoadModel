

import pytorch_lightning as pl

from typing import List, Sequence, Mapping

import torch
from torch import nn
from torch.nn.modules.loss import _Loss



class LoadSeparator(pl.LightningModule):
    def __init__(self, hparams: dict, country: str):
        super(LoadSeparator, self).__init__()
        self.save_hyperparameters(hparams)
        self.country = country
        self.not_use_wem_inputs = hparams['not_use_wem_inputs']
        subsector_mapping = hparams['subsector_mapping']
        self.subsectors = sorted(subsector_mapping.keys())

        self.main_net = MainNet(subsector_mapping=subsector_mapping, features=hparams['features'],
                                dim_hidden_shared=hparams['dim_hidden'],
                                dropout_rate = hparams['dropout_rate'],
                                tcn_channels=hparams.get('tcn_channels', []),
                                tcn_kernel_size=hparams.get('tcn_kernel_size', None),
                                dim_hidden_lstm = hparams.get('dim_hidden_lstm',0),
                                n_layers = hparams['n_layers'],
                                not_use_wem_inputs = hparams['not_use_wem_inputs'],
                                hparams = hparams)

        self.mse_total_load = TotalLoadMSE()
        self.heating_penalty = []
        self.cooling_penalty = []
        self.lighting_penalty_res = []
        self.lighting_penalty_ser = []
        self.interval_loss = IntervalLoss(h_ratio = hparams['h_ratio'])
        for i, subsector in enumerate(self.subsectors):
            subsector_conf = subsector_mapping[subsector]
            if subsector_conf.get('heating_penalty'):
                weight = subsector_conf['heating_penalty']['weight']
                threshold = subsector_conf['heating_penalty']['threshold']
                self.heating_penalty.append(TemperaturePenalty(subsector_index=i, cooling=False, weight=weight,
                                                               temperature_threshold=threshold))
            if subsector_conf.get('cooling_penalty'):
                weight = subsector_conf['cooling_penalty']['weight']
                threshold = subsector_conf['cooling_penalty']['threshold']
                self.cooling_penalty.append(TemperaturePenalty(subsector_index=i, cooling=True, weight=weight,
                                                               temperature_threshold=threshold))
            if subsector_conf.get('irradiance_penalty_res'):
                weight = subsector_conf['irradiance_penalty_res']['weight']
                threshold = subsector_conf['irradiance_penalty_res']['threshold']
                self.lighting_penalty_res.append(LightingPenalty(subsector_index = i,weight = weight,res = True, irradiance_threshold = threshold,
                                                                  not_use_wem_inputs = self.not_use_wem_inputs))
            if subsector_conf.get('irradiance_penalty_ser'):
                weight = subsector_conf['irradiance_penalty_ser']['weight']
                threshold = subsector_conf['irradiance_penalty_ser']['threshold']
                self.lighting_penalty_ser.append(LightingPenalty(subsector_index = i,weight = weight,res = False, irradiance_threshold = threshold,
                                                                  not_use_wem_inputs = self.not_use_wem_inputs))
                
    def forward(self, x: torch.Tensor, normalized_load=False):
        #if self.not_use_wem_inputs : 
        return self.main_net(x, normalized_load=normalized_load)
        

    def configure_optimizers(self):
        if self.hparams['optimizer_name'] == 'adam':
                # other optimizers could be used, but adam works fine (and converges faster than stochastic gradient
                # descent)
            return torch.optim.Adam(self.parameters(), lr=self.hparams['lr'])
        elif self.hparams['optimizer_name'] == 'adamax': 
            return torch.optim.Adamax(self.parameters(), lr = self.hparams['lr'])

    def training_step(self, batch, batch_idx):
        if self.not_use_wem_inputs : 
            x, y = batch
            y_hat = self(x)
            #Compute the scores
            #mean squared error on total load
            mse = self.mse_total_load(y_hat, y)
            # the final score that we try to minimize
            loss = mse 
            log = {'loss': loss, 'mse': mse}
           # print(f'epoch {self.current_epoch}',f'loss : {loss.item()}')
            for penalty in self.heating_penalty + self.cooling_penalty:
                # Compute the heating/cooling penalty
                p = penalty.weight * penalty(y_hat, y)
                loss += p
                # And add it to the scores to be logged
                # e.g 'heating_penalty_CLU_SH'
                name = f'{"cooling" if penalty.cooling else "heating"}_penalty_{self.subsectors[penalty.subsector_index]}'
                log[name] = p
            for penalty in self.lighting_penalty_res + self.lighting_penalty_ser : 
                p =  penalty(y_hat,y)
                loss += p
                name = f'{"res" if penalty.res else "ser"}_penalty_{self.subsectors[penalty.subsector_index]}'
                log[name] = p
            print('\r'+ f"epoch {self.current_epoch+1}/{self.hparams['max_epochs']} & loss : {loss.item()}", end = '')
            return {'loss': loss, 'log': log}
        
        else :
            #Computing the mse error on each subsector and making the sum. 
            x, y = batch
            y_hat = self(x)
            interval_loss =  self.interval_loss(y_hat,y)
            mse = self.mse_total_load(y_hat,y,not_use_wem_inputs = False)
            loss = interval_loss + interval_loss.item()*mse
            print('\r'+ f"epoch {self.current_epoch+1}/{self.hparams['max_epochs_wem']} & loss : {loss.item()}", end = '')
            log = {'loss' : loss , 'interval_loss' : interval_loss}
            return( {'loss' : loss , 'log' : log})


class MainNet(nn.Module):
    def __init__(self, subsector_mapping: Mapping[str, Mapping], features: Sequence[str],
                 dim_hidden_shared: int,dropout_rate:float, hparams : dict ,tcn_channels: List[int],dim_hidden_lstm : int,
                 n_layers : int, tcn_kernel_size: int,not_use_wem_inputs :bool):
        """

        Args:
            subsector_mapping:
            features:
            dim_hidden_shared: Size of the hidden layer
            tcn_channels:
            tcn_kernel_size:
        """
        super().__init__()
        self.subsector_count = len(subsector_mapping.keys())
        sub_features_indexes = [
            [features.index(f) for f in subsector_mapping[subsector]['features']]
            for subsector in sorted(subsector_mapping.keys())
        ]
        feature_count = len(features)
        self.annual_demand_indexes = list(range(feature_count, feature_count + self.subsector_count))

        # Individual networks for every subsector
        if len(tcn_channels) == 0 and dim_hidden_lstm == 0:
            self.sub_nets = nn.ModuleList([
                SubsectorNet(dim_hidden=dim_hidden_shared,hparams = hparams,not_use_wem_inputs = not_use_wem_inputs,
                             dropout_rate = dropout_rate, features_indexes=sub_features_indexes[i])
                             
                for i in range(self.subsector_count)
            ])
        elif dim_hidden_lstm != 0 : 
            self.sub_nets = nn.ModuleList([SubsectorNetLSTM(dim_hidden_lstm = dim_hidden_lstm, n_layers = n_layers,
                                                            hparams = hparams, features_indexes = sub_features_indexes[i]) 
                                           for i in range(self.subsector_count)])
        else:
            self.sub_nets = nn.ModuleList([
                SubsectorTCNNet(features_indexes=sub_features_indexes[i], num_channels=tcn_channels,
                                kernel_size=tcn_kernel_size)
                for i in range(self.subsector_count)
            ])

    def forward(self, x: torch.Tensor, normalized_load=False):
        # Predicted un-normalized load for every subsector, using the shared network
        y_sub_list = [m(x) for m in self.sub_nets]  # k x (N, T, 1)
        y_sub = torch.cat(y_sub_list, dim=2)  # N, T, k

        # For every sector, normalize subsector yearly load to 1
        y_sub = nn.functional.softmax(y_sub, dim=1)

        if normalized_load:
            return y_sub
        
        # And scale it up to (scaled) MW using subsector annual demands (scaled MWh)
        demand = x[:, :, self.annual_demand_indexes]
        y_sub = torch.mul(y_sub, demand)
        return y_sub



class SubsectorNet(pl.LightningModule): 
    def __init__(self, dim_hidden: int, dropout_rate: float, hparams : dict,
                        features_indexes: Sequence[int],not_use_wem_inputs : bool):
        super().__init__()
        self.D_in = len(features_indexes)
        self.features_indexes = features_indexes
        self.linear1 = nn.Linear(self.D_in, dim_hidden)
        self.linear2 = nn.Linear(dim_hidden, 1)
        self.dropout = nn.Dropout(p = dropout_rate)
        self.not_use_wem_inputs = not_use_wem_inputs
        self.save_hyperparameters()
        
    def forward(self, x :torch.Tensor) :
        if self.not_use_wem_inputs : 
            m = nn.Sigmoid()
            h_relu = m(self.linear1(x[:, :, self.features_indexes]))
            y_pred = self.linear2(h_relu) #The output is the hourly load for a whole set of 8760 values in a batch 
        if not self.not_use_wem_inputs : 
            m = nn.Sigmoid()
            h = self.dropout(m(self.linear1(x[:, :, self.features_indexes])))
            y_pred = self.linear2(h)
        return y_pred
    
#Experiment with a LSTM architecture for a time series approach. 
#See here for more details (https://towardsdatascience.com/lstms-in-pytorch-528b0440244)

class SubsectorNetLSTM(pl.LightningModule) : 
    def __init__(self, dim_hidden_lstm : int, n_layers : int, hparams : dict,features_indexes : Sequence[int]):
        super().__init__()
        self.D_in = len(features_indexes)
        self.features_indexes = features_indexes
        self.lstm = nn.LSTM(self.D_in,dim_hidden_lstm, n_layers, batch_first = True)
        self.linear1 = nn.Linear(dim_hidden_lstm, 1)
        self.dim_hidden = dim_hidden_lstm
        self.save_hyperparameters()
        
    def forward(self, x : torch.Tensor):
        lstm_out ,hs = self.lstm(x[:,:, self.features_indexes])        
        linear_out = self.linear1(lstm_out)
        return(linear_out)
        
        
    
# --- Experiment with Temporal Convolution Networks (https://github.com/locuslab/TCN)
# Would allow for a more 'time-series' approach using previous values of inputs as well (e.g temperature
# during the past hours). The first results were not really better than the base MLP, but maybe some wins possible
# with more tuning ?

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size]  # .contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1)

    def forward(self, x):
        return self.net(x)

class SubsectorTCNNet(nn.Module):
    def __init__(self, features_indexes: Sequence[int], num_channels: List[int], kernel_size: int):
        super().__init__()
        self.features_indexes = features_indexes
        layers = []
        num_inputs = len(features_indexes)
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size)]
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        # From x of shape (N, T, N_features+N_subsectors),
        # Get input of shape (N, f_subsector, T) for the TCN
        subsector_input = x[:, :, self.features_indexes].transpose(1, 2)
        return self.network(subsector_input).transpose(1, 2)


# --- Loss functions

class TotalLoadMSE(_Loss):
    def forward(self, input: torch.Tensor, target: torch.Tensor, not_use_wem_inputs = True):
        if not_use_wem_inputs : 
            N, T, k = input.shape
            assert target.shape == (N, T, 5) # load & temperature & irradiance & services & activity
        else : 
            N, T, k = input.shape
            assert target.shape == (N, T, 12)
        # sum across all sectors to get total load
        return nn.functional.mse_loss(torch.sum(input, dim=2), target[:, :, 0].view(N, T))
         


class TemperaturePenalty(_Loss):
    def __init__(self, subsector_index: int, cooling: bool, weight: float, temperature_threshold: float):
        """
        Return a penalty for this subsector when the temperature is under/over the temperature threshold.
            penalty_t = (load_t * delta_T)**2
                for t in [1, 8760]

        With
            delta_T = max(0, T_threshold - T_t)     if cooling
            delta_T = max(0, T_t - T_threshold)     else

        Args:
            subsector_index:
            cooling: If True, return a penalty for this subsector when the temperature is under the threshold
                (as there should be no cooling).
                If False, same but when the temperature is over the threshold (as there should be no heating).
            weight: Just for reference, not actually used here
            temperature_threshold:
        """
        super().__init__()
        self.subsector_index = subsector_index
        self.cooling = cooling
        self.threshold = temperature_threshold
        self.weight = weight

    def forward(self, input: torch.Tensor, target: torch.Tensor): #may be update with a penalty on time 
        N, T, k = input.shape
        assert target.shape == (N, T, 5)  # load & temperature & irradiance & services & activity

        if self.cooling:
            # Penalize temperatures under the threshold (there should be no cooling)
            penalty = (-target[:, :, 1] + self.threshold).clamp(min=0)
        else:
            # Penalize temperatures over the threshold
            penalty = (target[:, :, 1] - self.threshold).clamp(min=0)

        # Add a penalty proportional to the square of load * temperature delta
        return nn.functional.mse_loss(input[:, :, self.subsector_index] * penalty, torch.zeros(N, T))

class IntervalLoss(_Loss):
    def __init__(self, h_ratio : Mapping[str, float]):
        """
        
        This loss aims to train the algorithm on fuzzy training data as WEM outputs are (because themselves 
        are predicted based on model hypothesis and litterature knwoledge) 
    
        loss = sum_t { max(0,(1-h)y_t - y_predict) + max(0,y_predict - (1+h)y_t) }
        
        Where the confidence interval is : [(1-h)y_t , (1+h)y_t]. It means that there is a penalty when 
        the y_predict value is not included in the interval. 
                
        """
        
        super().__init__()
        self.h_ratio = h_ratio 
        
    def forward(self, output : torch.Tensor , target : torch.Tensor) : 
        N,T,k = output.shape
        assert target.shape == (N,T,12) #total actual load & load for each cluster (7) & temperature & lighting & services rate & activity
        
        max_lower = torch.zeros(N,T)
        max_upper = torch.zeros(N,T)
        for load_idx , ci_ratio in enumerate(self.h_ratio.keys()):
            if load_idx == 1  : pass
            else : 
                lower_bound = (1-self.h_ratio[ci_ratio])* target[:,:,load_idx+1].view(N,T) - output[:,:,load_idx].view(N,T)
                max_lower += torch.maximum(torch.zeros(N,T), lower_bound)
                upper_bound = output[:,:,load_idx].view(N,T) - (1+self.h_ratio[ci_ratio])* target[:,:,load_idx+1].view(N,T)
                max_upper += torch.maximum(torch.zeros(N,T), upper_bound)
            
        return nn.functional.l1_loss(torch.sum(max_upper+max_lower),torch.tensor(0))
        

class LightingPenalty(_Loss) : 
    def __init__(self,subsector_index : int, res : bool, weight : float, irradiance_threshold : float, not_use_wem_inputs : bool):
        
        """
        This loss aims to add a penalty when the irradiance surface is higher than a certain threshold. Because 
        lighting is unlikely to happen when the irradiance surface is high. 
        
        penalty_hour = 1-occupation_rate
        penalty_light = max(0,irradiance_surface_t - irradiance_threshold)
        loss =  (load_predicted_lighting_t*(weight * penalty_hour + penalty_light)^2
        
        """
        super().__init__()
        self.subsector_index = subsector_index 
        self.weight = weight
        self.irradiance_threshold =  irradiance_threshold
        self.not_use_wem_inputs = not_use_wem_inputs 
        self.res = res
    def forward(self, output : torch.Tensor , target : torch.Tensor) :
        N, T, k = output.shape
        if self.not_use_wem_inputs : 
            assert target.shape == (N, T, 5) # load & temperature & irradiance & services & activity
            if self.res : 
                irr = target[:, :, 2] - self.irradiance_threshold
                penalty = torch.maximum(torch.zeros(N,T), irr)
                penalty_time = torch.ones(N,T) - target[:,:,4]
            else : 
                irr = target[:, :, 2] - self.irradiance_threshold
                penalty = torch.maximum(torch.zeros(N,T), irr)
                penalty_time = torch.ones(N,T) - target[:,:,3]
            return nn.functional.mse_loss((self.weight * penalty_time + penalty)*output[:, :, self.subsector_index], torch.zeros(N, T))
            
        else : #to update
            assert target.shape == (N,T,12) #total actual load & load for each cluster (7) & temperature & lighting & services rate & activity
            irr = target[:, :, 2] - self.irradiance_threshold
            penalty = torch.maximum(torch.zeros(N,T), irr)
            penalty_time = torch.ones(N,T) - target[:,:,11]
            return nn.functional.mse_loss((self.weight * penalty_time + penalty)*output[:, :, self.subsector_index], torch.zeros(N, T))
            
            