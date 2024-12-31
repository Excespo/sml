import numpy as np
import sdf
from scipy.interpolate import RegularGridInterpolator

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import logging
from tqdm import tqdm

class LookUpTable(nn.Module):

    def __init__(self, lut_path):
        super().__init__()
        self.lut = sdf.load(
            filename=lut_path,
            objectname='/q',
            unit='W/m2',
            scale_units=['kg/(m2.s)', '1', 'Pa']
        )
        self.mass_flux_grid = torch.tensor(self.lut.scales[0].data, requires_grad=False)
        self.quality_grid = torch.tensor(self.lut.scales[1].data, requires_grad=False) 
        self.pressure_grid = torch.tensor(self.lut.scales[2].data, requires_grad=False)
        self.data = torch.tensor(self.lut.data, requires_grad=False)

        self.cpu_interpolator = RegularGridInterpolator(
            (self.mass_flux_grid.numpy(), self.quality_grid.numpy(), self.pressure_grid.numpy()),
            self.data.numpy(),
            method='linear',
            bounds_error=False,
            fill_value=None
        )

    def forward(self, pressure, mass_flux, quality):
        return self.cpu_interpolator((mass_flux, quality, pressure))

class LiuModel(nn.Module):
    def __init__(self, batch_size=1):
        super(LiuModel, self).__init__()
        self.batch_size = batch_size
        self.calculate_rho_f = nn.Linear(batch_size, 1)
        self.calculate_h_fg = nn.Linear(batch_size, 1)
        # init!
        self.calculate_rho_f.weight.data = torch.ones(1, batch_size) * 1000.0 / 32
        self.calculate_rho_f.bias.data = torch.zeros(1, batch_size)
        self.calculate_h_fg.weight.data = torch.ones(1, batch_size) * 2260e3 / 32
        self.calculate_h_fg.bias.data = torch.zeros(1, batch_size)
        pass  # No trainable parameters in the basic mechanistic 
    

    def forward(self, inputs):
        """
        inputs: Tensor of shape (batch_size, num_features)
                Expected features: pressure, mass_flux, D_e, D_h, length
        """
        # def calculate_rho_f(pressure):
        #     return 1000.0  # Placeholder for actual calculation

        # def calculate_h_fg(pressure):
        #     return 2260e3  # Placeholder for actual calculation

        
        pressure, mass_flux, D_e, D_h, length = inputs[:, 0], inputs[:, 1], inputs[:, 2], inputs[:, 3], inputs[:, 4]
        
        pressure_expanded = pressure.unsqueeze(1).expand(-1, self.batch_size)

        # Convert D_e, D_h, length from mm to meters if necessary
        D_e = D_e / 1000.0  # meters
        D_h = D_h / 1000.0  # meters
        length = length / 1000.0  # meters
        
        # Calculate fluid properties
        rho_f = self.calculate_rho_f(pressure_expanded)  # kg/m³
        h_fg = self.calculate_h_fg(pressure_expanded)  # J/kg
        
        # Compute liquid sublayer thickness (δ)
        # Placeholder: Implement Liu's correlation for δ
        # Example: δ = C1 * (mass_flux)^C2 * (pressure)^C3
        # Here, we'll use a simple relation for demonstration
        delta = 1e-5 * mass_flux ** 0.5 / pressure  # meters
        
        # Compute vapor blanket velocity (U_B)
        # Placeholder: Implement Liu's correlation for U_B
        U_B = mass_flux / (rho_f * D_e)  # m/s
        
        # Compute vapor blanket length (L_B)
        # Placeholder: Implement Liu's correlation for L_B
        L_B = D_h  # meters, assuming proportional to hydraulic diameter
        
        # Compute CHF
        CHF = (rho_f * delta * h_fg) / (L_B * U_B)  # W/m²
        
        return CHF
    

class FeedForwardNetwork(nn.Module):
    def __init__(self, layer_dims=[8, 50, 50, 50, 1], activation=nn.ReLU, dropout_p=0.5):
        super().__init__()
        self.layer_dims = layer_dims
        self.activation = activation
        self.dropout_p = dropout_p
        
        self.dtype = torch.float32 # ensure using float32
        self._build_layers()
        self._init_weights()

    def _build_layers(self):
        self.hidden_layers = []
        for input_layer_dim, output_layer_dim in zip(self.layer_dims[:-2], self.layer_dims[1:-1]):
            self.hidden_layers.extend([
                nn.Linear(input_layer_dim, output_layer_dim),
                # nn.BatchNorm1d(output_layer_dim),
                self.activation(),
                # nn.Dropout(p=self.dropout_p)
            ])

        self.hidden_layers = nn.Sequential(*self.hidden_layers)
        self.output_layer = nn.Linear(self.layer_dims[-2], self.layer_dims[-1])
        # self.output_bn = nn.BatchNorm1d(self.layer_dims[-1])

    def _init_weights(self):
        # for layer in self.hidden_layers:
        #     if isinstance(layer, nn.Linear):
        #         nn.init.kaiming_normal_(layer.weight)
        #         nn.init.constant_(layer.bias, 0)
        # nn.init.kaiming_normal_(self.output_layer.weight)
        # nn.init.constant_(self.output_layer.bias, 0)
        for layer in self.hidden_layers:
            if isinstance(layer, nn.Linear):
                nn.init.uniform_(layer.weight, -0.1, 0.1)
                nn.init.constant_(layer.bias, 0)
        nn.init.uniform_(self.output_layer.weight, -10, 10) 
        nn.init.constant_(self.output_layer.bias, 0)

    def forward(self, x):
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        # x = self.output_bn(x)
        return x

class HybridModel(nn.Module):
    def __init__(self, physical_model):
        super().__init__()
        if physical_model == "liu":
            self.physical = LiuModel()
        elif physical_model == "lut":
            self.physical = LookUpTable()
        else:
            raise ValueError(f"Unknown physical model: {physical_model}")
        
        self.ffn = FeedForwardNetwork()

    def forward(self, x):
        lut_output = self.look_up_table(x)
        ffn_output = self.feed_forward_network(x)
        return lut_output + ffn_output
