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
        # G massflux 
        # x quality
        # P pressure
        # q chf
        self.mass_flux_grid = self.lut.scales[0].data
        self.quality_grid = self.lut.scales[1].data
        self.pressure_grid = self.lut.scales[2].data
        self.chf = self.lut.data

    def _nearest_index(self, grid, value):
        value = value.detach().cpu().numpy()
        grid = np.asarray(grid)
        idx = np.searchsorted(grid, value)

        if idx == 0:
            return 0
        elif idx == len(grid):
            return len(grid) - 1
        
        if abs(value - grid[idx]) < abs(value - grid[idx - 1]):
            return idx
        else:
            return idx - 1

    def forward(self, mass_flux, quality, pressure):
       
        pressure = pressure * 1e6

        nearest_mass_flux = self._nearest_index(self.mass_flux_grid, mass_flux)
        nearest_quality = self._nearest_index(self.quality_grid, quality)
        nearest_pressure = self._nearest_index(self.pressure_grid, pressure)
        # print(f"mass flux grid: {self.mass_flux_grid}, mass flux: {mass_flux}, nearest: {nearest_mass_flux}")
        # print(f"quality grid: {self.quality_grid}, quality: {quality}, nearest: {nearest_quality}")
        # print(f"pressure grid: {self.pressure_grid}, pressure: {pressure}, nearest: {nearest_pressure}")

        chf = torch.tensor(
            self.chf[nearest_mass_flux, nearest_quality, nearest_pressure], 
            device=mass_flux.device
        )
        # print(f"chf: {chf}")

        return chf / 1e6 # In LUT unit is W/m2, convert to MW/m2

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
    def __init__(self, layer_dims=[8, 50, 50, 50, 1], activation=nn.ReLU, dropout_p=0.5, if_hybrid=False):
        super().__init__()
        self.layer_dims = layer_dims
        self.activation = activation
        self.dropout_p = dropout_p
        self.if_hybrid = if_hybrid
        
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
        if self.if_hybrid:
            nn.init.uniform_(self.output_layer.weight, -10, 10) 
        else:
            nn.init.uniform_(self.output_layer.weight, -1, 1) 
        nn.init.constant_(self.output_layer.bias, 0)

    def forward(self, x):
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        # x = self.output_bn(x)
        return x

class HybridModel(nn.Module):
    def __init__(self, physical_model, feature_ranges, layer_dims=[8, 50, 50, 50, 1], activation=nn.ReLU, dropout_p=0.5):
        super().__init__()
        self.physical_model = physical_model
        self.feature_ranges = feature_ranges
        if physical_model == "liu":
            self.physical = LiuModel()
        elif physical_model == "lut":
            self.physical = LookUpTable("thirdparty/2006_Groeneveld_CriticalHeatFlux_LUT/2006LUT.sdf")
        else:
            raise ValueError(f"Unknown physical model: {physical_model}")
        
        self.ffn = FeedForwardNetwork(
            layer_dims=layer_dims,
            activation=activation,
            dropout_p=dropout_p,
            if_hybrid=True
        )

    def forward(self, x):

        if self.physical_model == "lut":
            min_P, max_P = self.feature_ranges["pressure [MPa]"]["min"], self.feature_ranges["pressure [MPa]"]["max"]
            min_G, max_G = self.feature_ranges["mass_flux [kg/m2-s]"]["min"], self.feature_ranges["mass_flux [kg/m2-s]"]["max"]
            min_x, max_x = self.feature_ranges["x_e_out [-]"]["min"], self.feature_ranges["x_e_out [-]"]["max"]
            
            if x.dim() == 2:
                x = x.squeeze()
            device = x.device
            x = x.detach().cpu().numpy()

            pressure = torch.tensor((x[0] + 1) / 2 * (max_P - min_P) + min_P, device=device)
            mass_flux = torch.tensor((x[1] + 1) / 2 * (max_G - min_G) + min_G, device=device)
            quality = torch.tensor((x[2] + 1) / 2 * (max_x - min_x) + min_x, device=device)

            inputs = {
                "pressure": pressure,
                "mass_flux": mass_flux,
                "quality": quality
            }
            empirical_output = self.physical(**inputs)

        elif self.physical_model == "liu":
            empirical_output = self.physical(x)

        else:
            raise ValueError(f"Unknown physical model: {self.physical_model}")
        
        x = torch.tensor(x, device=device)
        return empirical_output + self.ffn(x)

class RandomForest:

    def __init__(self, algos):
        self.models = [CatBoostRegressor(**algo) for algo in algos]

    def fit(self, X, y):
        for model in self.models:
            model.fit(X, y)

    def predict(self, X):
        return [model.predict(X) for model in self.models]