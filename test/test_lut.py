import pandas as pd

from sml.model import LookUpTable

df = pd.read_csv('/mnt/sml/data/Data_CHF_Zhao_2020_ATE.csv')
pressure = df["pressure [MPa]"].values * 1e6
mass_flux = df["mass_flux [kg/m2-s]"].values
quality = df['x_e_out [-]'].values

model = LookUpTable('thirdparty/2006_Groeneveld_CriticalHeatFlux_LUT/2006LUT.sdf')
q_pred = model(pressure, mass_flux, quality)

q_pred_mw = q_pred / 1e6

print(q_pred_mw)