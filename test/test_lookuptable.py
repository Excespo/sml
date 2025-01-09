from sml.model import LookUpTable
from sml.dataset import build_train_and_test_dataset

import numpy as np
from torch.utils.data import Subset

model = LookUpTable(
    lut_path="thirdparty/2006_Groeneveld_CriticalHeatFlux_LUT/2006LUT.sdf"
)

train_dataset, test_dataset = build_train_and_test_dataset(
    data_paths=["data/Data_CHF_Zhao_2020_ATE.csv"],
    from_scratch=True,
    train_ratio=0.8,
    only_physical_features=True
)

print(len(train_dataset))
print(len(test_dataset))

# print(train_dataset[0])
# print(test_dataset[0])

all_abs_errors = []
all_mse_errors = []
all_rmse_errors = []
thresholds = [0.01, 0.1, 0.3, 0.5]
all_cumulative_fractions = {}

print(train_dataset)
print(train_dataset.dataset)
print(single_ds:=train_dataset.dataset.datasets[0])
print(fr:=single_ds.feature_ranges)


min_P, max_P = fr["pressure [MPa]"]["min"], fr["pressure [MPa]"]["max"]
min_G, max_G = fr["mass_flux [kg/m2-s]"]["min"], fr["mass_flux [kg/m2-s]"]["max"]
min_x, max_x = fr["x_e_out [-]"]["min"], fr["x_e_out [-]"]["max"]

for (features, chf_ref) in train_dataset:

    # print(f"features[0]: {features[0]}, min_P: {min_P}, max_P: {max_P}, denormalized: {(features[0] + 1) / 2 * (max_P - min_P) + min_P}")
    # print(f"features[1]: {features[1]}, min_G: {min_G}, max_G: {max_G}, denormalized: {(features[1] + 1) / 2 * (max_G - min_G) + min_G}")
    # print(f"features[2]: {features[2]}, min_x: {min_x}, max_x: {max_x}, denormalized: {(features[2] + 1) / 2 * (max_x - min_x) + min_x}")

    P = (features[0] + 1) / 2 * (max_P - min_P) + min_P
    G = (features[1] + 1) / 2 * (max_G - min_G) + min_G
    x = (features[2] + 1) / 2 * (max_x - min_x) + min_x

    inputs = {"mass_flux": G, "quality": x, "pressure": P}
    
    chf_pred = model(**inputs)
    
    abs_error = abs(chf_pred - chf_ref)
    mse_error = (chf_pred - chf_ref) ** 2
    rmse_error = np.sqrt(mse_error)

    print(f"P: {P}, G: {G}, x: {x}, chf_pred: {chf_pred}, chf_ref: {chf_ref}")
    
    all_abs_errors.append(abs_error)
    all_mse_errors.append(mse_error)
    all_rmse_errors.append(rmse_error)
    
all_abs_errors = sorted(all_abs_errors)
for threshold in thresholds:
    cumulative_fraction = np.sum(np.array(all_abs_errors) <= threshold) / len(all_abs_errors)
    all_cumulative_fractions[threshold] = cumulative_fraction

print(f"mean abs error: {np.mean(all_abs_errors)}")
print(f"mean mse error: {np.mean(all_mse_errors)}")
print(f"mean rmse error: {np.mean(all_rmse_errors)}")
for threshold in thresholds:
    print(f"cumulative fraction for threshold {threshold}: {round(100*all_cumulative_fractions[threshold], 2)}%")
