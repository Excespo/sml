import pandas as pd

from torch.utils.data import ConcatDataset

from sml.dataset import CHFDataset, build_train_and_test_dataset
from sml.utils import set_seed, get_logger

logger = get_logger(__name__)

set_seed(42)

dataset1 = CHFDataset("data/Data_CHF_Zhao_2020_ATE.csv")
# dataset2 = CHFDataset("data/playground-series-s3e15/data.csv")

# example_X, example_y = dataset1[0]
# print(example_X)
# print(example_y)
# feature_ranges = dataset1.feature_ranges

# min_P, max_P = feature_ranges["pressure [MPa]"]["min"], feature_ranges["pressure [MPa]"]["max"]
# min_G, max_G = feature_ranges["mass_flux [kg/m2-s]"]["min"], feature_ranges["mass_flux [kg/m2-s]"]["max"]
# min_x, max_x = feature_ranges["x_e_out [-]"]["min"], feature_ranges["x_e_out [-]"]["max"]

# print(f"features[0]: {example_X[0]}, min_P: {min_P}, max_P: {max_P}, denormalized: {(example_X[0]+1)/2 * (max_P - min_P) + min_P}")
# print(f"features[1]: {example_X[1]}, min_G: {min_G}, max_G: {max_G}, denormalized: {(example_X[1]+1)/2 * (max_G - min_G) + min_G}")
# print(f"features[2]: {example_X[2]}, min_x: {min_x}, max_x: {max_x}, denormalized: {(example_X[2]+1)/2 * (max_x - min_x) + min_x}")
# exit()

# dataset = ConcatDataset([dataset1, dataset2])

# print(dataset[0])

train_dataset, test_dataset = build_train_and_test_dataset(
    data_paths=[
        "/mnt/sml/data/Data_CHF_Zhao_2020_ATE.csv", 
        "/mnt/sml/data/playground-series-s3e15/data.csv"
    ],
    only_physical_features=True
)
print(train_dataset[0])