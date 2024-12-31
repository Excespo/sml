import pandas as pd

from torch.utils.data import ConcatDataset

from sml.dataset import CHFDataset, build_train_and_test_dataset
from sml.utils import set_seed, get_logger

logger = get_logger(__name__)

set_seed(42)

# dataset1 = CHFDataset("data/Data_CHF_Zhao_2020_ATE.csv")
# dataset2 = CHFDataset("data/playground-series-s3e15/data.csv")

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