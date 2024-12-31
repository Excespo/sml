from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder

import torch
from torch.utils.data import Dataset, ConcatDataset

from sml.utils import get_logger

logger = get_logger(__name__)

class CHFDataset(Dataset):

    def __init__(self, data_path, only_physical_features=False, drop_first_n=3):
        self.data_path = Path(data_path)
        self.data = pd.read_csv(self.data_path)
        self.only_physical_features = only_physical_features
        self.drop_first_n = drop_first_n
        
        self.y = self.data.loc[:, "chf_exp [MW/m2]"]
        self.data = self.data.drop(columns=["chf_exp [MW/m2]"])

        self._transform_text_label()
        self._drop_useless_columns(self.only_physical_features)
        self._impute()
        self._normalize()

        if type(self.data) != np.ndarray:
            self.data = self.data.to_numpy()
        self.data = torch.tensor(self.data, dtype=torch.float32)
        assert not torch.isnan(self.data).any(), "Abnormal data containing NaN values after imputation"

    def _transform_text_label(self):
        if not self.only_physical_features:
            le = LabelEncoder()
            if "author" in self.data.columns:
                self.data["author"] = le.fit_transform(self.data["author"])
            if "geometry" in self.data.columns:
                self.data["geometry"] = le.fit_transform(self.data["geometry"])

    def _impute(self):
        if not self.data.isnull().values.any():
            return
        # impute other missing values
        imputer = KNNImputer(n_neighbors=4)
        imputer.fit(self.data)
        self.data = imputer.transform(self.data)

    def _drop_useless_columns(self, only_physical_features):
        self.data.drop(columns=["id"], inplace=True)
        if only_physical_features:
            self.data.drop(columns=["author", "geometry"], inplace=True)

    def _normalize(self):
        if type(self.data) == np.ndarray:
            data_min = np.min(self.data, axis=0)
            data_max = np.max(self.data, axis=0)
            self.data = 2 * (self.data - data_min) / (data_max - data_min) - 1
        else:
            for column in self.data.columns:
                if self.data[column].dtype in [np.float64, np.int64]:
                    col_min = self.data[column].min()
                    col_max = self.data[column].max()
                    if col_max > col_min:  # 避免除以零
                        self.data[column] = 2 * (self.data[column] - col_min) / (col_max - col_min) - 1
        
        # 保存归一化参数，以便后续使用
        self.feature_ranges = {
            'min': data_min if type(self.data) == np.ndarray else self.data.min(),
            'max': data_max if type(self.data) == np.ndarray else self.data.max()
        }

    def __getitem__(self, index):
        return self.data[index], self.y[index]
    
    def __len__(self):
        return len(self.y)


def build_train_and_test_dataset(data_paths, cache_dir="./data/cache", train_ratio=0.8, only_physical_features=False):

    cache_dir = Path(cache_dir)
    dataset_suffix = ("_physical_features" if only_physical_features else "") + f"_{round(train_ratio, 2)}@{round(1-train_ratio, 2)}"
    train_cache = cache_dir / f"train_dataset{dataset_suffix}.pt"
    test_cache = cache_dir / f"test_dataset{dataset_suffix}.pt"
    
    if train_cache.exists() and test_cache.exists():
        logger.info(f"Loading cached train and test dataset... from cache: {train_cache.absolute()} and {test_cache.absolute()}")
        train_dataset = torch.load(train_cache)
        test_dataset = torch.load(test_cache)
        logger.info(f"Loaded {len(train_dataset)} train samples and {len(test_dataset)} test samples")
        return train_dataset, test_dataset

    logger.info(f"Building from scratch train and test dataset from data_paths: {data_paths}")
    # print(data_paths)
    # if type(data_paths) == str:
    #     data_paths = data_paths.split(",")
    dataset = ConcatDataset([CHFDataset(path, only_physical_features=only_physical_features) for path in data_paths])
    
    dataset_size = len(dataset)
    train_size = int(train_ratio * dataset_size)
    test_size = dataset_size - train_size
    
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, 
        [train_size, test_size]
    )
    
    logger.info(f"Saved {len(train_dataset)} train samples and {len(test_dataset)} test samples, to {cache_dir.absolute()}")
    cache_dir.mkdir(exist_ok=True, parents=True)
    torch.save(train_dataset, train_cache)
    torch.save(test_dataset, test_cache)
    
    return train_dataset, test_dataset