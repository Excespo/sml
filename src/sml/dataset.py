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

    def __init__(self, data_path, only_physical_features=False):
        self.data_path = Path(data_path)
        self.df = pd.read_csv(self.data_path)
        self.only_physical_features = only_physical_features
        
        self.y = self.df.loc[:, "chf_exp [MW/m2]"]
        self.df = self.df.drop(columns=["chf_exp [MW/m2]"])
        # self.idx_to_column = {i: col for i, col in enumerate(self.df.columns)}
        # self.column_to_idx = {col: i for i, col in enumerate(self.df.columns)}

        self._transform_text_label()
        self._drop_useless_columns(self.only_physical_features)
        self._impute()
        self._normalize()
        self._df_to_tensor()

    def _transform_text_label(self):
        if not self.only_physical_features:
            le = LabelEncoder()
            if "author" in self.df.columns:
                self.df["author"] = le.fit_transform(self.df["author"])
            if "geometry" in self.df.columns:
                self.df["geometry"] = le.fit_transform(self.df["geometry"])

    def _impute(self):
        if not self.df.isnull().values.any():
            return
        # impute other missing values
        imputer = KNNImputer(n_neighbors=4)
        imputed_data = imputer.fit_transform(self.df)
        self.df = pd.DataFrame(imputed_data, columns=self.df.columns)

    def _drop_useless_columns(self, only_physical_features):
        self.df.drop(columns=["id"], inplace=True)
        if only_physical_features:
            self.df.drop(columns=["author", "geometry"], inplace=True)

    def normalize(self):
        return self._normalize()
    
    def denormalize(self):
        pass
    
    def _normalize(self):
        self.feature_ranges = {}
        for column in self.df.columns:
            if self.df[column].dtype in [np.float64, np.int64]:
                print(f"Normalizing {column} with min: {self.df[column].min()} and max: {self.df[column].max()}, sample 10: {self.df[column].sample(10)}")
                col_min = self.df[column].min()
                col_max = self.df[column].max()
                self.feature_ranges[column] = {
                    'min': col_min,
                    'max': col_max
                }
                if col_max > col_min: # avoid division by zero
                    self.df[column] = 2 * (self.df[column] - col_min) / (col_max - col_min) - 1


    def _df_to_tensor(self):
        self.data = torch.tensor(self.df.to_numpy(), dtype=torch.float32)
        assert not torch.isnan(self.data).any(), "Abnormal data containing NaN values after imputation"

    def __getitem__(self, index):
        return self.data[index], self.y[index]
    
    def __len__(self):
        return len(self.y)


def build_train_and_test_dataset(data_paths, from_scratch=False, cache_dir="./data/cache", train_ratio=0.8, only_physical_features=False):

    cache_dir = Path(cache_dir)
    dataset_suffix = ("_physical_features" if only_physical_features else "") + f"_{round(train_ratio, 2)}@{round(1-train_ratio, 2)}"
    train_cache = cache_dir / f"train_dataset{dataset_suffix}.pt"
    test_cache = cache_dir / f"test_dataset{dataset_suffix}.pt"
    
    if train_cache.exists() and test_cache.exists() and not from_scratch:
        logger.info(f"Loading cached train and test dataset... from cache: {train_cache.absolute()} and {test_cache.absolute()}")
        train_dataset = torch.load(train_cache)
        test_dataset = torch.load(test_cache)
        logger.info(f"Loaded {len(train_dataset)} train samples and {len(test_dataset)} test samples")
        return train_dataset, test_dataset

    logger.info(f"Building from scratch train and test dataset from data_paths: {data_paths}")
    train_cache.unlink(missing_ok=True)
    test_cache.unlink(missing_ok=True)
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