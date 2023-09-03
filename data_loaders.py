

from typing import Any
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset, ConcatDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def normalize(df):
    # Normalize the data
    scaler = StandardScaler()
    scaler.fit(df.iloc[:, 2:].values)
    
    features = torch.tensor(scaler.transform(df.iloc[:, 2:].values), dtype=torch.float32)
    labels = torch.tensor(df['prot_fam'].values, dtype=torch.long)

    return features, labels




def split_into_train_val_test(df, train_size=0.8, val_size=0.1):
    # calculate the lens of each set
    len_dataset = len(df)
    len_train_set = int(train_size * len_dataset)  # 80% for training
    len_val_set = int(val_size * len_dataset)  # 10% for validation
    len_test_set = len_dataset - len_train_set - len_val_set  # 10% for testing

    # random split
    train_subset, val_subset, test_subset = random_split(df, [len_train_set, len_val_set, len_test_set])   
    
    train_df = df.iloc[train_subset.indices]
    val_df = df.iloc[val_subset.indices]
    test_df = df.iloc[test_subset.indices]    

    return train_df, val_df, test_df

def split_into_train_val_test_int(df, train_size):
    # split the df into train/test: n  /  len(df) - n
    train_subset, test_subset = random_split(df, [train_size, len(df) - train_size])

    train_df = df.iloc[train_subset.indices]
    test_df = df.iloc[test_subset.indices]

    return train_df, test_df



class CustomDataset(Dataset):
    def __init__(self, data_tuple):
        self.data, self.labels = data_tuple

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    

class NaiveDataLoader:
    """
    This class is used to create the naive dataloaders for the baseline models to compare the performance and
    convergence. It uses all the data for training, validation and testing. (No transfer learning here!)
    """    
    def __init__(self, df, batch_size=32):
        self.df = df
        self.batch_size = batch_size
        self.train_set, self.val_set, self.test_set = self.create_split()

    def create_split(self):
        # split the data into train/test/val : 80/10/10
        train_df, val_df, test_df = split_into_train_val_test(self.df)

        # Now normalize each
        train_set = normalize(train_df)
        val_set = normalize(val_df)
        test_set = normalize(test_df) 

        return train_set, val_set, test_set


    def create_dataloaders(self):
        train_features, train_labels = self.to_tensor(self.train_set)
        val_features, val_labels = self.to_tensor(self.val_set)
        test_features, test_labels = self.to_tensor(self.test_set)
        
        train_dataset = TensorDataset(train_features, train_labels)
        val_dataset = TensorDataset(val_features, val_labels)
        test_dataset = TensorDataset(test_features, test_labels)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        return train_loader, val_loader, test_loader
    
    def to_tensor(self, data):
        # Convert numpy arrays to PyTorch tensors
        features, labels = data
        return torch.tensor(features, dtype=torch.float32), torch.tensor(labels, dtype=torch.int64)    
    


class TransferDataLoader:
    """
    This class is used to create the dataloaders for the transfer learning models.
    Initialized with the transfer class, it creates the non-transfer class and splits it into train/val/test.
    N: # of transfer samples to use for training. If not set, ie None, it uses all the transfer samples.

    To get the dataloaders, after init, call get_nontransfer_dataloaders() and get_transfer_dataloaders()
    """

    def __init__(self, df, transfer_prot_fam = 5, batch_size=32, N= None):
        self.df = df
        self.transfer_prot_fam = transfer_prot_fam
        self.batch_size = batch_size
        self.N = N

        # These to be filled by create_nontrasfer_split()
        self.non_transfer_train_set = None
        self.non_transfer_val_set = None
        self.non_transfer_test_set = None
        self.transfer_train_set = None
        self.transfer_val_set = None
        self.transfer_test_set = None

        self.create_nontrasfer_split()


    def create_nontrasfer_split(self):
        # get the non-transfer class
        non_transfer_df = self.df[self.df['prot_fam'] != self.transfer_prot_fam].reset_index(drop=True)
        transfer_df = self.df[self.df['prot_fam'] == self.transfer_prot_fam].reset_index(drop=True)

        # split the non-transfer items to train/test/val : 80/10/10
        non_transfer_train_df, non_transfer_val_df, non_transfer_test_df = split_into_train_val_test(non_transfer_df)

        # normalize
        self.non_transfer_train_set = CustomDataset(normalize(non_transfer_train_df))
        self.non_transfer_val_set = CustomDataset(normalize(non_transfer_val_df))
        self.non_transfer_test_set = CustomDataset(normalize(non_transfer_test_df))
        
 
        # split the transfer class into train/val/test.     if N not given
        if self.N == None:
            transfer_train_df, transfer_val_df, transfer_test_df = split_into_train_val_test(transfer_df)
            self.transfer_train_set = CustomDataset(normalize(transfer_train_df))
            self.transfer_val_set = CustomDataset(normalize(transfer_val_df))
            transfer_test_set = CustomDataset(normalize(transfer_test_df))

            # concat transfer_test_set with self.non_transfer_test_set 
            self.transfer_test_set = ConcatDataset([transfer_test_set, self.non_transfer_test_set])


        # split the transfer class into train/test: n  /  len(transfer_df) -  n.  IF N GIVENN
        else:
            transfer_train_df, transfer_test_df = train_test_split(transfer_df, train_size=self.N)
            self.transfer_train_set = CustomDataset(normalize(transfer_train_df))
            transfer_test_set = CustomDataset(normalize(transfer_test_df))

            # concat transfer_test_set with self.non_transfer_test_set 
            self.transfer_test_set = ConcatDataset([transfer_test_set, self.non_transfer_test_set])

    
    def get_nontransfer_dataloaders(self):

        # Create dataloaders
        non_transfer_train_loader = DataLoader(self.non_transfer_train_set, batch_size=self.batch_size, shuffle=True)
        non_transfer_val_loader = DataLoader(self.non_transfer_val_set, batch_size=self.batch_size, shuffle=False)
        non_transfer_test_loader = DataLoader(self.non_transfer_test_set, batch_size=self.batch_size, shuffle=False)

        # create a dict of dataloaders
        non_transfer_data_loaders = {
            'train': non_transfer_train_loader,
            'test': non_transfer_test_loader,
            'val': non_transfer_val_loader
        }

        return non_transfer_data_loaders

    def get_transfer_dataloaders(self):

        # Create dataloaders
        transfer_train_loader = DataLoader(self.transfer_train_set, batch_size=self.batch_size, shuffle=True)
        transfer_test_loader = DataLoader(self.transfer_test_set, batch_size=self.batch_size, shuffle=False)

        # create a dict of dataloaders
        transfer_data_loaders = {
            'train': transfer_train_loader,
            'test': transfer_test_loader,
        }

        return transfer_data_loaders
    

 





## THIS HAS A 6-CLASS AFTER TRANSFER
"""
class CustomDataLoader:
    def __init__(self, df, transfer_prot_fam = 5, batch_size=32):
        self.df = df
        self.transfer_prot_fam = transfer_prot_fam
        self.batch_size = batch_size
        self.non_transfer_train_set, self.non_transfer_val_set, self.non_transfer_test_set = self.create_trasfer_split()

    def normalize(self, df):
        # Normalize the data
        
        features = self.scaler.transform(df.iloc[:, 2:].values)
        labels = df['prot_fam'].values

        return features, labels

    def create_trasfer_split(self):
        # get the non-transfer class
        non_transfer_df = self.df[self.df['prot_fam'] != self.transfer_prot_fam].reset_index(drop=True)

        # split the non-transfer items to train/test/val : 80/10/10
        len_non_transfer_dataset = len(non_transfer_df)
        len_non_transfer_train_set = int(0.8 * len_non_transfer_dataset)  # 80% for training
        len_non_transfer_val_set = int(0.1 * len_non_transfer_dataset)  # 10% for validation
        len_non_transfer_test_set = len_non_transfer_dataset - len_non_transfer_train_set - len_non_transfer_val_set  # 10% for testing

        # Compute the normalization parameters on the training data
        scaler = StandardScaler()
        scaler.fit(non_transfer_df.iloc[:len_non_transfer_train_set, 2:].values)
        
        # Normalize the non-transfer data and create the CustomDatasets
        
        features = scaler.transform(non_transfer_df.iloc[:, 2:].values)
        labels = non_transfer_df['prot_fam'].values

        non_transfer_train_set, non_transfer_val_set, non_transfer_test_set = random_split(
            CustomDataset(features, labels), 
            [len_non_transfer_train_set, len_non_transfer_val_set, len_non_transfer_test_set]) 

        return non_transfer_train_set, non_transfer_val_set, non_transfer_test_set

    def get_nontransfer_dataloaders(self):
        # create dataloaders
        non_transfer_train_loader = DataLoader(self.non_transfer_train_set, batch_size=self.batch_size, shuffle=True)
        non_transfer_val_loader = DataLoader(self.non_transfer_val_set, batch_size=self.batch_size, shuffle=False)
        non_transfer_test_loader = DataLoader(self.non_transfer_test_set, batch_size=self.batch_size, shuffle=False)

        return non_transfer_train_loader, non_transfer_val_loader, non_transfer_test_loader

    def get_transfer_dataloaders(self, n=1):
        # get the transfer class
        transfer_df = self.df[self.df['prot_fam'] == self.transfer_prot_fam].reset_index(drop=True)

        # Normalize the transfer data
        features = self.scaler.transform(transfer_df.iloc[:, 2:].values)
        labels = transfer_df['prot_fam'].values

        # split the transfer class into train/test: n  /  len(transfer_df) - n
        transfer_train_set, transfer_test_set = random_split(
            CustomDataset(features, labels), [n, len(transfer_df) - n])

        # concat transfer_test_set with self.non_transfer_test_set 
        transfer_test_set = torch.utils.data.ConcatDataset([transfer_test_set, self.non_transfer_test_set])

        # create the dataloaders for the transfer model
        transfer_train_loader = DataLoader(transfer_train_set, batch_size=n, shuffle=True)
        transfer_test_loader = DataLoader(transfer_test_set, batch_size=n, shuffle=False)

        return transfer_train_loader, transfer_test_loader

""" 











