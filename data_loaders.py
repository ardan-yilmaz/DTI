

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
    if 'prot_fam' in df.columns:
        labels = torch.tensor(df['prot_fam'].values, dtype=torch.long)
    
    else: # for transfer learning
        labels = torch.tensor(df['label'].values, dtype=torch.long)

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
    Params:
        df: the dataframe containing the data
        transfer_fam_idx: index of the transfer protein family
        N: number of samples to take from each non-transfer class

    The transfered model will be a binary classifier. 
    The transfer class will contain data from non-transfer classes, which will serve as negative samples.


    """
    def __init__(self, df, transfer_fam_idx=5, N=500, batch_size=32) -> None:
        self.df = df
        self.transfer_fam_idx = transfer_fam_idx
        self.N = N
        self.batch_size = batch_size

        self.transfer_df = None
        self.non_transfer_df = None
        self.create_split()



    def create_split(self):
        """
        Take N/5 samples from each non-transfer class, forming negative data, 
        and add N transfer data (positive data) to create a new dataframe.
        Create transfer and non-transfer dfs.
        """

        transfer_df = self.df[self.df['prot_fam'] == self.transfer_fam_idx].reset_index(drop=True)
        # change the prot_fam attribute to "label" and set the value to 1
        transfer_df['label'] = 1
        transfer_df = transfer_df.drop(columns=['prot_fam'])

        non_transfer_df = self.df[self.df['prot_fam'] != self.transfer_fam_idx].reset_index(drop=True)

        # take N random samples from each non-transfer class
        for pf in range(6):  # assuming there are always 6 protein families
            if pf != self.transfer_fam_idx:
                sample = self.df[self.df['prot_fam'] == pf].sample(n=self.N, random_state=1).reset_index(drop=True)
                
                # remove these samples from non_transfer_df
                non_transfer_df = non_transfer_df.loc[~non_transfer_df.index.isin(sample.index)]
                
                # add these samples to transfer_df with label = 0
                sample['label'] = 0
                sample = sample.drop(columns=['prot_fam'])
                transfer_df = pd.concat([transfer_df, sample], ignore_index=True)

        self.transfer_df = transfer_df
        self.non_transfer_df = non_transfer_df


    def get_non_transfer_dataloaders(self):
        # split the non_transfer items to train/validation/test: 80/10/10
        len_non_transfer_dataset = len(self.non_transfer_df)
        len_non_transfer_train_set = int(0.8 * len_non_transfer_dataset)
        len_non_transfer_val_set = int(0.1 * len_non_transfer_dataset)
        len_non_transfer_test_set = len_non_transfer_dataset - len_non_transfer_train_set - len_non_transfer_val_set

        # create random splits of these lengths
        train_subset, val_subset, test_subset = random_split(self.non_transfer_df, [len_non_transfer_train_set, len_non_transfer_val_set, len_non_transfer_test_set])

        # Now normalize each
        train_set = normalize(self.non_transfer_df.iloc[train_subset.indices])
        val_set = normalize(self.non_transfer_df.iloc[val_subset.indices])
        test_set = normalize(self.non_transfer_df.iloc[test_subset.indices])

        # create dataloaders
        train_loader = DataLoader(CustomDataset(train_set), batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(CustomDataset(val_set), batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(CustomDataset(test_set), batch_size=self.batch_size, shuffle=False)

        return train_loader, val_loader, test_loader

    

    def get_transfer_dataloaders(self):
        # split the transfer items to train/validation/test: 80/10/10
        len_transfer_dataset = len(self.transfer_df)
        len_transfer_train_set = int(0.8 * len_transfer_dataset)
        len_transfer_val_set = int(0.1 * len_transfer_dataset)
        len_transfer_test_set = len_transfer_dataset - len_transfer_train_set - len_transfer_val_set

        # create random splits of these lengths
        train_subset, val_subset, test_subset = random_split(self.transfer_df, [len_transfer_train_set, len_transfer_val_set, len_transfer_test_set])

        # Now normalize each
        train_set = normalize(self.transfer_df.iloc[train_subset.indices])
        val_set = normalize(self.transfer_df.iloc[val_subset.indices])
        test_set = normalize(self.transfer_df.iloc[test_subset.indices])

        # create dataloaders
        train_loader = DataLoader(CustomDataset(train_set), batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(CustomDataset(val_set), batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(CustomDataset(test_set), batch_size=self.batch_size, shuffle=False)

        return train_loader, val_loader, test_loader
    





class AdvancedLoader:
    """
    Init Params:
        df: dataframe containing the data
        transfer_fam_idx: index of the transfer protein family
        transfer_train_size: size of the transfer training set (//5 = size of each non-transfer training set)

    Use the rest of the transfer set for testing. 
    """
    def __init__(self, df, transfer_fam_idx, transfer_train_size=None, batch_size=32):
        self.df = df
        self.transfer_fam_idx = transfer_fam_idx
        self.batch_size = batch_size

        if transfer_train_size is None:
            self.transfer_train_size = 500 # default: take 500 samples from the transfer set for finetuning
            self.neg_sample_size = 100 # default: take 100 samples from each non-transfer class for finetuning
        else: 
            self.transfer_train_size = transfer_train_size
            self.neg_sample_size = self.get_neg_sample_size()   

        # self.transfer_test_neg_sampling_size is the number of negative samples to be taken from each non-transfer class for testing in a balanced manner
        self.transfer_test_neg_sampling_size = int((len(self.df[self.df['prot_fam'] == self.transfer_fam_idx]) - self.transfer_train_size)//5)


        self.non_transfer_df = None
        self.transfer_train_df = None
        self.transfer_test_df = None
        self.create_split()

        

    def create_split(self):
        """
        Creates non_transfer_df, transfer_train_df, and transfer_test_df.
            
        Building transfer_df:
        - Updates transfer_df labels as all positive (for transfer instances)
        - Takes random samples from non-transfer classes adding them to transfer_df as negative samples.
        - Drops those items (added as negative instances) from the transfer_df
        
        Building non_transfer_df:
        - Removes the transfer_df items from the original df
        """

        # 1. CREATE NON-TRANSFER DF
        self.non_transfer_df = self.df[self.df['prot_fam'] != self.transfer_fam_idx].reset_index(drop=True)        

        # 2. GET TRANSFER_TRAIN_DF AND TRANSFER_TEST_DF
        # create transfer_df with transfer items initially
        transfer_df = self.df[self.df['prot_fam'] == self.transfer_fam_idx].reset_index(drop=True)
        # sample self.transfer_train_size items from transfer_df
        self.transfer_train_df = transfer_df.sample(n=self.transfer_train_size, random_state=1).reset_index(drop=True)
        # put the rest of the items into transfer_test_df
        self.transfer_test_df = transfer_df.loc[~transfer_df.index.isin(self.transfer_train_df.index)].reset_index(drop=True)
        # update the prot_fam field of transfer_train_df & transfer_test_df to label and set the value to 1
        self.transfer_train_df['label'] = 1
        self.transfer_test_df['label'] = 1
        self.transfer_train_df = self.transfer_train_df.drop(columns=['prot_fam'])
        self.transfer_test_df = self.transfer_test_df.drop(columns=['prot_fam'])

        # 3. UPDATE TRANSFER DF AND NON-TRANSFER DF
        # take N random samples from each non-transfer class
        for pf in range(6):
            if pf != self.transfer_fam_idx:
                # get df of class:
                class_df = self.df[self.df['prot_fam'] == pf].reset_index(drop=True)
                # take negative sample from this class
                neg_sample = class_df.sample(n=self.neg_sample_size, random_state=1).reset_index(drop=True)
                # add these samples to transfer_df with label = 0
                neg_sample['label'] = 0
                neg_sample = neg_sample.drop(columns=['prot_fam'])
                self.transfer_train_df = pd.concat([self.transfer_train_df, neg_sample], ignore_index=True)
                # drop these negative samples from class_df
                class_df = class_df.loc[~class_df.index.isin(neg_sample.index)]

                # take negative test samples from this class
                neg_test_sample = class_df.sample(n=self.transfer_test_neg_sampling_size, random_state=1).reset_index(drop=True)
                # add these samples to transfer_test_df with label = 0
                neg_test_sample['label'] = 0
                neg_test_sample = neg_test_sample.drop(columns=['prot_fam'])
                self.transfer_test_df = pd.concat([self.transfer_test_df, neg_test_sample], ignore_index=True)

                # drop these negative samples from non_transfer_df
                self.non_transfer_df = self.non_transfer_df.loc[~self.non_transfer_df.index.isin(neg_sample.index)]
                # drop these negative test samples from non_transfer_df
                self.non_transfer_df = self.non_transfer_df.loc[~self.non_transfer_df.index.isin(neg_test_sample.index)]

        


    def get_non_transfer_dataloaders(self):
        # split the non_transfer items to train/validation/test: 80/10/10
        len_non_transfer_dataset = len(self.non_transfer_df)
        len_non_transfer_train_set = int(0.8 * len_non_transfer_dataset)
        len_non_transfer_val_set = int(0.1 * len_non_transfer_dataset)
        len_non_transfer_test_set = len_non_transfer_dataset - len_non_transfer_train_set - len_non_transfer_val_set

        # create random splits of these lengths
        train_subset, val_subset, test_subset = random_split(self.non_transfer_df, [len_non_transfer_train_set, len_non_transfer_val_set, len_non_transfer_test_set])

        # Now normalize each
        train_set = normalize(self.non_transfer_df.iloc[train_subset.indices])
        val_set = normalize(self.non_transfer_df.iloc[val_subset.indices])
        test_set = normalize(self.non_transfer_df.iloc[test_subset.indices])

        # create dataloaders
        train_loader = DataLoader(CustomDataset(train_set), batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(CustomDataset(val_set), batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(CustomDataset(test_set), batch_size=self.batch_size, shuffle=False)

        return train_loader, val_loader, test_loader

    

    def get_transfer_dataloaders(self):
        # split self.transfer_train_df into train/val
        len_transfer_train_dataset = len(self.transfer_train_df)
        len_transfer_train_train_set = int(0.8 * len_transfer_train_dataset)
        len_transfer_train_val_set = len_transfer_train_dataset - len_transfer_train_train_set

        # create random splits of these lengths
        train_subset, val_subset = random_split(self.transfer_train_df, [len_transfer_train_train_set, len_transfer_train_val_set])

        # Now normalize each
        train_set = normalize(self.transfer_train_df.iloc[train_subset.indices])
        val_set = normalize(self.transfer_train_df.iloc[val_subset.indices])
        test_set = normalize(self.transfer_test_df)

        # create dataloaders
        train_loader = DataLoader(CustomDataset(train_set), batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(CustomDataset(val_set), batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(CustomDataset(test_set), batch_size=self.batch_size, shuffle=False)

        return train_loader, val_loader, test_loader


    









    def get_neg_sample_size(self):
        """
        Calculates the number of negative samples from each trasfer class for training.

        if transfer train size, ie, the number of positive data points to be used for finetuning is greater than 5: 
            neg_sample_size is 1/5 of it to achieve a negative/positive balance.
        else: 
            neg_sample_size = 1
        """
        
        if self.transfer_train_size >= 5:
            return self.transfer_train_size // 5
        
        return 1
    

            

