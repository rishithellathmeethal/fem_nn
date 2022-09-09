import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class loader_creation():
    """ 
    data_in, k_all, f_all in numpy array
    """
    def __init__(self, data_in, k_all, f_all, length):

        if not torch.is_tensor(data_in):
            data_in = torch.tensor(data_in, dtype=torch.float32).to(device)
        if not torch.is_tensor(k_all):
            k_all = torch.tensor(k_all, dtype=torch.float32).to(device)
        if not torch.is_tensor(f_all):
            f_all = torch.tensor(f_all, dtype=torch.float32).to(device)
        
        if int(0.8*length)<1:
            split_index = 1
        else:
            split_index = int(0.8*length)

        # splitting to training and testing 
        train_in = data_in[0:split_index]
        test_in  = data_in[split_index:]
        k_train  = k_all[0:split_index]
        f_train  = f_all[0:split_index]
        k_test   = k_all[split_index:]
        f_test   = f_all[split_index:]
        
        #passing to dataset to make it iterable
        self.ds_train = customdataset_withKandF(train_in, k_train, f_train)
        self.ds_test = customdataset_withKandF(test_in, k_test, f_test)
    
    def get_loaders(self, b_size = 8, shuffle = True):
        if self.ds_train.data_in.shape[0]<5:
            shuffle_b = False
        else:
            shuffle_b = True

        train_loader = DataLoader(self.ds_train, batch_size = b_size, shuffle = shuffle_b, drop_last=True)
        
        if self.ds_test.data_in.shape[0]>0:
            test_loader  = DataLoader(self.ds_test , batch_size = b_size, shuffle = True, drop_last=True)
        else:
            test_loader = None 
        
        return train_loader, test_loader


class customdataset_withKandF(Dataset):
    """
    A class to load a custom data set with kf.
    """

    def __init__(self,train_in, k, f):
        """
        Parameters
        ----------
        train_in : TYPE
            Features - the points to be trained on.
        k : TYPE
            DESCRIPTION.
        f : TYPE
            DESCRIPTION.
        Returns
        -------
        None.
        """
        self.data_in = train_in
        self.k = k
        self.f = f 

    def __getitem__(self, index):
        """
        Parameters
        ----------
        index : TYPE
            Used to retrieve corresponding slice of ---.
        Returns
        -------
        self.data_in[index]
            Corresponding slice of training data features ´self.data_in´.
        self.k[index]
            DESCRIPTION.
        self.f[index]
            DESCRIPTION.
        """
        return self.data_in[index], self.k[index], self.f[index]
     
    def __len__(self):
        """
        Returns
        -------
        TYPE
            Size of the whole training dataset.
        """
        return len(self.data_in)