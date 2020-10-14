import numpy as np
import torch
import torchvision as tv
import torchvision.transforms as tf

# Given each user euqal number of samples if possible. If not, the last user 
# gets whatever is left after other users had their shares

def iid_split(num_clients, 
              train_data, 
              batch_size):
    
    return

# Sort the labels before splitting the data to each user

def non_iid_split(num_clients,
                  train_data, 
                  batch_size):
    
    return
    
def get_data(num_clients, dataset_name,
             mode="iid",
             batch_size=4,
             min_shard=1,
             max_shard=30):
    
    return