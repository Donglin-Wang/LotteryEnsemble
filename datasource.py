import numpy as np
import random
import torch
import torchvision as tv
import torchvision.transforms as tf
from sklearn.utils import shuffle

# Given each user euqal number of samples if possible. If not, the last user 
# gets whatever is left after other users had their shares

def iid_split(num_clients, 
              train_data, 
              batch_size, test_data):
    
    all_train_idx = np.arange(train_data.data.shape[0])
    
    sample_train_idx = np.array_split(all_train_idx, num_clients)

    all_test_idx = np.arange(test_data.data.shape[0])

    sample_test_idx = np.array_split(all_test_idx, num_clients)
    
    user_train_loaders = []
    user_test_loaders = []
    
    for idx in sample_train_idx:
        user_train_loaders.append(torch.utils.data.DataLoader(train_data, 
                                            sampler=idx,
                                            batch_size=batch_size))
    for idx in sample_test_idx:
        user_test_loaders.append(torch.utils.data.DataLoader(test_data,
                                                              sampler=idx,
                                                              batch_size=batch_size))
    
    return user_train_loaders, user_test_loaders

# Sort the labels before splitting the data to each user

def non_iid_split(num_clients,
                  train_data, 
                  batch_size, test_data):
    
    data_size = train_data.data.shape[0]

    #Test data size
    data_size_test = test_data.data.shape[0]
    
    # Making sure that each client gets at least 2 samples
    assert data_size > num_clients * 2
    
    # Sorting the data by their class labels
    label_idx_pairs = [ (i, train_data.targets[i]) for i in range(data_size)]
    label_idx_pairs = sorted(label_idx_pairs, key=lambda pair : pair[1])
    sorted_idx = [idx for idx, label in label_idx_pairs]


    # Sorting the data by their class labels TEST DATA
    label_idx_pairs_test = [ (i, test_data.targets[i]) for i in range(data_size_test)]
    label_idx_pairs_test = sorted(label_idx_pairs_test, key=lambda pair : pair[1])
    sorted_idx_test = [idx for idx, label in label_idx_pairs_test]

    #Random int for setting seed to make the Train data and Test data have the same labels
    rand_num = random.randint(1, 100000)
    
    # Split the class labels into 2 * num_clients chunks. If the data cannot
    # be equally divided, the last chunk will have less data than the rest.
    sample_bin_idx = np.array_split(sorted_idx, num_clients * 2)
    np.random.seed(rand_num)
    sample_bin_idx = np.random.permutation(sample_bin_idx)
    num_bins = len(sample_bin_idx)

    #For test data
    sample_bin_idx_test = np.array_split(sorted_idx_test, num_clients * 2)
    np.random.seed(rand_num)
    sample_bin_idx_test = np.random.permutation(sample_bin_idx_test)           

    #Training data loaders
    user_loaders = []
    #Test data loaders
    test_loaders = []

    
    for i in range(0, num_bins, 2):
        
        client_data_idx = sample_bin_idx[i]

        client_test_data_idx = sample_bin_idx_test[i]
        
        if i + 1 < num_bins:
            client_data_idx = np.append(client_data_idx, sample_bin_idx[i+1])
            client_test_data_idx = np.append(client_test_data_idx, sample_bin_idx_test[i+1])
            
        client_data_idx = shuffle(client_data_idx)
        client_test_data_idx = shuffle(client_test_data_idx)
            
        #Trainning data
        randomize_train_Sample = np.random.permutation(client_data_idx)
        cur_sampler = torch.utils.data.BatchSampler(randomize_train_Sample, 
                                                    batch_size, 
                                                    drop_last=False)
        cur_loader = torch.utils.data.DataLoader(train_data,
                                                 batch_sampler=cur_sampler)
        user_loaders.append(cur_loader)

        #Test data
        randomize_test_Sample = np.random.permutation(client_test_data_idx)
        cur_sampler_test = torch.utils.data.BatchSampler(randomize_test_Sample, 
                                                    batch_size, 
                                                    drop_last=False)
        cur_loader_test = torch.utils.data.DataLoader(test_data,
                                                 batch_sampler=cur_sampler_test)
        test_loaders.append(cur_loader_test)

    return user_loaders, test_loaders

def non_iid_unequal_split(num_clients, 
                          train_data,
                          batch_size,
                          min_size, 
                          max_size):
    
    
    return
    
def get_data(num_clients, dataset_name,
             mode="iid",
             batch_size=4,
             min_shard=1,
             max_shard=30):
    
    train_data, test_data = [], []
    
    transform = tf.Compose(
        [tf.ToTensor(), 
          tf.Normalize((0.5), (0.5))
         ]
    )
    
     # Downloading data based on inputs. If the data is already downloaded,
     # it won't be download twice
     
    if dataset_name == "cifar10":
        train_data = tv.datasets.CIFAR10(root="./data", 
                                 train=True, 
                                 download=True, 
                                 transform=transform)
        test_data = tv.datasets.CIFAR10(root="./data",
                                train=False,
                                download=True,
                                transform=transform)
    elif dataset_name == "mnist":
        train_data = tv.datasets.MNIST(root="./data", 
                                 train=True, 
                                 download=True, 
                                 transform=transform)
        test_data = tv.datasets.MNIST(root="./data",
                                train=False,
                                download=True,
                                transform=transform)

    elif dataset_name == "cifar100":
        train_data = tv.datasets.CIFAR100(root="./data", 
                                 train=True, 
                                 download=True, 
                                 transform=transform)
        test_data = tv.datasets.CIFAR100(root="./data",
                                train=False,
                                download=True,
                                transform=transform)

    else:
        print("You did not enter the name of a supported dataset")
        print("Supported datasets: {}, {}".format('"cifar10"', '"mnist"'))
        exit()
        
    global_test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)
    
    if mode == "iid":
        return iid_split(num_clients, 
                         train_data, 
                         batch_size, test_data), global_test_loader
    
    elif mode == "non-iid":
        return non_iid_split(num_clients, 
                             train_data,  
                             batch_size, test_data), global_test_loader
    
    elif mode == "non-iid-unequal":
        return non_iid_unequal_split(num_clients, 
                                     train_data, 
                                     batch_size,
                                     min_shard, 
                                     max_shard), global_test_loader
    else:
        print("You did not enter a supported data splitting scheme")
        print("Supported data splitting schemes: {}, {}, {}".format('"iid"', '"non-iid"', '"non-iid-unequal"'))
        exit()

    return 0

if __name__ == "__main__":
    
    #print("Load MNIST 10")
    #user_loaders, test_loader = get_data(10, "mnist")
    #assert len(user_loaders) == 10
    
    print("Load MNIST 10 non-iid")
    (users_data, test_loader), global_test_loader = get_data(400, "mnist", mode="non-iid", batch_size=32)
    print(len(users_data))
    print(len(test_loader))
    print(len(global_test_loader))

    count = 0
    print("training data")
    for data, label in users_data[0]:
        print(label)
        count += 1
    print(count)

    count = 0
    print("testing data")
    for data, label in test_loader[0]:
        print(label)
        count += 1
    print(count)

    print(len(users_data[0]))
    print(len(global_test_loader))

    
