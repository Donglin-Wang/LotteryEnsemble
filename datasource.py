import numpy as np
import torch
import torchvision as tv
import torchvision.transforms as tf

# Given each user euqal number of samples if possible. If not, the last user 
# gets whatever is left after other users had their shares

def iid_split(num_clients, 
              train_data, 
              batch_size):
    
    all_idx = np.arange(train_data.data.shape[0])
    
    sample_idx = np.array_split(all_idx, num_clients)
    
    user_train_loaders = []
    
    for idx in sample_idx:
        user_train_loaders.append(torch.utils.data.DataLoader(train_data, 
                                            sampler=idx,
                                            batch_size=batch_size))
    
    return user_train_loaders

# Sort the labels before splitting the data to each user

def non_iid_split(num_clients,
                  train_data, 
                  batch_size):
    
    data_size = train_data.data.shape[0]
    
    # Making sure that each client gets at least 2 samples
    assert data_size > num_clients * 2
    
    # Sorting the data by their class labels
    label_idx_pairs = [ (i, train_data.targets[i]) for i in range(data_size)]
    label_idx_pairs = sorted(label_idx_pairs, key=lambda pair : pair[1])
    sorted_idx = [idx for idx, label in label_idx_pairs]
    
    # Split the class labels into 2 * num_clients chunks. If the data cannot
    # be equally divided, the last chunk will have less data than the rest.
    sample_bin_idx = np.array_split(sorted_idx, num_clients * 2)
    sample_bin_idx = np.random.permutation(sample_bin_idx)
    num_bins = len(sample_bin_idx)
    
    user_loaders = []
    
    for i in range(0, num_bins, 2):
        client_data_idx = sample_bin_idx[i]
        if i + 1 < num_bins:
            client_data_idx = np.append(client_data_idx, sample_bin_idx[i+1])
            
        cur_sampler = torch.utils.data.BatchSampler(client_data_idx, 
                                                    batch_size, 
                                                    drop_last=False)
        cur_loader = torch.utils.data.DataLoader(train_data,
                                                 batch_sampler=cur_sampler)
        user_loaders.append(cur_loader)

    return user_loaders

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
                         batch_size), global_test_loader
    
    elif mode == "non-iid":
        return non_iid_split(num_clients, 
                             train_data,  
                             batch_size), global_test_loader
    
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
    
    print("Load MNIST 10")
    user_loaders, test_loader = get_data(10, "mnist")
    assert len(user_loaders) == 10
    
    print("Load MNIST 100")

    users_data, test_loader = get_data(100, "mnist", mode="non-iid")
    print(len(users_data))
    print(len(users_data[-1]))
    for data, label in users_data[0]:
        print(label)

    
