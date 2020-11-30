import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import random
import torch
import torchvision as tv
import torchvision.transforms as tf
from sklearn.utils import shuffle
torch.manual_seed(0)
np.random.seed(0)

class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)




#MNIST Non-IID Dataset split
def get_dataset_mnist_extr_noniid(num_users, n_class, nsamples, rate_unbalance, batch_size):
    data_dir = '../data/mnist/'
    apply_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                   transform=apply_transform)

    test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                  transform=apply_transform)

    # Chose euqal splits for every user
    user_groups_train, user_groups_test = mnist_extr_noniid(train_dataset, test_dataset, num_users, n_class, nsamples, rate_unbalance)

    #Training data loaders
    user_train_loaders = []
    #Test data loaders
    user_test_loaders = []


    #Trainning data
    for (_,c_t_idx) in user_groups_train.items():
        user_train_loaders.append(DataLoader(DatasetSplit(train_dataset, c_t_idx),
                                             batch_size=batch_size, shuffle=True))

    for  (_,c_t_idx) in user_groups_test.items():
        if len(c_t_idx) == 1000:
            c_t_idx = np.concatenate((c_t_idx, c_t_idx))
        if len(c_t_idx) > 2000:
            c_t_idx = c_t_idx[:2001]
        user_test_loaders.append(DataLoader(DatasetSplit(test_dataset, c_t_idx),
                                            batch_size=batch_size, shuffle=True))




    return user_train_loaders, user_test_loaders

def mnist_extr_noniid(train_dataset, test_dataset, num_users, n_class, num_samples, rate_unbalance):
    num_shards_train, num_imgs_train = int(60000/num_samples), num_samples
    num_classes = 10
    num_imgs_perc_test, num_imgs_test_total = 1000, 10000
    assert(n_class * num_users <= num_shards_train)
    assert(n_class <= num_classes)
    idx_class = [i for i in range(num_classes)]
    idx_shard = [i for i in range(num_shards_train)]
    dict_users_train = {i: np.array([]) for i in range(num_users)}
    dict_users_test = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards_train*num_imgs_train)
    # labels = dataset.train_labels.numpy()
    labels = np.array(train_dataset.targets)
    idxs_test = np.arange(num_imgs_test_total)
    labels_test = np.array(test_dataset.targets)
    #labels_test_raw = np.array(test_dataset.targets)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    labels = idxs_labels[1, :]

    idxs_labels_test = np.vstack((idxs_test, labels_test))
    idxs_labels_test = idxs_labels_test[:, idxs_labels_test[1, :].argsort()]
    idxs_test = idxs_labels_test[0, :]
    #print(idxs_labels_test[1, :])

    # divide and assign
    for i in range(num_users):
        user_labels = np.array([])
        rand_set = set(np.random.choice(idx_shard, n_class, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        unbalance_flag = 0
        for rand in rand_set:
            if unbalance_flag == 0:
                dict_users_train[i] = np.concatenate(
                    (dict_users_train[i], idxs[rand*num_imgs_train:(rand+1)*num_imgs_train]), axis=0)
                user_labels = np.concatenate((user_labels, labels[rand*num_imgs_train:(rand+1)*num_imgs_train]), axis=0)
            else:
                dict_users_train[i] = np.concatenate(
                    (dict_users_train[i], idxs[rand*num_imgs_train:int((rand+rate_unbalance)*num_imgs_train)]), axis=0)
                user_labels = np.concatenate((user_labels, labels[rand*num_imgs_train:int((rand+rate_unbalance)*num_imgs_train)]), axis=0)
            unbalance_flag = 1
        user_labels_set = set(user_labels)
        #print(user_labels_set)
        #print(user_labels)
        for label in user_labels_set:
            dict_users_test[i] = np.concatenate((dict_users_test[i], idxs_test[int(label)*num_imgs_perc_test:int(label+1)*num_imgs_perc_test]), axis=0)
        #print(set(labels_test_raw[dict_users_test[i].astype(int)]))
    return dict_users_train, dict_users_test



# Given each user euqal number of samples if possible. If not, the last user
# gets whatever is left after other users had their shares



#CIFAR10 Non-IID Dataset split
def get_dataset_cifar10_extr_noniid(num_users, n_class, nsamples, rate_unbalance, batch_size):
    data_dir = '../data/cifar/'
    apply_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                   transform=apply_transform)

    test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                      transform=apply_transform)

    # Chose euqal splits for every user
    user_groups_train, user_groups_test = cifar_extr_noniid(train_dataset, test_dataset, num_users, n_class, nsamples, rate_unbalance)

    #Training data loaders
    user_train_loaders = []
    #Test data loaders
    user_test_loaders = []


    #Trainning data
    for (_,c_t_idx) in user_groups_train.items():
        user_train_loaders.append(DataLoader(DatasetSplit(train_dataset, c_t_idx),
                                             batch_size=batch_size, shuffle=True))

    for  (_,c_t_idx) in user_groups_test.items():
        user_test_loaders.append(DataLoader(DatasetSplit(test_dataset, c_t_idx),
                                            batch_size=batch_size, shuffle=True))

    return user_train_loaders, user_test_loaders


def cifar_extr_noniid(train_dataset, test_dataset, num_users, n_class, num_samples, rate_unbalance):
    num_shards_train, num_imgs_train = int(50000/num_samples), num_samples
    num_classes = 10
    num_imgs_perc_test, num_imgs_test_total = 1000, 10000
    assert(n_class * num_users <= num_shards_train)
    assert(n_class <= num_classes)
    idx_class = [i for i in range(num_classes)]
    idx_shard = [i for i in range(num_shards_train)]
    dict_users_train = {i: np.array([]) for i in range(num_users)}
    dict_users_test = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards_train*num_imgs_train)
    # labels = dataset.train_labels.numpy()
    labels = np.array(train_dataset.targets)
    idxs_test = np.arange(num_imgs_test_total)
    labels_test = np.array(test_dataset.targets)
    #labels_test_raw = np.array(test_dataset.targets)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    labels = idxs_labels[1, :]

    idxs_labels_test = np.vstack((idxs_test, labels_test))
    idxs_labels_test = idxs_labels_test[:, idxs_labels_test[1, :].argsort()]
    idxs_test = idxs_labels_test[0, :]
    #print(idxs_labels_test[1, :])


    # divide and assign
    for i in range(num_users):
        user_labels = np.array([])
        rand_set = set(np.random.choice(idx_shard, n_class, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        unbalance_flag = 0
        for rand in rand_set:
            if unbalance_flag == 0:
                dict_users_train[i] = np.concatenate(
                    (dict_users_train[i], idxs[rand*num_imgs_train:(rand+1)*num_imgs_train]), axis=0)
                user_labels = np.concatenate((user_labels, labels[rand*num_imgs_train:(rand+1)*num_imgs_train]), axis=0)
            else:
                dict_users_train[i] = np.concatenate(
                    (dict_users_train[i], idxs[rand*num_imgs_train:int((rand+rate_unbalance)*num_imgs_train)]), axis=0)
                user_labels = np.concatenate((user_labels, labels[rand*num_imgs_train:int((rand+rate_unbalance)*num_imgs_train)]), axis=0)
            unbalance_flag = 1
        user_labels_set = set(user_labels)
        #print(user_labels_set)
        #print(user_labels)
        for label in user_labels_set:
            dict_users_test[i] = np.concatenate((dict_users_test[i], idxs_test[int(label)*num_imgs_perc_test:int(label+1)*num_imgs_perc_test]), axis=0)
        #print(set(labels_test_raw[dict_users_test[i].astype(int)]))

    return dict_users_train, dict_users_test







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
        cur_sampler = torch.utils.data.BatchSampler(client_data_idx,
                                                    batch_size,
                                                    drop_last=False)
        cur_loader = torch.utils.data.DataLoader(train_data,
                                                 batch_sampler=cur_sampler)
        user_loaders.append(cur_loader)


        #Test data
        cur_sampler_test = torch.utils.data.BatchSampler(client_test_data_idx,
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
             max_shard=30, n_class=2, n_samples=20, rate_unbalance=1.0):

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

        if dataset_name == "mnist":
            return get_dataset_mnist_extr_noniid(num_clients, n_class, n_samples, rate_unbalance, batch_size), global_test_loader
        elif dataset_name == "cifar10":
            return get_dataset_cifar10_extr_noniid(num_clients, n_class, n_samples, rate_unbalance, batch_size), global_test_loader
        else:
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

    print("Load cifar10 non-iid")
    (users_data, test_loader), global_test_loader = get_data(400, "cifar10", mode="non-iid", batch_size=10, rate_unbalance=1)
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
