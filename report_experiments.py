from experiment import build_args
from client import Client
from server import Server
from genesis import ClientGenesis, ServerGenesis


MNIST_experiments = {
    'MNIST_standalone':
        build_args(arch='mlp',
                   dataset='mnist',
                   data_split='non-iid',
                   client=Client,
                   server=Server,
                   avg_logic='standalone',
                   num_clients=20,
                   comm_rounds=400,
                   frac=.025,
                   prune_step=0.0,     # effective disable pruning
                   prune_percent=2,    # effective disable pruning
                   acc_thresh=2,       # effective disable pruning
                   client_epoch=10,
                   batch_size=32,
                   lr=0.001,
                   rate_unbalance=1.0,
                   n_samples=20,
                   n_class=2),

    'MNIST_FedAvg':
        build_args(arch='mlp',
                   dataset='mnist',
                   data_split='non-iid',
                   client=Client,
                   server=Server,
                   avg_logic='fed_avg',
                   num_clients=400,
                   comm_rounds=400,
                   frac=.025,
                   prune_step=0.0,  # effective disable pruning
                   prune_percent=2, # effective disable pruning
                   acc_thresh=2,    # effective disable pruning
                   client_epoch=10,
                   batch_size=32,
                   lr=0.001,
                   rate_unbalance=1.0,
                   n_samples=20,
                   n_class=2),

    'MNIST_LotteryFL':
        build_args(arch='mlp',
                   dataset='mnist',
                   data_split='non-iid',
                   client=Client,
                   server=Server,
                   avg_logic='lottery_fl_avg',
                   num_clients=400,
                   comm_rounds=400,
                   frac=.025,
                   prune_step=0.2,
                   prune_percent=0.1,
                   acc_thresh=0.5,
                   client_epoch=10,
                   batch_size=32,
                   lr=0.001,
                   rate_unbalance=1.0,
                   n_samples=20,
                   n_class=2),

    'MNIST_Genesis':
        build_args(arch='mlp',
                   dataset='mnist',
                   data_split='non-iid',
                   client=ClientGenesis,
                   server=ServerGenesis,
                   avg_logic='lottery_fl_avg',
                   num_clients=400,
                   comm_rounds=400,
                   frac=.025,
                   prune_step=0.2,
                   prune_percent=0.1,
                   acc_thresh=0.75,
                   client_epoch=10,
                   batch_size=32,
                   lr=0.001,
                   rate_unbalance=1.0,
                   n_samples=20,
                   n_class=2),
}

# These are the experiments for the CIFAR10 dataset using a cnn model
CIFAR10_experiments = {
    'CIFAR10_standalone':
        build_args(arch='cnn',
                   dataset='cifar10',
                   data_split='non-iid',
                   client=Client,
                   server=Server,
                   avg_logic='standalone',
                   num_clients=400,
                   comm_rounds=400,
                   frac=.025,
                   prune_step=0.0,     # effective disable pruning
                   prune_percent=2,    # effective disable pruning
                   acc_thresh=2,       # effective disable pruning
                   client_epoch=10,
                   batch_size=32,
                   lr=0.001,
                   rate_unbalance=1.0,
                   n_samples=20,
                   n_class=2),

    'CIFAR10_FedAvg':
        build_args(arch='cnn',
                   dataset='cifar10',
                   data_split='non-iid',
                   client=Client,
                   server=Server,
                   avg_logic='fed_avg',
                   num_clients=400,
                   comm_rounds=400,
                   frac=.025,
                   prune_step=0.0,  # effective disable pruning
                   prune_percent=2, # effective disable pruning
                   acc_thresh=2,    # effective disable pruning
                   client_epoch=10,
                   batch_size=32,
                   lr=0.001,
                   rate_unbalance=1.0,
                   n_samples=20,
                   n_class=2),

    'CIFAR10_LotteryFL':
        build_args(arch='cnn',
                   dataset='cifar10',
                   data_split='non-iid',
                   client=Client,
                   server=Server,
                   avg_logic='lottery_fl_avg',
                   num_clients=400,
                   comm_rounds=400,
                   frac=.025,
                   prune_step=0.2,
                   prune_percent=0.1,
                   acc_thresh=0.5,
                   client_epoch=10,
                   batch_size=32,
                   lr=0.001,
                   rate_unbalance=1.0,
                   n_samples=20,
                   n_class=2),

    'CIFAR10_Genesis':
        build_args(arch='cnn',
                   dataset='cifar10',
                   data_split='non-iid',
                   client=ClientGenesis,
                   server=ServerGenesis,
                   avg_logic='lottery_fl_avg',
                   num_clients=400,
                   comm_rounds=400,
                   frac=.025,
                   prune_step=0.2,
                   prune_percent=0.1,
                   acc_thresh=0.75,
                   client_epoch=10,
                   batch_size=32,
                   lr=0.001,
                   rate_unbalance=1.0,
                   n_samples=20,
                   n_class=2),
}

