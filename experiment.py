import torch
import torch.nn as nn
from util import evaluate, prune_fixed_amount, train, copy_model, get_prune_summary
import math
from client import Client
from server import Server
from datasource import get_data

# User defined update method
def client_update_method1(client_self, global_model, global_init_model):
    print(f'***** Client #{client_self.client_id} *****', flush=True)
    # Checking if the client object has been properly initialized
    assert isinstance(client_self.model, nn.Module), "A model must be a PyTorch module"
    assert 0 <= client_self.args.prune_percent <= 1, "The prune percentage must be between 0 and 1"
    assert client_self.args.client_epoch, '"args" must contain a "client_epoch" field'
    assert client_self.test_loader, "test_loader field does not exist. Check if the client is initialized correctly"
    assert client_self.train_loader, "train_loader field does not exist. Check if the client is initialized correctly"
    assert isinstance(client_self.train_loader, torch.utils.data.DataLoader), "train_loader must be a DataLoader type"
    assert isinstance(client_self.test_loader, torch.utils.data.DataLoader), "test_loader must be a DataLoader type"
    
    
    client_self.model = copy_model(global_model, 
                                   client_self.args.dataset,
                                   client_self.args.arch)
    
    num_pruned, num_params = get_prune_summary(client_self.model)
    cur_prune_rate = num_pruned / num_params
    prune_step = math.floor(num_params * client_self.args.prune_step)
    
    for i in range(client_self.args.client_epoch):
        print(f'Epoch {i+1}')
        train(client_self.model, 
              client_self.train_loader, 
              lr=client_self.args.lr,
              verbose=client_self.args.train_verbosity)
    
    score = evaluate(client_self.model, 
                     client_self.test_loader, 
                     verbose=client_self.args.test_verbosity)
    
    if score['Accuracy'][0] > client_self.args.acc_thresh and cur_prune_rate < client_self.args.prune_percent:
        
        prune_fixed_amount(client_self.model, 
                           prune_step,
                           verbose=client_self.args.prune_verbosity)
    
    
    return copy_model(client_self.model, client_self.args.dataset, client_self.args.arch)
    
    # <INSERT MORE UPDATE METHOD HERE>

# Method for running the experiment
# If you want to change the default values, change it here in the funciton signature
def build_args(arch='mlp',
               dataset='mnist',
               data_split='iid',
               num_clients=10,
               lr=0.001,
               batch_size=4,
               comm_rounds=10,
               frac=0.3,
               client_epoch=10,
               acc_thresh=0.5,
               prune_iterations=None,
               prune_percent=0.75,
               prune_step=0.15,
               prune_type=None,
               train_verbosity=False,
               test_verbosity=False,
               prune_verbosity=False,
               val_freq=0
               ):
    
    args = type('', (), {})()
    args.arch = arch
    args.dataset = dataset
    args.data_split = data_split
    args.num_clients = num_clients
    args.lr = lr
    args.batch_size = batch_size
    args.comm_rounds = comm_rounds
    args.frac = frac
    args.client_epoch = client_epoch
    args.acc_thresh = acc_thresh
    args.prune_iterations = prune_iterations
    args.prune_percent = prune_percent
    args.prune_step= prune_step
    args.prune_type = prune_type
    args.train_verbosity = train_verbosity
    args.test_verbosity = test_verbosity
    args.prune_verbosity = prune_verbosity
    args.val_freq = val_freq
    return args
    
def run_experiment(args, client_update, server_update):
    
    client_loaders, test_loader = get_data(args.num_clients, 
                                           args.dataset, 
                                           mode=args.data_split, 
                                           batch_size=args.batch_size)
    
    clients = []
    
    for i in range(args.num_clients):
        clients.append(Client(args, 
                              client_loaders[i], 
                              test_loader, 
                              client_update_method=client_update,
                              client_id=i))
    
    server = Server(args, clients, server_update_method=server_update, test_loader=test_loader)
    
    server.server_update()
    
if __name__ == '__main__':
    
    experiments = [
        {
            'args': build_args(client_epoch=10,
                               comm_rounds=400,
                               frac=0.5,
                               prune_step=0.1,
                               acc_thresh=0.5,
                               batch_size=32,
                               num_clients=400),
            'client_update': None,
            'server_update': None
        },
        # # This experiment contains a custom update method that client uses
        # {
        #     'args': build_args(client_epoch=1, 
        #                        comm_rounds=2, 
        #                        frac=0.2, 
        #                        acc_thresh=0.1),
        #     'client_update': client_update_method1,
        #     'server_update': None
        # }
    ]
    
    for experiment in experiments:
        run_experiment(experiment['args'], 
                       experiment['client_update'],
                       experiment['server_update'])
