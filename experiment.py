import torch
import torch.nn as nn
from util import evaluate, prune_fixed_amount, train, copy_model, get_prune_summary
import math
from client import Client
from server import Server
from datasource import get_data
import numpy as np

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
               data_split='non-iid',
               num_clients=10,
               lr=0.001,
               batch_size=4,
               comm_rounds=10,
               frac=0.3,
               client_epoch=10,
               acc_thresh=0.5,
               prune_iterations=None,
               prune_percent=0.45,
               prune_step=0.15,
               prune_type=None,
               train_verbosity=True,
               test_verbosity=True,
               prune_verbosity=True,
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
    
    (client_loaders, test_loader), global_test_loader = get_data(args.num_clients,
                                           args.dataset, 
                                           mode=args.data_split, 
                                           batch_size=args.batch_size)
    
    clients = []
    
    for i in range(args.num_clients):
        clients.append(Client(args, 
                              client_loaders[i], 
                              test_loader[i], 
                              client_update_method=client_update,
                              client_id=i))
    
    server = Server(args, np.array(clients, dtype=np.object), server_update_method=server_update,
                    test_loader=global_test_loader)
    
    server.server_update()
    return server, clients
    
if __name__ == '__main__':
    num_rounds = 10
    num_local_epoch = 10

    experiments = [
        # This exepriment's setting is all default
        {
            'args': build_args(data_split = "non-iid",
                                client_epoch=num_local_epoch,
                               comm_rounds=num_rounds,
                               frac=0.5,
                               prune_step=0.1,
                               acc_thresh=0.5,
                               batch_size=32,
                               num_clients=10),
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

    experiment = experiments[0]
    server, clients = run_experiment(experiment['args'],
                                     experiment['client_update'],
                                     experiment['server_update'])

    print("###########################################################")
    print(f"server acc {server.accuracies}")
    print("###########################################################")
    for i, c in enumerate(clients):
        print(f"client #{i} accuracies {c.accuracies}")
        print(f"client #{i} losses {c.losses}")
        print(f"client #{i} prune_rates {c.prune_rates}")
        print("\n\n\n")

    import numpy as np
    num_clients = len(clients)


    mu_client_losses = np.zeros((num_clients, num_rounds, num_local_epoch))

    for i, c in enumerate(clients):
        c_tmp_loss = np.zeros((num_rounds, num_local_epoch))
        for j, loss in enumerate(c.losses):
            c_tmp_loss[j] = np.array(loss)
        mu_client_losses[i] = c_tmp_loss


    with open('mu_client_losses.npy', 'wb') as f:
        np.save(f, mu_client_losses)

    mu_client_accs = np.zeros((num_clients, num_rounds, num_local_epoch))

    for i, c in enumerate(clients):
        c_tmp_acc = np.zeros((num_rounds, num_local_epoch))
        for j, acc in enumerate(c.accuracies):
            c_tmp_acc[j] = np.array(acc)
        mu_client_accs[i] = c_tmp_acc

    with open('mu_client_accs.npy', 'wb') as f:
        np.save(f, mu_client_accs)


    mu_client_pr_rates = np.zeros((num_clients, num_rounds))
    for i, c in enumerate(clients):
        mu_client_pr_rates[i] = c.prune_rates


    with open('mu_client_pr_rates.npy', 'wb') as f:
        np.save(f, mu_client_accs)


    mu_client_losses_by_r = np.ma.masked_equal(mu_client_losses.mean(axis=2), 0).mean(axis=0).data
    with open('mu_client_losses_by_r.npy', 'wb') as f:
        np.save(f, mu_client_losses_by_r)


    mu_client_accs_by_r = np.ma.masked_equal(mu_client_accs.mean(axis=2), 0).mean(axis=0).data
    with open('mu_client_accs_by_r.npy', 'wb') as f:
        np.save(f, mu_client_accs_by_r)

    mu_client_pr_rate_by_r = np.ma.masked_equal(mu_client_pr_rates.mean(axis=0), 0).data
    with open('mu_client_pr_rate_by_r.npy', 'wb') as f:
        np.save(f, mu_client_pr_rate_by_r)


    with open('server_accs.npy', 'wb') as f:
        server_accs = np.array(server.accuracies)
        np.save(f, server_accs)

    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 1)
    axs.plot(range(num_rounds), server_accs)
    axs.set_title("Rounds vs Server Accuracies")
    axs.set_ylabel("Rounds")
    fig.savefig("rounds_vs_server_accs.png")

    fig, axs = plt.subplots(1, 1)
    axs.plot(range(num_rounds), mu_client_pr_rate_by_r)
    axs.set_title("Rounds vs mean Client PR Rate")
    axs.set_xlabel("Rounds")
    axs.set_ylabel("client pr rate")
    fig.savefig("mu_client_pr_rate_by_r.png")

    fig, axs = plt.subplots(1, 1)
    axs.plot(range(num_rounds), mu_client_accs_by_r)
    axs.set_title("Rounds vs mean Client accuracies Rate")
    axs.set_xlabel("Rounds")
    axs.set_ylabel("accuracies")
    fig.savefig("mu_client_accs_by_r.png")

    fig, axs = plt.subplots(1, 1)
    axs.plot(range(num_rounds), mu_client_losses_by_r)
    axs.set_title("Rounds vs mean Client loss Rate")
    axs.set_xlabel("Rounds")
    axs.set_ylabel("loss Rate")
    fig.savefig("mu_client_losses_by_r.png")


