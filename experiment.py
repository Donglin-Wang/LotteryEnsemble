import time
import numpy as np
from client import Client
from server import Server
from genesis import ClientGenesis, ServerGenesis
from lottery_fl_ds import get_data


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
               val_freq=0,
               avg_logic=None,
               n_class = 2,
               n_samples = 20,
               rate_unbalance = 1
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
    args.avg_logic = avg_logic
    args.n_class = n_class
    args.n_samples = n_samples
    args.rate_unbalance = rate_unbalance
    return args
    
def run_experiment(args, client_factory, server_factory):
    (client_loaders, test_loader), global_test_loader =\
        get_data(args.num_clients,
                 args.dataset, mode=args.data_split, batch_size=args.batch_size,
                 n_samples = args.n_samples, n_class = args.n_class, rate_unbalance=args.rate_unbalance)

    clients = []
    for i in range(args.num_clients):
        clients.append(client_factory(args, client_loaders[i], test_loader[i], client_id=i))
    
    server = server_factory(args, np.array(clients, dtype=np.object), test_loader=global_test_loader)
    
    server.server_update()
    return server, clients
    
if __name__ == '__main__':
    data_split = 'iid'
    num_rounds = 10
    num_local_epoch = 10
    num_clients = 10
    batch_size = 32
    avg_logic = "fed_avg"
    running_on_cloud = False
    n_class = 2
    n_samples = 20
    rate_unbalance = 1


    experiments = [
        # This exepriment's setting is all default
        {
            'args': build_args(data_split=data_split,
                               client_epoch=num_local_epoch,
                               comm_rounds=num_rounds,
                               frac=1,
                               prune_step=0.1,
                               acc_thresh=0.75,
                               batch_size=batch_size,
                               num_clients=num_clients,
                               avg_logic=avg_logic, rate_unbalance = rate_unbalance, n_samples = n_samples, n_class = n_class),
            'client': Client,
            'server': Server
        },
        # {
        #     'args': build_args(data_split=data_split,
        #                        client_epoch=num_local_epoch,
        #                        comm_rounds=num_rounds,
        #                        frac=1,
        #                        prune_step=0.1,
        #                        acc_thresh=0.75,
        #                        batch_size=batch_size,
        #                        num_clients=num_clients,
        #                        avg_logic=avg_logic, rate_unbalance=rate_unbalance, n_samples=n_samples,
        #                        n_class=n_class),
        #     'client': ClientGenesis,
        #     'server': ServerGenesis
        # },
        # Ashwin RJ non-iid
        # This experiment contains a custom update method that client uses
        # {
        #     'args': build_args(data_split = 'non-iid',
        #                        client_epoch=10,
        #                        comm_rounds=10,
        #                        frac=0.1,
        #                        prune_step=0.1,
        #                        acc_thresh=2,
        #                        batch_size=10,
        #                        num_clients=100,
        #                        avg_logic=None),
        #     'client': None,
        #     'server': None
        # },
        #  Ashwin RJ iid
        # {
        #     'args': build_args(data_split = 'iid',
        #                        client_epoch=10,
        #                        comm_rounds=10,
        #                        frac=0.1,
        #                        prune_step=0.1,
        #                        acc_thresh=2,
        #                        batch_size=10,
        #                        num_clients=100,
        #                        avg_logic=None),
        #     'client': None,
        #     'server': None
        # },
        # Fed Avg non-iid
        # {
        #     'args': build_args(data_split = 'non-iid',
        #                        client_epoch=10,
        #                        comm_rounds=10,
        #                        frac=0.1,
        #                        prune_step=0.1,
        #                        acc_thresh=2,
        #                        batch_size=10,
        #                        num_clients=100,
        #                        avg_logic='fed_avg'),
        #     'client': None,
        #     'server': None
        # },
        # Lottery FL non-iid
        # {
        #     'args': build_args(data_split = 'non-iid',
        #                        client_epoch=10,
        #                        comm_rounds=10,
        #                        frac=0.1,
        #                        prune_step=0.1,
        #                        acc_thresh=0.5,
        #                        batch_size=10,
        #                        num_clients=100,
        #                        avg_logic='lottery_fl_avg'),
        #     'client': None,
        #     'server': None
        # }
    ]

    save_path = f"{ './MyDrive' if running_on_cloud else '.' }/weights/{data_split}"

    experiment = experiments[0]
    start = time.time()
    server, clients = run_experiment(experiment['args'],
                                     experiment['client'],
                                     experiment['server'])
    end = time.time()

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


    with open(f'{save_path}/mu_client_losses.npy', 'wb') as f:
        np.save(f, mu_client_losses)

    mu_part_client_accs = np.zeros((num_clients, num_rounds, num_local_epoch))

    for i, c in enumerate(clients):
        c_tmp_acc = np.zeros((num_rounds, num_local_epoch))
        for j, acc in enumerate(c.accuracies):
            c_tmp_acc[j] = np.array(acc)
        mu_part_client_accs[i] = c_tmp_acc

    with open(f'{save_path}/mu_client_accs.npy', 'wb') as f:
        np.save(f, mu_part_client_accs)


    mu_client_pr_rates = np.zeros((num_clients, num_rounds))
    for i, c in enumerate(clients):
        mu_client_pr_rates[i] = c.prune_rates

    with open(f'{save_path}/mu_client_pr_rates.npy', 'wb') as f:
        np.save(f, mu_client_pr_rates)


    mu_client_losses_by_r = np.ma.masked_equal(mu_client_losses.mean(axis=2), 0).mean(axis=0).data
    with open(f'{save_path}/mu_client_losses_by_r.npy', 'wb') as f:
        np.save(f, mu_client_losses_by_r)


    mu_client_part_accs_by_r = np.ma.masked_equal(mu_part_client_accs.mean(axis=2), 0).mean(axis=0).data
    with open(f'{save_path}/mu_client_part_accs_by_r.npy', 'wb') as f:
        np.save(f, mu_client_part_accs_by_r)

    mu_client_pr_rate_by_r = np.ma.masked_equal(mu_client_pr_rates.mean(axis=0), 0).data
    with open(f'{save_path}/mu_client_pr_rate_by_r.npy', 'wb') as f:
        np.save(f, mu_client_pr_rate_by_r)

    mu_client_accs_by_r = np.ma.masked_equal(server.client_accuracies, 0).mean(axis=0).data
    with open(f'{save_path}/mu_client_accs_by_r.npy', 'wb') as f:
        np.save(f, mu_client_accs_by_r)

    with open(f'{save_path}/server_accs.npy', 'wb') as f:
        server_accs = np.array(server.accuracies)
        np.save(f, server_accs)

    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 1)
    axs.plot(range(num_rounds), server_accs)
    axs.set_title("Rounds vs Server Accuracies")
    axs.set_ylabel("Rounds")
    fig.savefig(f"{save_path}/rounds_vs_server_accs.png")

    fig, axs = plt.subplots(1, 1)
    axs.plot(range(num_rounds), mu_client_pr_rate_by_r)
    axs.set_title("Rounds vs mean Client PR Rate")
    axs.set_xlabel("Rounds")
    axs.set_ylabel("Client PR Rate")
    fig.savefig(f"{save_path}/mu_client_pr_rate_by_r.png")

    fig, axs = plt.subplots(1, 1)
    axs.plot(range(num_rounds), mu_client_part_accs_by_r)
    axs.set_title("Rounds vs mean Participating Client Train Accuracies")
    axs.set_xlabel("Rounds")
    axs.set_ylabel("Accuracies")
    fig.savefig(f"{save_path}/mu_client_part_accs_by_r.png")

    fig, axs = plt.subplots(1, 1)
    axs.plot(range(num_rounds), mu_client_accs_by_r)
    axs.set_title("Rounds vs mean All Client Accuracies")
    axs.set_xlabel("Rounds")
    axs.set_ylabel("Accuracies")
    fig.savefig(f"{save_path}/mu_client_accs_by_r.png")

    fig, axs = plt.subplots(1, 1)
    axs.plot(range(num_rounds), mu_client_losses_by_r)
    axs.set_title("Rounds vs mean Client loss")
    axs.set_xlabel("Rounds")
    axs.set_ylabel("Mean Loss")
    fig.savefig(f"{save_path}/mu_client_losses_by_r.png")


    print('Runtime: ', end - start)