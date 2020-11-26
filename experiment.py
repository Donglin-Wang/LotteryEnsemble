import datetime, time, os
import numpy as np
import matplotlib.pyplot as plt
from client import Client
from server import Server
from genesis import ClientGenesis, ServerGenesis
from lottery_fl_ds import get_data


def build_args(arch='mlp',
               dataset='mnist',
               data_split='non-iid',
               client=Client,
               server=Server,
               n_class = 2,
               n_samples = 20,
               rate_unbalance = 1,
               avg_logic=None,
               num_clients=10,
               comm_rounds=10,
               frac=0.3,
               prune_step=0.15,
               prune_percent=0.45,
               acc_thresh=0.5,
               client_epoch=10,
               batch_size=4,
               lr=0.001,
               train_verbosity=True,
               test_verbosity=True,
               prune_verbosity=True,
               ):
    
    args = type('', (), {})()
    args.arch = arch
    args.dataset = dataset
    args.data_split = data_split
    args.client = client
    args.server = server
    args.num_clients = num_clients
    args.lr = lr
    args.batch_size = batch_size
    args.comm_rounds = comm_rounds
    args.frac = frac
    args.client_epoch = client_epoch
    args.acc_thresh = acc_thresh
    args.prune_percent = prune_percent
    args.prune_step= prune_step
    args.train_verbosity = train_verbosity
    args.test_verbosity = test_verbosity
    args.prune_verbosity = prune_verbosity
    args.avg_logic = avg_logic
    args.n_class = n_class
    args.n_samples = n_samples
    args.rate_unbalance = rate_unbalance
    return args


def log_experiment(server, clients, exp_name, exp_settings):
    print("###########################################################")
    print(f"server acc {server.accuracies}")
    print("###########################################################")
    for i, c in enumerate(clients):
        print(f"client #{i} accuracies\n{c.accuracies}")
        print(f"client #{i} losses\n{c.losses}")
        print(f"client #{i} prune_rates\n{c.prune_rates}")
        print("\n\n\n")

    num_clients     = exp_settings.num_clients
    num_rounds      = exp_settings.comm_rounds
    num_local_epoch = exp_settings.client_epoch
    save_path_root  = './MyDrive' if exp_settings.running_on_cloud else '.'
    save_path       = os.path.join(save_path_root, exp_settings.log_folder, exp_name)

    os.makedirs(save_path, exist_ok=True)

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

    mu_client_pr_rate_by_r = mu_client_pr_rates.mean(axis=0)
    with open(f'{save_path}/mu_client_pr_rate_by_r.npy', 'wb') as f:
        np.save(f, mu_client_pr_rate_by_r)

    mu_client_accs_by_r = server.client_accuracies.mean(axis=0)
    with open(f'{save_path}/mu_client_accs_by_r.npy', 'wb') as f:
        np.save(f, mu_client_accs_by_r)

    with open(f'{save_path}/server_accs.npy', 'wb') as f:
        server_accs = np.array(server.accuracies)
        np.save(f, server_accs)

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


def run_experiment(args, overrides):
    for  k, v in overrides.items():
        setattr(args, k, v)

    (client_loaders, test_loader), global_test_loader =\
        get_data(args.num_clients,
                 args.dataset, mode=args.data_split, batch_size=args.batch_size,
                 n_samples = args.n_samples, n_class = args.n_class, rate_unbalance=args.rate_unbalance)

    clients = []
    for i in range(args.num_clients):
        clients.append(args.client(args, client_loaders[i], test_loader[i], client_id=i))
    
    server = args.server(args, np.array(clients, dtype=np.object), test_loader=global_test_loader)
    
    server.server_update()
    return server, clients


def run_experiments(experiments, overrides):
    run_times = {}
    start = time.time()
    for exp_name, exp_settings in experiments.items():
        run_start = time.time()
        server, clients = run_experiment(exp_settings, overrides)
        log_experiment(server, clients, exp_name, exp_settings)
        run_times[exp_name] = round(time.time() - run_start)
    end = time.time()

    print('Runtimes:')
    for exp_name, t in run_times.items():
        print(f'  {exp_name}: {str(datetime.timedelta(seconds=t))}')
    print(f'  TOTAL: {str(datetime.timedelta(seconds=round(end - start)))}')
