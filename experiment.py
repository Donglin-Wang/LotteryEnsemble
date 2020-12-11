import datetime, time, os, math
import numpy as np
import matplotlib.pyplot as plt
from client import Client
from server import Server
from genesis import ClientGenesis, ServerGenesis
from datasource import get_data
from util import create_model
import torch
torch.manual_seed(0)
np.random.seed(0)

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


    mu_client_losses = np.zeros((num_clients, num_rounds))

    for i, c in enumerate(clients):
        for j, loss in enumerate(c.losses):
            mu_client_losses[i][j] = loss[-1]


    with open(f'{save_path}/mu_client_losses.npy', 'wb') as f:
        np.save(f, mu_client_losses)

    mu_part_client_accs = np.zeros((num_clients, num_rounds))

    for i, c in enumerate(clients):
        for j, acc in enumerate(c.accuracies):
            mu_part_client_accs[i][j] = acc[-1]

    with open(f'{save_path}/mu_client_accs.npy', 'wb') as f:
        np.save(f, mu_part_client_accs)


    mu_client_pr_rates = np.zeros((num_clients, num_rounds))
    for i, c in enumerate(clients):
        mu_client_pr_rates[i] = c.prune_rates

    with open(f'{save_path}/mu_client_pr_rates.npy', 'wb') as f:
        np.save(f, mu_client_pr_rates)


    mu_client_losses_by_r = mu_client_losses.mean(axis=0)
    with open(f'{save_path}/mu_client_losses_by_r.npy', 'wb') as f:
        np.save(f, mu_client_losses_by_r)


    mu_client_part_accs_by_r = mu_part_client_accs.mean(axis=0)
    with open(f'{save_path}/mu_client_part_accs_by_r.npy', 'wb') as f:
        np.save(f, mu_client_part_accs_by_r)

    mu_client_pr_rate_by_r = mu_client_pr_rates.mean(axis=0)
    with open(f'{save_path}/mu_client_pr_rate_by_r.npy', 'wb') as f:
        np.save(f, mu_client_pr_rate_by_r)

    mu_client_accs_by_r = server.client_accuracies.mean(axis=0)
    with open(f'{save_path}/mu_client_accs_by_r.npy', 'wb') as f:
        np.save(f, mu_client_accs_by_r)

    with open(f'{save_path}/client_accs.npy', 'wb') as f:
        np.save(f, server.client_accuracies)

    with open(f'{save_path}/selected_client_tally.npy', 'wb') as f:
        np.save(f, server.selected_client_tally)


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

    # log class and mask overlap for every pair of clients
    mask_start_time = time.time()
    num_clients = len(clients)
    overlap_arr = np.zeros((int(num_clients * (num_clients - 1) / 2), 5), dtype='float32')
    i = 0
    for c1 in range(len(clients)):
        for c2 in range(c1 + 1, len(clients)):
            mask_c1 = clients[c1].get_mask()
            mask_c2 = clients[c2].get_mask()
            # Sanity check:
            if mask_c1.sum() == 0:
                print(f'PROBLEM: Client {c1} has mask of all zeros.')
            if mask_c2.sum() == 0:
                print(f'PROBLEM: Client {c2} has mask of all zeros.')
            mask_overlap = (mask_c1 * mask_c2).sum()
            combined_mask_extent = np.logical_or(mask_c1, mask_c2)
            normalized_mask_overlap = mask_overlap / combined_mask_extent.sum()  # denominator should never be 0
            if math.isnan(normalized_mask_overlap):
                print(f'Nan found for clients {c1}, {c2}. Mask sums: {mask_c1.sum()}, {mask_c2.sum()}')

            class_overlap = 0
            classes_c1 = clients[c1].get_class_counts('train')
            classes_c2 = clients[c2].get_class_counts('train')
            for k, v in classes_c1.items():
                if k in classes_c2:
                    class_overlap += 1
            overlap_arr[i] = [c1, c2, class_overlap, mask_overlap, normalized_mask_overlap]
            i += 1
    np.save(f'{save_path}/class_and_mask_overlap.npy', overlap_arr)
    # figure for class and mask overlap
    class_overlaps = overlap_arr[:, 2]
    ave_mask_overlap = np.zeros((3,), dtype='float32')
    for co in [0, 1, 2]:
        if np.count_nonzero(class_overlaps == co) > 0:
            ave_mask_overlap[co] = overlap_arr[:, 4][class_overlaps == co].mean()
        else:
            ave_mask_overlap[co] = 0
    fig, axs = plt.subplots(1, 1)
    plt.bar([0, 1, 2], ave_mask_overlap)
    axs.set_title("Summary of pairwise client mask overlap")
    axs.set_xlabel("Class overlap")
    axs.set_ylabel("Mean mask overlap (# weights)")
    axs.set_xticks([0, 1, 2])
    fig.savefig(f"{save_path}/mask_overlap_by_class_overlap.png")
    print(f'Time to compute mask info:   {str(datetime.timedelta(seconds=round(time.time() - mask_start_time)))}\n')


def run_experiment(args, overrides):
    for  k, v in overrides.items():
        setattr(args, k, v)
    args.log_folder = overrides['log_folder'] + '/' + overrides['exp_name']
    print("Started getting data")
    (client_loaders, val_loaders, test_loader), global_test_loader =\
        get_data(args.num_clients,
                 args.dataset, mode=args.data_split, batch_size=args.batch_size,
                 num_train_samples_perclass = args.n_samples, n_class = args.n_class, rate_unbalance=args.rate_unbalance)
    print("Finished getting data")
    clients = []
    print("Initializing clients")
    for i in range(args.num_clients):
        print("Client " + str(i))
        clients.append(args.client(args, client_loaders[i], test_loader[i], client_id=i))
    
    server = args.server(args, np.array(clients, dtype=np.object), test_loader=global_test_loader)
    print("Now running the algorithm")
    server.server_update()
    return server, clients


def run_experiments(experiments, overrides):
    run_times = {}
    start = time.time()
    for exp_name, exp_settings in experiments.items():
        overrides['exp_name'] = exp_name
        run_start = time.time()
        server, clients = run_experiment(exp_settings, overrides)
        log_experiment(server, clients, exp_name, exp_settings)
        run_times[exp_name] = round(time.time() - run_start)
    end = time.time()

    print('Runtimes:')
    for exp_name, t in run_times.items():
        print(f'  {exp_name}: {str(datetime.timedelta(seconds=t))}')
    print(f'  TOTAL: {str(datetime.timedelta(seconds=round(end - start)))}')
