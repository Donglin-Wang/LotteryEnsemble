from experiment import build_args, run_experiments
from client import Client
from server import Server
from genesis import ClientGenesis, ServerGenesis

if __name__ == '__main__':
    overrides = {
        'log_folder': './weights',
        'running_on_cloud': False
    }

    data_split = 'non-iid'
    num_rounds = 400
    num_local_epoch = 10
    num_clients = 400
    batch_size = 32
    rate_unbalance = 1
    n_samples = 20
    n_class = 2

    experiments = {
<<<<<<< HEAD
        data_split:  # this key determines the output folder name under log_folder
            build_args(arch='cnn',
=======
        'fed_avg_ashwin':  # this key determines the output folder name under log_folder
            build_args(arch='mlp',
>>>>>>> a7f4d14... adding fed_ashiwin + fed_ashiwn lottery case
                       dataset='mnist',
                       data_split=data_split,
                       client=Client,
                       server=Server,
                       avg_logic=None,
                       num_clients=num_clients,
                       comm_rounds=num_rounds,
                       frac=0.025,
                       prune_step=0.0,
                       prune_percent=2,
                       acc_thresh=2,
                       client_epoch=num_local_epoch,
                       batch_size=batch_size,
                       lr=0.001,
                       rate_unbalance=rate_unbalance,
                       n_samples=n_samples,
                       n_class=n_class),
        'lottery_fl_fed_avg_ashwin':  # this key determines the output folder name under log_folder
            build_args(arch='mlp',
                       dataset='mnist',
                       data_split=data_split,
                       client=Client,
                       server=Server,
                       avg_logic='lottery_fl_avg',
                       num_clients=num_clients,
                       comm_rounds=num_rounds,
                       frac=0.025,
                       prune_step=0.2,
                       prune_percent=0.5,
                       acc_thresh=0.5,
                       client_epoch=num_local_epoch,
                       batch_size=batch_size,
                       lr=0.001,
                       rate_unbalance=rate_unbalance,
                       n_samples=n_samples,
                       n_class=n_class),

        # 'genesis':  # this key determines the output folder name under log_folder
        #     build_args(arch='mlp',
        #                dataset='mnist',
        #                data_split='non-iid',
        #                client=ClientGenesis,
        #                server=ServerGenesis,
        #                num_clients=num_clients,
        #                comm_rounds=num_rounds,
        #                frac=0.025,
        #                prune_step=0.1,
        #                prune_percent=0.45,
        #                acc_thresh=0.8,
        #                client_epoch=num_local_epoch,
        #                batch_size=batch_size,
        #                lr=0.001,
        #                rate_unbalance=rate_unbalance,
        #                n_samples=n_samples,
        #                n_class=n_class),

        # !!! NOTE: This experiment still uses the old style specification. Adjust to resemble an earlier example.
        # !!!
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

        # !!! NOTE: This experiment still uses the old style specification. Adjust to resemble an earlier example.
        # !!!
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

        # !!! NOTE: This experiment still uses the old style specification. Adjust to resemble an earlier example.
        # !!!
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
    }

    # To run 1 or more set selection
    selection = ['lottery_fl_fed_avg_ashwin',]
    run_experiments({k: experiments[k] for k in selection}, overrides)
