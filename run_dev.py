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
    num_rounds = 1#400
    num_local_epoch = 2#10
    num_clients = 2#400
    batch_size = 32
    avg_logic = "lottery_fl_avg"
    rate_unbalance = 1
    n_samples = 20
    n_class = 2

    experiments = {
        data_split:  # this key determines the output folder name under log_folder
            build_args(arch='cnn',
                       dataset='mnist',
                       data_split=data_split,
                       client=Client,
                       server=Server,
                       avg_logic=avg_logic,
                       num_clients=num_clients,
                       comm_rounds=num_rounds,
                       frac=0.025,
                       prune_step=0.1,
                       prune_percent=0.45,
                       acc_thresh=0.75,
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
    selection = [data_split,]
    run_experiments({k: experiments[k] for k in selection}, overrides)
