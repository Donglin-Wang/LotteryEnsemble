import numpy as np
import torch

torch.manual_seed(0)
np.random.seed(0)
from util import average_weights, create_model, copy_model, log_obj, evaluate, fed_avg, lottery_fl_v2, lottery_fl_v3, \
    train_client_model, train_client_model_orig

import multiprocessing as mp

from collections import OrderedDict


class Server():

    def __init__(self, args, clients, test_loader=None):

        self.args = args
        self.comm_rounds = args.comm_rounds
        self.num_clients = args.num_clients
        self.frac = args.frac
        self.clients = np.array(clients)
        self.client_data_num = []
        self.elapsed_comm_rounds = 0
        self.accuracies = np.zeros(args.comm_rounds)
        self.client_accuracies = np.zeros((self.args.num_clients, self.args.comm_rounds))
        self.selected_client_tally = np.zeros((self.args.comm_rounds, self.args.num_clients))
        self.test_loader = test_loader
        self.init_model = create_model(args.dataset, args.arch)
        for client in self.clients:
            self.client_data_num.append(len(client.train_loader))
            client.state_dict = self.init_model.state_dict()
        self.client_data_num = np.array(self.client_data_num)

        # The extra 1 entry in client_models and global_models are used to store
        # the results after last communication round
        self.client_models = np.zeros((self.comm_rounds + 1, self.num_clients), dtype='object')
        self.global_models = None  # np.zeros((self.comm_rounds + 1,), dtype='object')

        self.global_models = self.init_model
        self.global_init_model = copy_model(self.init_model, args.dataset, args.arch)

        assert self.num_clients == len(clients), "Number of client objects does not match command line input"

    def server_update(self):
        ctx = mp.get_context('spawn')
        num_selected_clients = max(int(self.frac * self.num_clients), 1)
        with ctx.Pool(num_selected_clients) as p:
            self.elapsed_comm_rounds += 1
            # Recording the update and storing them in record
            self.global_models.train()
            for i in range(0, self.comm_rounds):
                update_or_not = [0] * self.num_clients
                # Randomly select a fraction of users to update
                idx_list = np.random.choice(range(self.num_clients),
                                            num_selected_clients,
                                            replace=False)
                for idx in idx_list:
                    update_or_not[idx] = 1

                self.selected_client_tally[i, idx_list] += 1
                for m in self.clients[idx_list]:
                    if m.model is None:
                        m.model = copy_model(self.global_models, self.args.dataset, self.args.arch)
                        m.state_dict = m.model.state_dict()

                print('-------------------------------------', flush=True)
                print(f'Communication Round #{i}', flush=True)
                print('-------------------------------------', flush=True)
                a = p.starmap(train_client_model_orig, [(self.args.acc_thresh, self.args.prune_percent, self.args.prune_step,
                                                    self.args.prune_verbosity, self.args.dataset, self.args.arch,
                                                    self.args.lr, False,self.args.client_epoch, self.args.log_folder,
                                                    i, c, self.clients[c].model.state_dict(),
                                                    self.global_models.state_dict() if self.args.avg_logic != "standalone" else self.clients[c].model.state_dict(),
                                                    self.clients[c].train_loader,
                                                    self.clients[c].test_loader, self.clients[c].last_client_acc
                                                    ) for c in idx_list])
                for (k, vals) in enumerate(a):
                    self.clients[idx_list[k]].losses[i:] = vals[0]
                    self.clients[idx_list[k]].accuracies[i:] = vals[1]
                    self.clients[idx_list[k]].prune_rates[i:] = vals[2]
                    self.clients[idx_list[k]].model.load_state_dict(vals[3])
                    self.client_accuracies[idx_list[k]][i:] = vals[4]

                if self.args.avg_logic == "fed_avg":
                    self.global_models = fed_avg([m.model for m in self.clients], self.args.dataset,
                                                 self.args.arch,
                                                 self.client_data_num)
                elif self.args.avg_logic == 'lottery_fl_v2':
                    self.global_models = lottery_fl_v2(self.global_models, [m.model for m in self.clients[idx_list]],
                                                       self.args.dataset,
                                                       self.args.arch,
                                                       self.client_data_num[idx_list])
                elif self.args.avg_logic == 'lottery_fl_v3':
                    self.global_models = lottery_fl_v3(self.global_models, [m.model for m in self.clients[idx_list]],
                                                       self.args.dataset,
                                                       self.args.arch,
                                                       self.client_data_num[idx_list])
                elif self.args.avg_logic == "standalone":
                    pass  # no averaging in the server
                else:
                    self.global_models = average_weights([m.model for m in self.clients[idx_list]], self.args.dataset,
                                                         self.args.arch,
                                                         self.client_data_num[idx_list])
                print(f"Mean client accs: {self.client_accuracies.mean(axis=0)[i]}")
