import math
import numpy as np
from server import Server
from client import Client
from util import copy_model, evaluate, fed_avg, get_prune_summary, prune_fixed_amount, train

class ClientGenesis(Client):
    def __init__(self, args, train_loader, test_loader, client_id):
        super().__init__(args, train_loader, test_loader, client_id)

    def client_update(self, global_model, global_init_model, round_index):
        self.elapsed_comm_rounds += 1
        print(f'***** Client #{self.client_id} *****', flush=True)
        self.model = copy_model(global_model, self.args.dataset, self.args.arch)

        losses = []
        accuracies = []
        for i in range(self.args.client_epoch):
            train_score = train(self.client_id, i, self.model,
                                self.train_loader,
                                lr=self.args.lr,
                                verbose=self.args.train_verbosity)
            losses.append(train_score['Loss'][-1].data.item())
            accuracies.append(train_score['Accuracy'][-1])

        num_pruned, num_params = get_prune_summary(self.model)
        cur_prune_rate = num_pruned / num_params
        print(f"num_pruned {num_pruned}, num_params {num_params}, cur_prune_rate {cur_prune_rate}")
        self.losses[round_index] = np.array(losses)
        self.accuracies[round_index] = np.array(accuracies)
        self.prune_rates[round_index] = cur_prune_rate


class ServerGenesis(Server):
    def __init__(self, args, clients, test_loader=None):
        super().__init__(args, clients, test_loader)

    def server_update(self):
        self.elapsed_comm_rounds += 1
        self.global_models.train()

        for i in range(0, self.comm_rounds):
            print('-------------------------------------', flush=True)
            print(f'Communication Round #{i}', flush=True)
            print('-------------------------------------', flush=True)
            for c in [self.clients[i] for i in np.random.choice(self.num_clients,
                                                                max(int(self.frac * self.num_clients), 1),
                                                                replace=False)]:
                c.client_update(self.global_models, self.global_init_model, i)

            self.global_models = fed_avg([c.model for c in self.clients],
                                         self.args.dataset, self.args.arch, self.client_data_num)

            # gather server accuracies
            eval_score = evaluate(self.global_models, self.test_loader, verbose=self.args.test_verbosity)
            self.accuracies[i] = eval_score['Accuracy'][-1]
            # gather client accuracies
            for k, m in enumerate(self.clients):
                self.client_accuracies[k][i] = m.evaluate()

            # prune global model if appropriate
            num_pruned, num_params = get_prune_summary(self.global_models)
            cur_prune_rate = num_pruned / num_params
            if self.client_accuracies[:, i].mean() > self.args.acc_thresh and cur_prune_rate < self.args.prune_percent:
                prune_step = math.floor(num_params * self.args.prune_step)
                prune_fixed_amount(self.global_models, prune_step, verbose=self.args.prune_verbosity)
                self.global_models = copy_model(global_init_model,
                                                self.args.dataset,
                                                self.args.arch,
                                                dict(self.global_models.named_buffers()))
