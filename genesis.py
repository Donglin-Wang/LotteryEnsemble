import math
import numpy as np
from server import Server
from client import Client
from util import copy_model, evaluate, fed_avg, get_prune_summary, prune_fixed_amount, train


class ClientGenesis(Client):
    def __init__(self, args, train_loader, test_loader, client_id):
        super().__init__(args, train_loader, test_loader, client_id)


    def client_update(self, global_model, global_init_model, comm_round):
        self.elapsed_comm_rounds += 1
        print(f'***** Client #{self.client_id} *****', flush=True)
        self.model = copy_model(global_model, self.args.dataset, self.args.arch)

        losses = []
        accuracies = []
        for epoch in range(self.args.client_epoch):
            train_score = train(comm_round, self.client_id, epoch, self.model,
                                self.train_loader,
                                lr=self.args.lr,
                                verbose=self.args.train_verbosity)
            losses.append(train_score['Loss'][-1].data.item())
            accuracies.append(train_score['Accuracy'][-1])

        num_pruned, num_params = get_prune_summary(self.model)
        cur_prune_rate = num_pruned / num_params
        print(f"num_pruned {num_pruned}, num_params {num_params}, cur_prune_rate {cur_prune_rate}")
        self.losses[comm_round:] = np.array(losses)
        self.accuracies[comm_round:] = np.array(accuracies)
        self.prune_rates[comm_round:] = cur_prune_rate


class ServerGenesis(Server):
    def __init__(self, args, clients, test_loader=None):
        super().__init__(args, clients, test_loader)

    def server_update(self):
        self.elapsed_comm_rounds += 1
        self.global_models.train()

        for comm_round in range(self.comm_rounds):
            selected_clients = np.random.choice(self.num_clients,
                                                max(int(self.frac * self.num_clients), 1),
                                                replace=False)
            print('-------------------------------------', flush=True)
            print(f'Communication Round #{comm_round} Clients={selected_clients}', flush=True)
            print('-------------------------------------', flush=True)
            for c in [self.clients[i] for i in selected_clients]:
                c.client_update(self.global_models, self.global_init_model, comm_round)

            new_model = fed_avg([c.model for c in self.clients],
                                self.args.dataset, self.args.arch, self.client_data_num)
            # fed_avg clobbers the mask, so we need to copy it back into the global model
            global_buffers = dict(self.global_models.named_buffers())
            for name, buffer in new_model.named_buffers():
                buffer.data.copy_(global_buffers[name])
            self.global_models = new_model

            # server accuracies are not useful for Genesis
            self.accuracies[comm_round] = 0
            # gather client accuracies
            for k, m in enumerate(self.clients):
                if k in selected_clients:
                    self.client_accuracies[k][comm_round] = m.evaluate()
                elif comm_round > 0:
                    self.client_accuracies[k][comm_round] = self.client_accuracies[k][comm_round - 1]


            print(f"End of round accuracy: all={self.client_accuracies[:, comm_round].mean()}, "
                  f"participating={self.client_accuracies[selected_clients, comm_round].mean()}")

            # prune global model if appropriate
            num_pruned, num_params = get_prune_summary(self.global_models)
            cur_prune_rate = num_pruned / num_params
            if self.client_accuracies[:, comm_round].mean() > self.args.acc_thresh \
                    and cur_prune_rate < self.args.prune_percent:
                prune_fixed_amount(self.global_models, self.args.prune_step, verbose=self.args.prune_verbosity)
                self.global_models = copy_model(self.global_init_model,
                                                self.args.dataset,
                                                self.args.arch,
                                                dict(self.global_models.named_buffers()))
