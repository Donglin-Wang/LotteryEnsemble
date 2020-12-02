import numpy as np
import torch
torch.manual_seed(0)
np.random.seed(0)
from util import average_weights, create_model, copy_model, log_obj, evaluate, fed_avg, lottery_fl_v2, lottery_fl_v3


class Server():

    def __init__(self, args, clients, test_loader=None):

        self.args = args
        self.comm_rounds = args.comm_rounds
        self.num_clients = args.num_clients
        self.frac = args.frac
        self.clients = clients
        self.client_data_num = []
        self.elapsed_comm_rounds = 0
        self.accuracies = np.zeros(args.comm_rounds)
        self.client_accuracies = np.zeros((self.args.num_clients, self.args.comm_rounds))
        self.selected_client_tally = np.zeros((self.args.comm_rounds, self.args.num_clients))
        self.test_loader = test_loader

        for client in self.clients:
            self.client_data_num.append(len(client.train_loader))
        self.client_data_num = np.array(self.client_data_num)

        # The extra 1 entry in client_models and global_models are used to store
        # the results after last communication round
        self.client_models = np.zeros((self.comm_rounds + 1, self.num_clients), dtype='object')
        self.global_models = None  # np.zeros((self.comm_rounds + 1,), dtype='object')

        self.init_model = create_model(args.dataset, args.arch)

        self.global_models = self.init_model
        self.global_init_model = copy_model(self.init_model, args.dataset, args.arch)

        assert self.num_clients == len(clients), "Number of client objects does not match command line input"

    def server_update(self):
        self.elapsed_comm_rounds += 1
        # Recording the update and storing them in record
        self.global_models.train()
        for i in range(0, self.comm_rounds):
            update_or_not = [0] * self.num_clients
            # Randomly select a fraction of users to update
            num_selected_clients = max(int(self.frac * self.num_clients), 1)
            idx_list = np.random.choice(range(self.num_clients),
                                        num_selected_clients,
                                        replace=False)
            for idx in idx_list:
                update_or_not[idx] = 1

            print('-------------------------------------', flush=True)
            print(f'Communication Round #{i}', flush=True)
            print('-------------------------------------', flush=True)
            for j in range(len(update_or_not)):

                if update_or_not[j]:
                    if i == 0:
                        prev_model_acc = 0
                    else:
                        prev_model_acc = self.client_accuracies[j][i - 1]

                    if self.args.avg_logic == "standalone":
                        self.clients[j].client_update(self.clients[j].model, self.global_init_model, i, prev_model_acc)
                    else:
                        self.clients[j].client_update(self.global_models, self.global_init_model, i, prev_model_acc)
                else:
                    pass
                    # copy_model(self.clients[j].model, self.args.dataset, self.args.arch)

            models = []
            self.selected_client_tally[i, idx_list] += 1
            for m in self.clients[idx_list]:
                models.append(m.model)
            if self.args.avg_logic == "fed_avg":
                self.global_models = fed_avg(list(map(lambda x: x.model, self.clients)), self.args.dataset,
                                             self.args.arch,
                                             self.client_data_num)
            elif self.args.avg_logic == 'lottery_fl_v2':
                self.global_models = lottery_fl_v2(self.global_models, models, self.args.dataset,
                                                          self.args.arch,
                                                          self.client_data_num[idx_list])
            elif self.args.avg_logic == 'lottery_fl_v3':
                self.global_models = lottery_fl_v3(self.global_models, models, self.args.dataset,
                                                          self.args.arch,
                                                          self.client_data_num[idx_list])
            elif self.args.avg_logic == "standalone":
                pass #no averaging in the server
            else:
                self.global_models = average_weights(models, self.args.dataset,
                                                          self.args.arch,
                                                          self.client_data_num[idx_list])
            del models

            eval_score = evaluate(self.global_models,
                                  self.test_loader,
                                  verbose=self.args.test_verbosity)
            print(f"Server accuracies over the batch + avg at the end: {eval_score['Accuracy']}")
            self.accuracies[i] = eval_score['Accuracy'][-1]

            for k, m in enumerate(self.clients):
                if k in idx_list:
                    self.client_accuracies[k][i] = m.evaluate()
                else:
                    if i == 0:
                        pass
                    else:
                        self.client_accuracies[k][i] = self.client_accuracies[k][i - 1]
            print(f"Mean client accs: {self.client_accuracies.mean(axis=0)[i]}")
