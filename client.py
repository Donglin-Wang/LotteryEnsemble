import math
from util import train, evaluate, prune_fixed_amount, copy_model, \
                 create_model, get_prune_summary, log_obj
import numpy as np
import torch 
torch.manual_seed(0)
np.random.seed(0)

class Client:
    def __init__(self, 
                 args, 
                 train_loader, 
                 test_loader,
                 client_id=None):
        self.args = args
        self.model = create_model(self.args.dataset, self.args.arch)
        self.init_model = copy_model(self.model, self.args.dataset, self.args.arch)
        self.test_loader = test_loader
        self.train_loader = train_loader
        self.client_id = client_id
        self.elapsed_comm_rounds = 0
        self.accuracies = np.zeros((args.comm_rounds, self.args.client_epoch))
        self.losses = np.zeros((args.comm_rounds, self.args.client_epoch))
        self.prune_rates = np.zeros(args.comm_rounds)
        assert self.model, "Something went wrong and the model cannot be initialized"


    def client_update(self, global_model, global_init_model, round_index):
        self.elapsed_comm_rounds += 1
        print(f'***** Client #{self.client_id} *****', flush=True)
        self.model = copy_model(global_model,
                                self.args.dataset,
                                self.args.arch,
                                dict(self.model.named_buffers()))
        
        num_pruned, num_params = get_prune_summary(self.model)
        cur_prune_rate = num_pruned / num_params
        #prune_step = math.floor(num_params * self.args.prune_step)
        
        eval_score = evaluate(self.model, 
                         self.test_loader,
                         verbose=self.args.test_verbosity)
        
        if eval_score['Accuracy'][0] > self.args.acc_thresh and cur_prune_rate < self.args.prune_percent:
            prune_fixed_amount(self.model, 
                               self.args.prune_step,
                               verbose=self.args.prune_verbosity, glob=True)
            self.model = copy_model(global_init_model,
                                    self.args.dataset,
                                    self.args.arch,
                                    dict(self.model.named_buffers()))
        losses = []
        accuracies = []
        for i in range(self.args.client_epoch):
            train_score = train(round_index, self.client_id, i, self.model,
                  self.train_loader, 
                  lr=self.args.lr,
                  verbose=self.args.train_verbosity)
           
            losses.append(train_score['Loss'][-1].data.item())
            accuracies.append(train_score['Accuracy'][-1])
           
            
        mask_log_path = f'{self.args.log_folder}/round{round_index}/c{self.client_id}.mask'
        client_mask = dict(self.model.named_buffers())
        log_obj(mask_log_path, client_mask)

        num_pruned, num_params = get_prune_summary(self.model)
        cur_prune_rate = num_pruned / num_params
        prune_step = math.floor(num_params * self.args.prune_step)
        print(f"num_pruned {num_pruned}, num_params {num_params}, cur_prune_rate {cur_prune_rate}, prune_step: {prune_step}")


        self.losses[round_index:] = np.array(losses)
        self.accuracies[round_index:] = np.array(accuracies)
        self.prune_rates[round_index:] = cur_prune_rate


        return copy_model(self.model, self.args.dataset, self.args.arch)

    def evaluate(self):
        eval_score = evaluate(self.model,
                              self.test_loader,
                              verbose=self.args.test_verbosity)
        return eval_score['Accuracy'][-1]

