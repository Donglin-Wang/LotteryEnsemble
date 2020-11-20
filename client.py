import math
from util import train, evaluate, prune_fixed_amount, copy_model, \
                 create_model, get_prune_summary, log_obj

class Client:
    def __init__(self, 
                 args, 
                 train_loader, 
                 test_loader, 
                 client_update_method=None, 
                 client_id=None):
        
        self.args = args
        self.model = create_model(self.args.dataset, self.args.arch)
        self.init_model = copy_model(self.model, self.args.dataset, self.args.arch)
        self.client_update_method = client_update_method
        self.test_loader = test_loader
        self.train_loader = train_loader
        self.client_id = client_id
        self.elapsed_comm_rounds = 0
        
        
        assert self.model, "Something went wrong and the model cannot be initialized"
        
    def client_update(self, global_model, global_init_weight, round_num=None):
        if self.client_update_method:
            return self.client_update_method(self, global_model, global_init_weight)
        else:
            return self.default_client_update_method(global_model, 
                                                     global_init_weight,
                                                     round_num=round_num)
        
    def default_client_update_method(self, 
                                     global_model, 
                                     global_init_model,
                                     round_num = None):
        self.elapsed_comm_rounds += 1
        
        log_round_idx = round_num if round_num else self.elapsed_comm_rounds
        
        print(f'***** Client #{self.client_id} *****', flush=True)
        self.model = copy_model(global_model,
                                self.args.dataset,
                                self.args.arch,
                                dict(self.model.named_buffers()))
        
        num_pruned, num_params = get_prune_summary(self.model)
        cur_prune_rate = num_pruned / num_params
        prune_step = math.floor(num_params * self.args.prune_step)
        
        # eval_score = evaluate(self.model, 
        #                  self.test_loader,
        #                  verbose=self.args.test_verbosity)
        
        # eval_log_path = f'./log/clients/client{self.client_id}/'\
        #                 f'round{log_round_idx}/'\
        #                 f'eval_score_round{self.elapsed_comm_rounds}.pickle'
        # log_obj(eval_log_path, eval_score)
        
        
        # if eval_score['Accuracy'][0] > self.args.acc_thresh and cur_prune_rate < self.args.prune_percent:
        #     prune_fixed_amount(self.model, 
        #                        prune_step,
        #                        verbose=self.args.prune_verbosity)
        #     self.model = copy_model(global_init_model,
        #                             self.args.dataset,
        #                             self.args.arch,
        #                             source_buff=dict(self.model.named_buffers()))
        
        for i in range(self.args.client_epoch):
            print(f'Epoch {i+1}')
            train_score = train(self.model, 
                  self.train_loader, 
                  lr=self.args.lr,
                  verbose=self.args.train_verbosity)
            
            train_log_path = f'./log/clients/client{self.client_id}'\
                             f'/round{log_round_idx}/'
            epoch_score_path = train_log_path + f'client_train_score_epoch{i}.pickle'
            log_obj(epoch_score_path, train_score)
            
        
        eval_score = evaluate(self.model, 
                         self.test_loader,
                         verbose=self.args.test_verbosity)
        
        eval_log_path = f'./log/clients/client{self.client_id}/'\
                        f'round{log_round_idx}/'\
                        f'eval_score_round{self.elapsed_comm_rounds}.pickle'
        log_obj(eval_log_path, eval_score)
        
        
        if eval_score['Accuracy'][0] > self.args.acc_thresh and cur_prune_rate < self.args.prune_percent:
            prune_fixed_amount(self.model, 
                               prune_step,
                               verbose=self.args.prune_verbosity)
            # self.model = copy_model(global_init_model,
            #                         self.args.dataset,
            #                         self.args.arch,
            #                         source_buff=dict(self.model.named_buffers()))
            
        mask_log_path = f'./log/clients/client{self.client_id}'\
                        f'/round{log_round_idx}/'\
                        f'client_mask.pickle'
        client_mask = dict(self.model.named_buffers())
        log_obj(mask_log_path, client_mask)
            
        
        return copy_model(self.model, self.args.dataset, self.args.arch)
