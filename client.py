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
        self.log_path = f'./log/clients/client{self.client_id}/'
        
        assert self.model, "Something went wrong and the model cannot be initialized"
        
    def client_update(self, global_model, global_init_weight):
        if self.client_update_method:
            return self.client_update_method(self, global_model, global_init_weight)
        else:
            return self.default_client_update_method(global_model, global_init_weight)
        
    def default_client_update_method(self, global_model, global_init_model):
        print(f'***** Client #{self.client_id} *****', flush=True)
        self.model = copy_model(global_model,
                                self.args.dataset,
                                self.args.arch)
        
        num_pruned, num_params = get_prune_summary(self.model)
        cur_prune_rate = num_pruned / num_params
        prune_step = math.floor(num_params * self.args.prune_step)
       
       
        score = evaluate(self.model, 
                         self.test_loader,
                         verbose=self.args.test_verbosity)
        
        if score['Accuracy'][0] > self.args.acc_thresh and cur_prune_rate < self.args.prune_percent:
            prune_fixed_amount(self.model, 
                               prune_step,
                               verbose=self.args.prune_verbosity)
        
        for i in range(self.args.client_epoch):
            
            print(f'Epoch {i+1}')
            train(self.model, 
                  self.train_loader, 
                  lr=self.args.lr,
                  verbose=self.args.train_verbosity)
            epoch_path = self.log_path + f'client_model_epoch{i}.torch'
            log_obj(epoch_path, self.model)
            
        
        return copy_model(self.model, self.args.dataset, self.args.arch)
