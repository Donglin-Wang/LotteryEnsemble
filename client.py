import math
from util import train, evaluate, prune_fixed_amount, copy_model, create_model, get_prune_summary

class Client:
    def __init__(self, args, train_loader, test_loader, client_update_method=None):
        
        self.args = args
        self.model = create_model(self.args.dataset, self.args.arch)
        self.init_model = copy_model(self.model, self.args.dataset, self.args.arch)
        self.client_update_method = client_update_method
        self.test_loader = test_loader
        self.train_loader = train_loader
        
        assert self.model, "Something went wrong and the model cannot be initialized"
        
    def client_update(self, global_model, global_init_weight):
        if self.client_update_method:
            return self.client_update_method(self, global_model, global_init_weight)
        else:
            return self.default_client_update_method(global_model, global_init_weight)
        
    def default_client_update_method(self, global_model, global_init_model):
        
        self.model = global_model
        
        num_pruned, num_params = get_prune_summary(self.model)
        cur_prune_rate = num_pruned / num_params
        prune_step = math.floor(num_params * self.args.prune_percent)
        
       
        score = evaluate(self.model, self.test_loader)
        
        if score['Accuracy'][0] > self.args.acc_thresh and cur_prune_rate < self.args.prune_percent:
            prune_fixed_amount(self.model, prune_step)
        
        for i in range(self.args.client_epoch):
            print(f'Epoch {i+1}')
            train(self.model, self.train_loader, lr=self.args.lr)
            
        
        return copy_model(self.model, self.args.dataset, self.args.arch)

# This is a dummmy test to make sure that Client is running properly
if __name__ == "__main__":
    from datasource import get_data
    from archs.mnist.mlp import MLP
    
    
    # Creating an empty object to which we can add any attributes
    args = type('', (), {})()
    
    args.dataset = 'mnist'
    args.arch = 'mlp'
    args.lr = 0.001
    args.client_epoch = 3
    args.prune_type = 'reinit'
    args.prune_percent = 0.45
    args.batch_size = 4
    
    global_model = MLP()
    global_init_model = copy_model(global_model, args.dataset, args.arch)
    
    client_loaders, test_loader = get_data(10, 'mnist', mode='iid', batch_size=args.batch_size)
    
    client = Client(args, client_loaders[0], test_loader)
    for i in range(2):
        global_model = client.client_update(global_model, global_init_model)
    # client.train(global_model, 5)
    
    
    
    
