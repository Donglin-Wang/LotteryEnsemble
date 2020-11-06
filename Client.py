# Core dependencies
import math
import copy


from util import train, evaluate, prune_fixed_amount

class Client:
    def __init__(self, args, train_loader, test_loader, client_update_method=None):
        
        # Initializing the model
        self.model = None
        # Initializing client update method
        self.client_update_method = client_update_method
        
        if args.dataset == "mnist": from archs.mnist import mlp
        if args.arch == 'mlp': self.model = mlp.MLP()
        else:
            print("You did not enter the name of a supported architecture for this dataset")
            print("Supported datasets: {}, {}".format('"CIFAR10"', '"MNIST"'))
            exit()
        
        assert self.model, "Something went wrong and the model cannot be initialized"
        self.init_model_state = copy.deepcopy(self.model.state_dict())
        
        # Getting the settings for the model
        self.lr = args.lr
        self.val_precent = 0.15
        self.batch_size = args.batch_size
        self.client_epoch = args.client_epoch
        self.prune_iterations = args.prune_iterations
        self.prune_percent = args.prune_percent
        # self.prune_amount = args.prune_amount
        
        # Splitting the data
        self.test_loader = test_loader
        self.train_loader = train_loader
        
    def client_update(self, global_model, global_init_weight):
        if self.client_update_method:
            return self.client_update_method(self, global_model, global_init_weight)
        else:
            return self.default_client_update_method(global_model, global_init_weight)
        
    def default_client_update_method(self, global_model, global_init_weight):
        
        self.model = copy.deepcopy(global_model)
        
        num_params = sum([p.numel() for p in self.model.parameters() if p.requires_grad])
        prune_step = math.floor(num_params * self.prune_percent / self.prune_iterations)
        
        score = evaluate(self.model, self.test_loader)
        
        if score['Accuracy'][0] > 0.5:
            prune_fixed_amount(self.model, prune_step)
        
        train(self.model, self.train_loader, self.client_epoch)
        
        return copy.deepcopy(self.model.state_dict())
   
if __name__ == "__main__":
    from DataSource import get_data
    from archs.mnist.mlp import MLP
    
    
    # Creating an empty object to which we can add any attributes
    args = type('', (), {})()
    
    args.dataset = 'mnist'
    args.arch = 'mlp'
    args.lr = 0.001
    args.client_epoch = 3
    args.prune_iterations = 2
    args.prune_type = 'reinit'
    args.prune_percent = 0.45
    args.batch_size = 4
    
    global_model = MLP()
    global_init_state = copy.deepcopy(global_model.state_dict())
    global_state = copy.deepcopy(global_model.state_dict())
    
    client_loaders, test_loader = get_data(10, 'MNIST', mode='iid', batch_size=args.batch_size)
    
    client = Client(args, client_loaders[0], test_loader)
    for i in range(2):
        global_state = client.client_update(global_state, global_init_state)
    # client.train(global_model, 5)
    
    
    
    