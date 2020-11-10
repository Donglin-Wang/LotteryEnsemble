from util import average_weights, create_model, copy_model
import copy
import numpy as np
from archs.mnist.mlp import MLP

class Server():
    
    def __init__(self, args, clients, server_update_method=None):
        
        self.args = args
        self.comm_rounds = args.comm_rounds
        self.num_clients = args.num_clients
        self.frac = args.frac
        self.clients = clients
        self.server_update_method = server_update_method
        
        # The extra 1 entry in client_models and global_models are used to store
        # the results after last communication round
        self.client_models = np.zeros((self.comm_rounds + 1, self.num_clients), dtype='object')
        self.global_models = np.zeros((self.comm_rounds + 1,), dtype='object')
        
        init_model = create_model(args.dataset, args.arch)
    
        self.global_models[0] = init_model
        self.global_init_model = copy_model(init_model, args.dataset, args.arch)
        
        assert self.num_clients == len(clients),  "Number of client objects does not match command line input"
        
    def server_update(self):
        
        if self.server_update_method:
            self.server_update_method(self)
        else:
            self.default_server_update()
            
    def default_server_update(self):
        
        # For each client, 0 means no update and 1 means update
        update_or_not = [0] * self.num_clients
        
        # Recording the update and storing them in record
        for i in range(1, self.comm_rounds):
            
            # Randomly select a fraction of users to update
            num_selected_clients = max(int(self.frac * self.num_clients), 1)
            idx_list = np.random.choice(range(self.num_clients), num_selected_clients)
            for idx in idx_list:
                update_or_not[idx] = 1
            
            print('-------------------------------------', flush=True)
            print(f'Communication Round #{i}', flush=True)
            print('-------------------------------------', flush=True)
            for j in range(len(update_or_not)):
                
                if update_or_not[j]:
                    print(f'***** Client #{j+1} *****', flush=True)
                    self.client_models[i][j] = self.clients[j].client_update(self.global_models[i-1], self.global_init_model)
                else:
                    self.client_models[i][j] = copy_model(self.clients[j].model, self.args.dataset, self.args.arch)
            
            models = self.client_models[i][idx_list]
            self.global_models[i] = average_weights(models)

# This is a dummy test to see if the server works
if __name__ == '__main__':
    from client import Client
    from datasource import get_data

    # Creating an empty object to which we can add any attributes
    args = type('', (), {})()
    
    args.dataset = 'mnist'
    args.arch = 'mlp'
    args.lr = 0.001
    args.client_epoch = 2
    args.prune_type = 'reinit'
    args.prune_percent = 0.15
    args.acc_thresh = 0.5
    args.batch_size = 4
    
    args.frac = 0.3
    args.comm_rounds = 2
    args.num_clients = 10
     
    global_model = MLP()
    global_init_model = copy.deepcopy(global_model.state_dict())
    global_state = copy.deepcopy(global_model.state_dict())
    
    client_loaders, test_loader = get_data(args.num_clients, 'mnist', mode='iid', batch_size=args.batch_size)
    
    clients = [Client(args, client_loaders[i], test_loader) for i in range(args.num_clients)]
    
    server = Server(args, clients)
    
    server.server_update()
    
    # client = Client(args, client_loaders[0], test_loader)
    # client.client_update(global_state, global_init_model)
    # client.train(0, 5)
    
