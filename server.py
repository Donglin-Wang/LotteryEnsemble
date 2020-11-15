import numpy as np
from util import average_weights, create_model, copy_model

class Server():
    
    def __init__(self, args, clients, server_update_method=None):
        
        self.args = args
        self.comm_rounds = args.comm_rounds
        self.num_clients = args.num_clients
        self.frac = args.frac
        self.clients = clients
        self.server_update_method = server_update_method
        self.client_data_num = []
        
        for client in self.clients:
            self.client_data_num.append(len(client.train_loader))
        
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
       
        
        # Recording the update and storing them in record
        for i in range(1, self.comm_rounds+1):
            update_or_not = [0] * self.num_clients
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
                    self.client_models[i][j] = self.clients[j].client_update(self.global_models[i-1], self.global_init_model)
                else:
                    self.client_models[i][j] = copy_model(self.clients[j].model, self.args.dataset, self.args.arch)
            
            models = self.client_models[i][idx_list]
            self.global_models[i] = average_weights(models, 
                                                    self.args.dataset, 
                                                    self.args.arch,
                                                    self.client_data_num)

