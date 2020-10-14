class Client:
    def __init__(self, args, model_type):
        
        self.epoch = args.client_train_iter
        self.prune_iter = args.client_prune_iter
        self.prune_perc = args.cleint_prune_perc
        
        # TODO: implement the following initilizations
        # self.rtarget = ???  (See LotterFL Page 4 Algo 1) 
        # self.acc_target = ??? (See LotterFL Page 4 Algo 1)
        # self.model = ???
        # self.mask = ???
        
    def client_update_loop():
        return
    
    def train():
        return
    
    def test():
        return
    
    def init_mask():
        return
    
    def prune():
        return
    
    def set_to_init_weights():
        return
    