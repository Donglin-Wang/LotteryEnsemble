import copy
import numpy as np
import torch
import torchvision as tv
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as tf

class Client:
    def __init__(self, args, data_loader):
        
        if args.dataset == "mnist": from archs.mnist import mlp
        else:
            print("You did not enter the name of a supported dataset")
            print("Supported datasets: {}, {}".format('"CIFAR10"', '"MNIST"'))
            exit()
        
        if args.arch == 'mlp': self.model = mlp.MLP()
        else:
            print("You did not enter the name of a supported architecture for this dataset")
            print("Supported datasets: {}, {}".format('"CIFAR10"', '"MNIST"'))
            exit()
        
        assert self.model, "Something went wrong and the model cannot be initialized"
        
        self.log_freq = args.log_freq
        self.init_model_state = copy.deepcopy(self.model.state_dict())
        self.train_loader = data_loader
        self.client_epoch = args.client_epoch
        self.prune_iterations = args.prune_iterations
        self.prune_percent = args.prune_percent
        self.mask = self.init_mask()
        
        # TODO: implement the following initilizations
        # self.rtarget = ???  (See LotterFL Page 4 Algo 1) 
        # self.acc_target = ??? (See LotterFL Page 4 Algo 1)
        
    def client_update_loop():
        return
    
    def trainModel(self):
        loss_function = nn.CrossEntropyLoss()
        opt = optim.Adam(self.model.parameters(), lr=self.lr)
        error = []
        print("Model is starting to train...")
        for epoch in range(self.client_epoch):
            total_loss = 0.0
            for i, data in enumerate(self.train_loader, 0):
                # Getting inputs and predictions
                inputs, labels = data
                opt.zero_grad()
                pred = self.model(inputs)
                # Gradient descent
                loss = loss_function(pred, labels)
                loss.backward()
                opt.step()
                # Loss Calculation
                total_loss += loss.item()
                if i % self.log_freq == self.log_freq - 1:
                   error.append(total_loss / self.log_freq)
                   total_loss = 0.0
        print("Finished Training")
        return error
    
    def test():
        return
    
    def init_mask(self):
        
        assert self.model, "init_mask() is called before the model is initialized"
        
        layer = 0
        for name, param in self.model.named_parameters(): 
            if 'weight' in name:
                layer = layer + 1
        self.mask = [None]* layer
        
        layer = 0
        for name, param in self.model.named_parameters(): 
            if 'weight' in name:
                tensor = param.data.cpu().numpy()
                self.mask[layer] = np.ones_like(tensor)
                layer = layer + 1
        layer = 0
    
            
    def prune():
        return
    
    def set_to_init_weights():
        return
    def prun_by_percent(self, percent):
        
        assert self.model, "prune_by_percent() is called before the model is initialized"
        
        # Calculate percentile value
        layer = 0
        for name, param in self.model.named_parameters():

            # We do not prune bias term
            if 'weight' in name:
                tensor = param.data.cpu().numpy()
                alive = tensor[np.nonzero(tensor)] # flattened array of nonzero values
                percentile_value = np.percentile(abs(alive), percent)

                # Convert Tensors to numpy and calculate
                weight_dev = param.device
                new_mask = np.where(abs(tensor) < percentile_value, 0, self.mask[layer])
                
                # Apply new weight and mask
                param.data = torch.from_numpy(tensor * new_mask).to(weight_dev)
                self.mask[layer] = new_mask
                layer += 1
        layer = 0

if __name__ == "__main__":
    from DataSource import get_data
    # Creating an empty object to which we can add any attributes
    args = type('', (), {})()
    
    args.dataset = 'mnist'
    args.arch = 'mlp'
    args.client_epoch = 100
    args.prune_iterations = 10
    args.prune_type = 'reinit'
    args.prune_percent = 10
    
    client_loaders, test_loader = get_data(1, 'mnist')
    client = Client(args, client_loaders[0])