import math
from util import train, evaluate, prune_fixed_amount, copy_model, \
                 create_model, get_prune_summary, log_obj
import numpy as np
import torch 
torch.manual_seed(0)
np.random.seed(0)
from multiprocessing import Pool

class Client:
    def __init__(self, 
                 args, 
                 train_loader, 
                 test_loader,
                 client_id=None):
        self.args = args
        print("Creating model for client "+ str(client_id))
        self.model = None
        #self.model = create_model(self.args.dataset, self.args.arch)
        print("Copying model for client "+ str(client_id))
        #self.init_model = copy_model(self.model, self.args.dataset, self.args.arch)
        print("Done Copying model for client "+ str(client_id))
        self.test_loader = test_loader
        self.train_loader = train_loader
        self.client_id = client_id
        self.elapsed_comm_rounds = 0
        self.accuracies = np.zeros((args.comm_rounds, self.args.client_epoch))
        self.losses = np.zeros((args.comm_rounds, self.args.client_epoch))
        self.prune_rates = np.zeros(args.comm_rounds)
        self.named_buffers = None
        self.last_client_acc = 0
        self.state_dict = None
       # assert self.model, "Something went wrong and the model cannot be initialized"

        # This is a sanity check that we're getting proper data. Once we are confident about this, we can delete this.
        # train_classes =  self.get_class_counts('train')
        # test_classes  =  self.get_class_counts('test')
        # assert len(train_classes.keys()) == 2,\
        #     f'Client {self.client_id} should have 2 classes in train set but has {len(train_classes.keys())}.'
        # assert len(test_classes.keys()) == 2,\
        #     f'Client {self.client_id} should have 2 classes in test set but has {len(test.keys())}.'
        # assert set(train_classes.keys()) == set(test_classes.keys()),\
        #     f'Client {self.client_id} has different keys for train ({train_classes.keys()}) and test ({test_classes.keys()}).'


    def client_update(self, global_model, global_init_model, round_index):
        self.elapsed_comm_rounds += 1
        print(f'***** Client #{self.client_id} *****', flush=True)
        if self.model is None:
            self.model = global_model
        if self.named_buffers is None:
            self.named_buffers = dict(self.model.named_buffers())

        potential_model = copy_model(global_model,
                                self.args.dataset,
                                self.args.arch,
                                self.named_buffers)
        
        num_pruned, num_params = get_prune_summary(potential_model)
        cur_prune_rate = num_pruned / num_params
        #prune_step = math.floor(num_params * self.args.prune_step)
        
        eval_score = evaluate(potential_model,
                         self.test_loader,
                         verbose=self.args.test_verbosity)

        print(f"previous client acc: {self.last_client_acc} current client acc: {eval_score['Accuracy'][-1]}")
        eval_score = eval_score['Accuracy'][-1]
        if eval_score > self.last_client_acc:
            del self.model
            self.model = potential_model
            self.last_client_acc = eval_score
        else:
            del potential_model
        
        if self.last_client_acc > self.args.acc_thresh and cur_prune_rate < self.args.prune_percent:
            # I'm adding 0.001 just to ensure we go clear the target prune_percent. This may not be needed
            prune_fraction = min(self.args.prune_step, 0.001 + self.args.prune_percent - cur_prune_rate)
            prune_fixed_amount(self.model, 
                               prune_fraction,
                               verbose=self.args.prune_verbosity, glob=True)
            self.named_buffers = dict(self.model.named_buffers())
            del self.model
            self.model = copy_model(global_init_model,
                                    self.args.dataset,
                                    self.args.arch,
                                    self.named_buffers)
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
        self.state_dict = self.model.state_dict()


        #return copy_model(self.model, self.args.dataset, self.args.arch)

    def evaluate(self):
        eval_score = evaluate(self.model,
                              self.test_loader,
                              verbose=self.args.test_verbosity)
        self.last_client_acc = eval_score['Accuracy'][-1]
        return self.last_client_acc

    def clear_model(self):
        del self.model
        self.model = None

    def get_mask(self):
        result = np.array(())
        for k, v in self.state_dict:
            if 'weight_mask' in k:
                result = np.append(result, [v.data.numpy().reshape(-1)])
        return np.array(result)

    def get_class_counts(self, dataset):
        if dataset == 'train':
            ds = self.train_loader
        elif dataset == 'test':
            ds = self.test_loader
        else:
            raise Error('get_class_counts() - invalid value for parameter dataset: ', dataset)

        class_counts = {}
        for batch in ds:
            for label in batch[1]:
                if label.item() not in class_counts:
                    class_counts[label.item()] = 0
                class_counts[label.item()] += 1
        return class_counts
