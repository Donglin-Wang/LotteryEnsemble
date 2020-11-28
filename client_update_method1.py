import math
import torch
import torch.nn as nn
from util import copy_model, evaluate, get_prune_summary, train, prune_fixed_amount


# User defined update method
def client_update_method1(client_self, global_model, global_init_model):
    print(f'***** Client #{client_self.client_id} *****', flush=True)
    # Checking if the client object has been properly initialized
    assert isinstance(client_self.model, nn.Module), "A model must be a PyTorch module"
    assert 0 <= client_self.args.prune_percent <= 1, "The prune percentage must be between 0 and 1"
    assert client_self.args.client_epoch, '"args" must contain a "client_epoch" field'
    assert client_self.test_loader, "test_loader field does not exist. Check if the client is initialized correctly"
    assert client_self.train_loader, "train_loader field does not exist. Check if the client is initialized correctly"
    assert isinstance(client_self.train_loader, torch.utils.data.DataLoader), "train_loader must be a DataLoader type"
    assert isinstance(client_self.test_loader, torch.utils.data.DataLoader), "test_loader must be a DataLoader type"

    client_self.model = copy_model(global_model,
                                   client_self.args.dataset,
                                   client_self.args.arch)

    num_pruned, num_params = get_prune_summary(client_self.model)
    cur_prune_rate = num_pruned / num_params
    prune_step = math.floor(num_params * client_self.args.prune_step)

    for i in range(client_self.args.client_epoch):
        print(f'Epoch {i + 1}')
        train(client_self.model,
              client_self.train_loader,
              lr=client_self.args.lr,
              verbose=client_self.args.train_verbosity)

    score = evaluate(client_self.model,
                     client_self.test_loader,
                     verbose=client_self.args.test_verbosity)

    if score['Accuracy'][0] > client_self.args.acc_thresh and cur_prune_rate < client_self.args.prune_percent:
        prune_fixed_amount(client_self.model,
                           prune_step,
                           verbose=client_self.args.prune_verbosity)

    return copy_model(client_self.model, client_self.args.dataset, client_self.args.arch)
