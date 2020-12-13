import os
import sys
import errno
import pickle
import math
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics as skmetrics
import numpy as np
from tqdm import tqdm
from tabulate import tabulate
import torch.nn.utils.prune as prune

torch.manual_seed(0)
np.random.seed(0)

def fed_avg(models, dataset, arch, data_nums):
    new_model = create_model(dataset, arch) #copy_model(server_model, dataset, arch, source_buff=dict(server_model.named_buffers()))
    num_models = len(models)
    num_data_total = sum(data_nums)
    data_nums = data_nums / num_data_total
    with torch.no_grad():
        # Getting all the weights and masks from original models
        weights, masks = [], []
        for i in range(num_models):
            weights.append(dict(models[i].named_parameters()))
            masks.append(dict(models[i].named_buffers()))

        for name, param in new_model.named_parameters():
            param.data.copy_(torch.zeros_like(param.data))
        # Averaging weights
        for name, param in new_model.named_parameters():
            for i in range(num_models):
                weighted_param = torch.mul(weights[i][name], data_nums[i])
                param.data.copy_(param.data + weighted_param)
        avg = torch.div(param.data, num_models)
        param.data.copy_(avg)
    return new_model

def lottery_fl_v2(server_model, models, dataset, arch, data_nums):
    new_model = create_model(dataset, arch) #copy_model(server_model, dataset, arch, source_buff=dict(server_model.named_buffers()))
    num_models = len(models)
    num_data_total = sum(data_nums)
    data_nums = data_nums / num_data_total
    with torch.no_grad():
        # Get server weights 
        server_weights = dict(server_model.named_parameters())

        # Getting all the weights and masks from original models
        weights, masks = [], []
        for i in range(num_models):
            weights.append(dict(models[i].named_parameters()))
            masks.append(dict(models[i].named_buffers()))

        for name, param in new_model.named_parameters():
            param.data.copy_(torch.zeros_like(param.data))
        # Averaging weights
        for name, param in new_model.named_parameters():
            for i in range(num_models):
                #parameters_to_prune, num_global_weights, _ = get_prune_params(models[i])

                model_masks = masks[i]

                try:
                    layer_mask = model_masks[name.strip("_orig") + "_mask"]
                    weights[i][name] *= layer_mask
                    weights[i][name] = np.where(weights[i][name] != 0, weights[i][name], server_weights[name])
                except Exception as e:
                    #print("exceptions")
                    #print(e)
                    pass
                weighted_param = weights[i][name]
                #weighted_param = torch.mul(weights[i][name], data_nums[i])
                param.data.copy_(param.data + weighted_param)
            avg = torch.div(param.data, num_models)
            param.data.copy_(avg)
    return new_model

def lottery_fl_v3(server_model, models, dataset, arch, data_nums):
    new_model = create_model(dataset, arch) #copy_model(server_model, dataset, arch, source_buff=dict(server_model.named_buffers()))
    num_models = len(models)
    num_data_total = sum(data_nums)
    with torch.no_grad():
        # Getting all the weights and masks from original models
        weights, masks = [], []
        for i in range(num_models):
            new_c_model = copy_model(models[i], dataset, arch)
            parameters_to_prune, _, _ = get_prune_params(new_c_model)
            for m, n in parameters_to_prune:
                prune.remove(m, n)
            weights.append(dict(new_c_model.named_parameters()))

        for name, param in new_model.named_parameters():
            param.data.copy_(torch.zeros_like(param.data))
        # Averaging weights
        for name, param in new_model.named_parameters():
            for i in range(num_models):
                weighted_param = weights[i][name.strip("_orig")] #torch.mul(weights[i][name], data_nums[i])
                param.data.copy_(param.data + weighted_param)
            avg = torch.div(param.data, num_models)
            param.data.copy_(avg)
    return new_model

def average_weights(models, dataset, arch, data_nums):
    new_model = create_model(dataset, arch) #copy_model(server_model, dataset, arch, source_buff=dict(server_model.named_buffers()))
    num_models = len(models)
    num_data_total = sum(data_nums)
    with torch.no_grad():
        # Getting all the weights and masks from original models
        weights, masks = [], []
        for i in range(num_models):
            weights.append(dict(models[i].named_parameters()))
            masks.append(dict(models[i].named_buffers()))

        for name, param in new_model.named_parameters():
            param.data.copy_(torch.zeros_like(param.data))
        # Averaging weights
        for name, param in new_model.named_parameters():
            for i in range(num_models):
                weighted_param = weights[i][name] #torch.mul(weights[i][name], data_nums[i])
                param.data.copy_(param.data + weighted_param)
            avg = torch.div(param.data, num_models)
            param.data.copy_(avg)
    return new_model

def copy_model(model, dataset, arch, source_buff=None):
    new_model = create_model(dataset, arch)
    source_weights = dict(model.named_parameters())
    source_buffers = source_buff if source_buff else dict(model.named_buffers())
    for name, param in new_model.named_parameters():
        param.data.copy_(source_weights[name])
    for name, buffer in new_model.named_buffers():
        buffer.data.copy_(source_buffers[name])
    return new_model

def create_model(dataset_name, model_type):
    
    if dataset_name == "mnist": 
        from archs.mnist import mlp, cnn

    elif dataset_name == "cifar10":
        from archs.cifar10 import mlp, cnn

    else: 
        print("You did not enter the name of a supported architecture for this dataset")
        print("Supported datasets: {}, {}".format('"CIFAR10"', '"MNIST"'))
        exit()
    
    if model_type == 'mlp':
        new_model = mlp.MLP()
        # This pruning call is made so that the model is set up for accepting
        # weights from another pruned model. If this is not done, the weights
        # will be incompatible
        prune_fixed_amount(new_model, 0.0, verbose=False)
        return new_model

    elif model_type == 'cnn':
        new_model = cnn.CNN()
        prune_fixed_amount(new_model, 0, verbose=False)
        return new_model

    else:
        print("You did not enter the name of a supported architecture for this dataset")
        print("Supported models: {}, {}".format('"mlp"', '"cnn"'))
        exit()
    

def train(round,
          client_id,
          epoch,
          model,
          train_loader,
          lr=0.001,
          verbose=True):
    
    loss_function = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=lr)
    num_batch = len(train_loader)
    model.train()
    metric_names = ['Loss',
                    'Accuracy']

    score = {name:[] for name in metric_names}

    # progress_bar = tqdm(enumerate(train_loader),
    #                     total = num_batch,
    #                     file=sys.stdout)
    # Iterating over all mini-batches
    for i, data in enumerate(train_loader):
    
        x, ytrue = data

        yraw = model(x)
        
        loss = loss_function(yraw, ytrue)
        
        model.zero_grad()
        
        loss.backward()
        
        opt.step()
        
        # Truning the raw output of the network into one-hot result
        _, ypred = torch.max(yraw, 1)
       
        score = calculate_metrics(score, ytrue, yraw, ypred)
        
    average_scores = {}
        
    for k, v in score.items():
        average_scores[k] = [sum(v) / len(v)]
        score[k].append(sum(v) / len(v))
    
    if verbose:
        print(f"round={round}, client={client_id}, epoch= {epoch}: ")
        print(tabulate(average_scores, headers='keys', tablefmt='github'))
    
    return score

def evaluate(model, data_loader, verbose=True):
    # Swithicing off gradient calculation to save memory
    torch.no_grad()
    # Switch to eval mode so that layers like Dropout function correctly
    model.eval()
    
    metric_names = ['Loss',
                    'Accuracy']
    
    score = {name:[] for name in metric_names}
    
    num_batch = len(data_loader)
    
    #progress_bar = tqdm(enumerate(data_loader),
                        # total=num_batch,
                        # file=sys.stdout)
    
    for i, (x, ytrue) in enumerate(data_loader):
        
        yraw = model(x)
        
        _, ypred = torch.max(yraw, 1)
        
        score = calculate_metrics(score, ytrue, yraw, ypred)
        
        #progress_bar.set_description('Evaluating')
    
   
        
    for k, v in score.items():
        score[k] = [sum(v) / len(v)]
        
    
    if verbose:
        print('Evaluation Score: ')   
        print(tabulate(score, headers='keys', tablefmt='github'), flush=True)
    model.train()
    torch.enable_grad()
    return score

def prune_fixed_amount(model, amount, verbose=True, glob=True):
    parameters_to_prune, num_global_weights, layers_w_count = get_prune_params(model)

    if glob:
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount = math.floor(amount * num_global_weights))
    else:
        for i, (m, n) in enumerate(parameters_to_prune):
            prune.l1_unstructured(m, name=n, amount = math.floor(amount * layers_w_count[i][1]))

    num_global_zeros, num_layer_zeros, num_layer_weights = 0, 0, 0
    global_prune_percent, layer_prune_percent = 0, 0
    prune_stat = {'Layers': [],
                  'Weight Name': [],
                  'Percent Pruned': [],
                  'Total Pruned': []}
    
    # Pruning is done in-place, thus parameters_to_prune is updated
    for layer, weight_name in parameters_to_prune:
        
        num_layer_zeros = torch.sum(getattr(layer, weight_name) == 0.0).item()
        num_global_zeros += num_layer_zeros
        num_layer_weights = torch.numel(getattr(layer, weight_name))
        layer_prune_percent = num_layer_zeros / num_layer_weights * 100
        prune_stat['Layers'].append(layer.__str__())
        prune_stat['Weight Name'].append(weight_name)
        prune_stat['Percent Pruned'].append(f'{num_layer_zeros} / {num_layer_weights} ({layer_prune_percent:.5f}%)')
        prune_stat['Total Pruned'].append(f'{num_layer_zeros}')
        
    global_prune_percent = num_global_zeros / num_global_weights
    if verbose:
        print('Pruning Summary', flush=True)
        print(tabulate(prune_stat, headers='keys'), flush=True)
        print(f'Percent Pruned Globaly: {global_prune_percent:.2f}', flush=True)

def get_prune_summary(model):
    num_global_zeros = 0
    parameters_to_prune, num_global_weights, _ = get_prune_params(model)

    masks = dict(model.named_buffers())

    for i, (layer, weight_name) in enumerate(parameters_to_prune):
        attr = getattr(layer, weight_name)
        try:
            attr *= masks[list(masks)[i]]
        except Exception as e:
            print(e)

        num_global_zeros += torch.sum(attr == 0.0).item()

    return num_global_zeros, num_global_weights
        
def get_prune_params(model):
    layers = []
    layers_weight_count = []
    
    num_global_weights = 0
    
    modules = list(model.modules())
    
    for layer in modules:
        
        is_sequential = type(layer) == nn.Sequential
        
        is_itself = type(layer) == type(model) if len(modules) > 1 else False
        
        if (not is_sequential) and (not is_itself):
            for name, param in layer.named_parameters():
                
                field_name = name.split('.')[-1]
                
                # This might break if someone does not adhere to the naming
                # convention where weights of a module is stored in a field
                # that has the word 'weight' in it
                
                if 'weight' in field_name and param.requires_grad:
                    
                    if field_name.endswith('_orig'):
                        field_name = field_name[:-5]
                    
                    # Might remove the param.requires_grad condition in the future
                    
                    layers.append((layer, field_name))
                    
                    layer_weight_count = torch.numel(param)
                    
                    layers_weight_count.append((layer, layer_weight_count))
                
                    num_global_weights += layer_weight_count
                    
    return layers, num_global_weights, layers_weight_count


loss = nn.CrossEntropyLoss()
def calculate_metrics(score, ytrue, yraw, ypred):
    global loss
    if 'Loss' in score:
        score['Loss'].append(loss(yraw, ytrue))
    if 'Accuracy' in score:
        score['Accuracy'].append(skmetrics.accuracy_score(ytrue, ypred))
    if 'Balanced Accuracy' in score:
        score['Balanced Accuracy'].append(skmetrics.balanced_accuracy_score(ytrue, ypred))
    if 'Precision Micro' in score:
        score['Precision Micro'].append(skmetrics.precision_score(ytrue, 
                                                                  ypred, 
                                                                  average='micro',
                                                                  zero_division=0))
    if 'Recall Micro' in score:
        score['Recall Micro'].append(skmetrics.recall_score(ytrue, 
                                                            ypred, 
                                                            average='micro',
                                                            zero_division=0))
    if 'Precision Macro' in score:
        score['Precision Macro'].append(skmetrics.precision_score(ytrue, 
                                                                  ypred, 
                                                                  average='macro',
                                                                  zero_division=0))
    if 'Recall Macro' in score:
        score['Recall Macro'].append(skmetrics.recall_score(ytrue, 
                                                            ypred, 
                                                            average='macro',
                                                            zero_division=0))
    
    return score
        
def log_obj(path, obj):
    pass
    # if not os.path.exists(os.path.dirname(path)):
    #     try:
    #         os.makedirs(os.path.dirname(path))
    #     except OSError as exc: # Guard against race condition
    #         if exc.errno != errno.EEXIST:
    #             raise
    #
    # with open(path, 'wb') as file:
    #     if isinstance(obj, nn.Module):
    #         torch.save(obj, file)
    #     else:
    #         pickle.dump(obj, file)

from itertools import chain
from collections import OrderedDict

def train_client_model(acc_thresh, prune_percent, prune_step, prune_verbosity, dataset, arch, lr, train_verbosity, client_epoch, log_folder,
                       round, client_id, state_dict, global_state_dict, train_loader, test_loader, last_client_acc):

    mlp = create_model(dataset, arch)
    prune_fixed_amount(mlp, 0.0, False)
    mlp.load_state_dict(state_dict)



    potential_model = copy_model(mlp,
                                 dataset,
                                 arch, source_buff=dict(mlp.named_buffers()))
    potential_model.load_state_dict(global_state_dict)

    global_model = copy_model(potential_model,
                              dataset,
                              arch)

    num_pruned, num_params = get_prune_summary(potential_model)
    cur_prune_rate = num_pruned / num_params
    #prune_step = math.floor(num_params * self.args.prune_step)

    eval_score = evaluate(potential_model,
                          test_loader,
                          verbose=False)

    print(f"previous client acc: {last_client_acc} current client acc: {eval_score['Accuracy'][-1]}")
    eval_score = eval_score['Accuracy'][-1]
    if eval_score > last_client_acc:
        del mlp
        mlp = potential_model
        last_client_acc = eval_score
    else:
        del potential_model

    if last_client_acc > acc_thresh and cur_prune_rate < prune_percent:
        # I'm adding 0.001 just to ensure we go clear the target prune_percent. This may not be needed
        prune_fraction = min(prune_step, 0.001 + prune_percent - cur_prune_rate)
        prune_fixed_amount(mlp,
                           prune_fraction,
                           verbose=prune_verbosity, glob=True)
        named_buffers = dict(mlp.named_buffers())
        del mlp
        mlp = copy_model(global_model,
                                dataset,
                                arch,
                                named_buffers)
    losses = []
    accuracies = []
    for i in range(client_epoch):
        train_score = train(round, client_id, i, mlp,
                            train_loader,
                            lr=lr,
                            verbose=train_verbosity)

        losses.append(train_score['Loss'][-1].data.item())
        accuracies.append(train_score['Accuracy'][-1])


    mask_log_path = f'{log_folder}/round{round}/c{client_id}.mask'
    client_mask = dict(mlp.named_buffers())
    log_obj(mask_log_path, client_mask)

    num_pruned, num_params = get_prune_summary(mlp)
    cur_prune_rate = num_pruned / num_params
    prune_step = math.floor(num_params * prune_step)
    print(f"num_pruned {num_pruned}, num_params {num_params}, cur_prune_rate {cur_prune_rate}, prune_step: {prune_step}")
    del global_model

    return np.array(losses), np.array(accuracies), cur_prune_rate, mlp.state_dict()



def train_client_model_orig(acc_thresh, prune_percent, prune_step, prune_verbosity, dataset, arch, lr, train_verbosity, client_epoch, log_folder,
                       round, client_id, state_dict, global_state_dict, train_loader, test_loader, val_loader, last_client_acc):

    model = create_model(dataset, arch)
    model.load_state_dict(state_dict)

    global_model = copy_model(model,
                              dataset,
                              arch)
    global_model.load_state_dict(global_state_dict)
    model = copy_model(global_model,dataset, arch, source_buff=dict(model.named_buffers()))




    num_pruned, num_params = get_prune_summary(model)
    cur_prune_rate = num_pruned / num_params
    #prune_step = math.floor(num_params * self.args.prune_step)

    eval_score = evaluate(model,
                          val_loader,
                          verbose=False)

    #print(f"previous client acc: {last_client_acc} current client acc: {eval_score['Accuracy'][-1]}")
    eval_score = eval_score['Accuracy'][-1]


    if eval_score > acc_thresh and cur_prune_rate < prune_percent:
        # I'm adding 0.001 just to ensure we go clear the target prune_percent. This may not be needed
        prune_fraction = min(prune_step, 0.001 + prune_percent - cur_prune_rate)
        prune_fixed_amount(model,
                           prune_fraction,
                           verbose=prune_verbosity, glob=True)
        model = copy_model(global_model,
                         dataset,
                         arch,
                         dict(model.named_buffers()))
    losses = []
    accuracies = []
    for i in range(client_epoch):
        train_score = train(round, client_id, i, model,
                            train_loader,
                            lr=lr,
                            verbose=train_verbosity)

        losses.append(train_score['Loss'][-1].data.item())
        accuracies.append(train_score['Accuracy'][-1])


    mask_log_path = f'{log_folder}/round{round}/c{client_id}.mask'
    client_mask = dict(model.named_buffers())
    log_obj(mask_log_path, client_mask)

    num_pruned, num_params = get_prune_summary(model)
    cur_prune_rate = num_pruned / num_params
    prune_step = math.floor(num_params * prune_step)
    print(f"num_pruned {num_pruned}, num_params {num_params}, cur_prune_rate {cur_prune_rate}, prune_step: {prune_step}")
    del global_model

    eval_score = evaluate(model,
                          test_loader,
                          verbose=False)

    #print(f"previous client acc: {last_client_acc} current client acc: {eval_score['Accuracy'][-1]}")
    eval_score = eval_score['Accuracy'][-1]

    return np.array(losses), np.array(accuracies), cur_prune_rate, model.state_dict(), eval_score


def train_client_model_genesis(acc_thresh, prune_percent, prune_step, prune_verbosity, dataset, arch, lr, train_verbosity, client_epoch, log_folder,
                       round, client_id, global_state_dict, train_loader, test_loader, last_client_acc):

    model = create_model(dataset, arch)
    model.load_state_dict(global_state_dict)

    losses = []
    accuracies = []
    for i in range(client_epoch):
        train_score = train(round, client_id, i, model,
                            train_loader,
                            lr=lr,
                            verbose=train_verbosity)

        losses.append(train_score['Loss'][-1].data.item())
        accuracies.append(train_score['Accuracy'][-1])


    mask_log_path = f'{log_folder}/round{round}/c{client_id}.mask'
    client_mask = dict(model.named_buffers())
    log_obj(mask_log_path, client_mask)

    num_pruned, num_params = get_prune_summary(model)
    cur_prune_rate = num_pruned / num_params
    prune_step = math.floor(num_params * prune_step)
    print(f"num_pruned {num_pruned}, num_params {num_params}, cur_prune_rate {cur_prune_rate}, prune_step: {prune_step}")

    eval_score = evaluate(model,
                          test_loader,
                          verbose=False)

    #print(f"previous client acc: {last_client_acc} current client acc: {eval_score['Accuracy'][-1]}")
    eval_score = eval_score['Accuracy'][-1]

    return np.array(losses), np.array(accuracies), cur_prune_rate, model.state_dict(), eval_score
