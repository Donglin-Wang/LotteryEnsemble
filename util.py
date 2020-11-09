import sys
import copy

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics as skmetrics


from tqdm import tqdm
from tabulate import tabulate
import torch.nn.utils.prune as prune


def average_weights(models):
    with torch.no_grad():
        weights = []
        for model in models:
            weights.append(dict(model.named_parameters()))
        
        avg = copy.deepcopy(weights[0])
        for key in avg.keys():
            for i in range(1, len(weights)):
                avg[key] += weights[i][key]
            avg[key] = torch.div(avg[key], len(weights))
    return avg



def copy_model(model, dataset_name, model_type):
    new_model = create_model(dataset_name, model_type)
    copy_weights(new_model, model.state_dict())
    return new_model
    
def copy_weights(target_model, source_state_dict):
    for name, param in target_model.named_parameters():
        if name in source_state_dict:
            param.data.copy_(source_state_dict[name].data)

def create_model(dataset_name, model_type):
    
    if dataset_name == "mnist": 
        from archs.mnist import mlp
    else: 
        print("You did not enter the name of a supported architecture for this dataset")
        print("Supported datasets: {}, {}".format('"CIFAR10"', '"MNIST"'))
        exit()
    
    if model_type == 'mlp':
        new_model = mlp.MLP()
        # This pruning call is made so that the model is set up for accepting
        # weights from another pruned model. If this is not done, the weights
        # will be incompatible
        prune_fixed_amount(new_model, 0, verbose=False)
        return new_model
    else:
        print("You did not enter the name of a supported architecture for this dataset")
        print("Supported datasets: {}, {}".format('"CIFAR10"', '"MNIST"'))
        exit()
    

def train(model, 
          train_loader,
          lr=0.001,
          verbose=True):
    
    loss_function = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=lr)
    num_batch = len(train_loader)
    metric_names = ['Loss',
                    'Accuracy', 
                    'Balanced Accuracy',
                    'Precision Micro',
                    'Recall Micro',
                    'Precision Macro',
                    'Recall Macro']

    score = {name:[] for name in metric_names}

    progress_bar = tqdm(enumerate(train_loader),
                        total = num_batch,
                        file=sys.stdout)   
    # Iterating over all mini-batches
    for i, data in progress_bar:
    
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
        print("Average scores for the epoch: ")
        print(tabulate(average_scores, headers='keys', tablefmt='github'))
    
    return average_scores

def evaluate(model, data_loader, verbose=True):
    # Swithicing off gradient calculation to save memory
    torch.no_grad()
    # Switch to eval mode so that layers like Dropout function correctly
    model.eval()
    
    metric_names = ['Loss',
                    'Accuracy', 
                    'Balanced Accuracy',
                    'Precision Micro',
                    'Recall Micro',
                    'Precision Macro',
                    'Recall Macro']
    
    score = {name:[] for name in metric_names}
    
    num_batch = len(data_loader)
    
    progress_bar = tqdm(enumerate(data_loader), 
                        total=num_batch,
                        file=sys.stdout)
    
    for i, (x, ytrue) in progress_bar:
        
        yraw = model(x)
        
        _, ypred = torch.max(yraw, 1)
        
        score = calculate_metrics(score, ytrue, yraw, ypred)
        
        progress_bar.set_description('Evaluating')
    
   
        
    for k, v in score.items():
        score[k] = [sum(v) / len(v)]
        
    
    if verbose:
        print('Evaluation Score: ')   
        print(tabulate(score, headers='keys', tablefmt='github'), flush=True)
    
    torch.enable_grad()
    return score

def prune_fixed_amount(model, amount, verbose=True):
    parameters_to_prune, num_global_weights = get_prune_params(model)
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount)

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
    parameters_to_prune, num_global_weights = get_prune_params(model)
    for layer, weight_name in parameters_to_prune:
        num_global_zeros += torch.sum(getattr(layer, weight_name) == 0.0).item()
    
    return num_global_zeros, num_global_weights
        
    
    
def get_prune_params(model):
    layers = []
    
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
                
                    num_global_weights += torch.numel(param)
                    
    return layers, num_global_weights
        
    

def calculate_metrics(score, ytrue, yraw, ypred):
    if 'Loss' in score:
        loss = nn.CrossEntropyLoss()
        score['Loss'].append(loss(yraw, ytrue))
    if 'Accuracy' in score:
        score['Accuracy'].append(skmetrics.accuracy_score(ytrue, ypred))
    if 'Balanced Accuracy' in score:
        score['Balanced Accuracy'].append(skmetrics.balanced_accuracy_score(ytrue, ypred))
    if 'Precision Micro' in score:
        score['Precision Micro'].append(skmetrics.precision_score(ytrue, ypred, average='micro'))
    if 'Recall Micro' in score:
        score['Recall Micro'].append(skmetrics.recall_score(ytrue, ypred, average='micro'))
    if 'Precision Macro' in score:
        score['Precision Macro'].append(skmetrics.precision_score(ytrue, ypred, average='macro'))
    if 'Recall Macro' in score:
        score['Recall Macro'].append(skmetrics.recall_score(ytrue, ypred, average='macro'))
    
    return score
        
        
   

    