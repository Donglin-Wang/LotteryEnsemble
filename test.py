# TEST 1: Test if log_obj in util.py will correctly log files in non-existent
# directories

def test1_log_obj():
    from archs.mnist.mlp import MLP
    from util import log_obj
    
    # Logging a PyTorch model
    mlp = MLP()
    log_obj('./log/clients/client1/epoch1_model.pickle', mlp)

    # Logging a list of models
    models = [MLP() for i in range(10)]
    log_obj('./log/server/round1_client_models.pickle', models)
    
# TEST 2: Test if the logged objects can be correctly read from files

def test2_read_obj():
    import pickle
    
    file = open('./log/server/round1_client_models.pickle', 'rb')
    obj = pickle.load(file)
    if isinstance(obj, list):
        for model in obj:
            print(model)
    
# TEST 3: Test if pruning would lead to a failed pickle
# NOTE: This test is implemented because pruning would cause copy.deepcopy to
#       fail. Therefore, it might be a good idea to check if pickle would also 
#       fail.

def test3_prune_pickle():
    import torch
    from archs.mnist.mlp import MLP
    from util import log_obj
    
    mlp = MLP()
    
    log_obj('./log/clients/client1/epoch1_model.torch', mlp)
    
    with open('./log/clients/client1/epoch1_model.torch', 'rb') as file:
        model = torch.load(file)
        print(model)
        
    return

# TEST 4: Test if pruning is preserved after writing to and loading from file

def test4_log_and_load():
    import torch
    from archs.mnist.mlp import MLP
    from util import log_obj, prune_fixed_amount
    
    mlp = MLP()
    prune_fixed_amount(mlp, 150000, verbose=False)
    
    log_obj('./log/clients/client1/epoch1_model.torch', mlp)
    
    with open('./log/clients/client1/epoch1_model.torch', 'rb') as file:
        model = torch.load(file)
        print(list(model.named_buffers()))


if __name__ == '__main__':
    
    print('Running tests...')
    
    # test1_log_obj()
    
    # test2_read_obj()
    
    # test3_prune_pickle()
    
    # test4_log_and_load()
    
    