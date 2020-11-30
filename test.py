# TEST 1: Test if log_obj in util.py will correctly log files in non-existent
# directories

def test1_log_obj():
    from archs.mnist.mlp import MLP
    from util import log_obj
    
    # Logging a PyTorch model
    mlp = MLP()
    log_obj('./log_test/clients/client1/epoch1_model.pickle', mlp)

    # Logging a list of models
    models = [MLP() for i in range(10)]
    log_obj('./log_test/server/round1_client_models.pickle', models)
    
# TEST 2: Test if the logged objects can be correctly read from files

def test2_read_obj():
    import pickle
    
    file = open('./log_test/server/round1_client_models.pickle', 'rb')
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
    
    log_obj('./log_test/clients/client1/epoch1_model.torch', mlp)
    
    with open('./log_test/clients/client1/epoch1_model.torch', 'rb') as file:
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
    
    log_obj('./log_test/clients/client1/epoch1_model.torch', mlp)
    
    with open('./log_test/clients/client1/epoch1_model.torch', 'rb') as file:
        model = torch.load(file)
        print(list(model.named_buffers()))

# TEST 5: Test the file size that contains a list of 400 models
# NOTE: In the LotteryFL paper, the MNIST experiment is run with 400 clients
#       over 2000 epochs. A list 400 clients takes about 2.5GB of hardrive 
#       space
def test5_test_model_list_size():
    import os
    from util import log_obj, create_model
    models = [create_model('mnist', 'mlp') for _ in range(400)]
    log_obj('./log_test/server/400_models_list.model_list', models)
    filesize = os.path.getsize('./log_test/server/400_models_list.model_list')
    byte_per_gb = 1E9
    print(f'The log file size is {filesize/byte_per_gb}GB')

# TEST 6: Tes whether copying model affects the pruning

def test6_prune_again():
    from archs.mnist.mlp import MLP
    from util import copy_model, prune_fixed_amount
    
    mlp = MLP()
    
    prune_fixed_amount(mlp, 0.5)
    
    new_mlp = copy_model(mlp, 'mnist', 'mlp')
    
    prune_fixed_amount(new_mlp, 0.5)
    

if __name__ == '__main__':
    
    print('Running tests...')
    
    # test1_log_obj()
    
    # test2_read_obj()
    
    # test3_prune_pickle()
    
    # test4_log_and_load()
    
    # test5_test_model_list_size()
    
    test6_prune_again()
    
    
    