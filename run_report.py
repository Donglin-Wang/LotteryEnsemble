from report_experiments import MNIST_experiments, CIFAR10_experiments
from experiment import run_experiments
import torch, numpy as np
torch.manual_seed(0)
np.random.seed(0)

if __name__ == '__main__':
    overrides = {
        'log_folder': './report_output',
        'running_on_cloud': False
    }

    # ------------------
    # MNIST
    # ------------------
    # To run 1 or more set selection
    # selection = ['MNIST_standalone', 'MNIST_FedAvg']
    # run_experiments({k: MNIST_experiments[k] for k in selection}, overrides)

    # To run all MNIST tests
    #run_experiments(MNIST_experiments, overrides)

    # ------------------
    # CIFAR
    # ------------------
    # To run 1 or more set selection
    # selection = ['CIFAR10_standalone', 'CIFAR10_FedAvg']
    # run_experiments({k: CIFAR10_experiments[k] for k in selection}, overrides)

    # To run all CIFAR10 tests
    run_experiments(CIFAR10_experiments, overrides)
