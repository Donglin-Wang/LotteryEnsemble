# Foreman

* deals with parameters for the experiment to be run
* loads the requested data
* pick the right model for the data
* creates the global model
* creates the clients
* creates the server

## Foreman parameters
The parameters are all grouped together, but different parameters are intended for different audiences.

### experiment parameters
Parameters which govern the experiment to be run.
* <b>algo</b>:
    * "FedAvg" for algorithm from McMahan et al.
    * "LotteryFL" for algorithm from Li et al.
    * "Genesis" for our modification.
* <b>data</b>: The dataset to use. For now, this determins the NN model that will be used.
    * "MNIST_IID"
    * "MNIST_NON-IID"
    * "CIFAR10_IID"
    * "CIFAR10_NON-IID"
* <b>K</b>: number of clients

### client parameters
Hyper parameters required for clients and server as specified in the articles by McMahan et al. and Li et al.

* <b>E</b>: number of epochs clients will train in a single round
* <b>B</b>: client minibatch size
* <b>eta</b>: the learning rate
* <b>acc_threshold</b>: the accuracy threshold required before pruning
* <b>r_target</b>: the target pruning rate, between 0 and 1
* <b>r_p</b>: the pruning rate to use on a given iteration, between 0 and 1

### For Server
Hyper parameters for the server.

* <b>R</b>: number of rounds
* <b>C</b>: the fraction of clients to run a round on (C=0 means 1 client)
* <b>K</b>: number of clients