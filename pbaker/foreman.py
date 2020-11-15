import tensorflow as tf
from models import create_MNIST_model
from data import split_data
from client_FedAvg import Client_FedAvg
from client_LotteryFL import Client_LotteryFL
from server_FedAvg import Server_FedAvg
from server_LotterFL import Server_LotteryFL


class Foreman():
    def __init__(self, params):
        self.params = params

        # load data, initialize model_creator
        if params['data'] == 'MNIST_IID':
            (self.X_train, self.y_train), (self.X_test, self.y_test) = tf.keras.datasets.mnist.load_data()
            self.model_creator = create_MNIST_model
        else:
            assert False, f"Unsupported argument for data: {params['data']}"

        # split data
        partition = split_data((self.X_train, self.y_train), params['K'])

        # create global model
        self.global_model = self.model_creator()
        self.initial_weights = self.global_model.get_weights()

        # handle clients
        self.clients = []
        for k in range(self.params['K']):
            if params['algo'] == 'FedAvg':
                self.clients.append(Client_FedAvg(k,
                                                  self.model_creator,
                                                  {key: params[key] for key in ('E', 'B', 'eta')},
                                                  partition[k][0],
                                                  partition[k][1]))
            elif params['algo'] == 'LotteryFL':
                self.clients.append(Client_LotteryFL(k,
                                                     self.model_creator,
                                                     self.initial_weights,
                                                     {key: params[key] for key in ('E', 'B', 'eta', 'acc_threshold',
                                                                                   'r_target', 'r_p')},
                                                     partition[k][0],
                                                     partition[k][1]))
            else:
                assert False, f"Unsupported argument for algo: {params['algo']}"

        # handle server
        if params['algo'] == 'FedAvg':
            self.server = Server_FedAvg(self.clients, self.global_model,
                                        {key: params[key] for key in ('R', 'C', 'K')},
                                        self.X_train, self.y_train)
        elif params['algo'] == 'LotteryFL':
            self.server = Server_LotteryFL(self.clients, self.global_model,
                                           {key: params[key] for key in ('R', 'C', 'K')},
                                           self.X_train, self.y_train)
        else:
            self.server = Server_Genesis(self.clients, self.global_model,
                                         {key: params[key] for key in ('R', 'C', 'K')},
                                         self.X_train, self.y_train)

    def run(self):
        print("Foreman: run with initial weights")
        self.global_model.evaluate(self.X_train, self.y_train)

        self.server.run()
        print("Foreman: run with trained global model")
        self.global_model.set_weights(self.server.get_weights())
        self.global_model.evaluate(self.X_train, self.y_train)

        # we evaluate final client and server model performance at this point
        print("Foreman: rerun with initial weights")
        self.global_model.set_weights(self.initial_weights)
        self.global_model.evaluate(self.X_train, self.y_train)