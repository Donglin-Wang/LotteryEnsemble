from abc import ABC, abstractmethod
import functools, operator
import numpy as np


class ServerBase(ABC):
    def __init__(self, clients, model, hyper_params, X, y):
        self.clients = clients
        self.model = model  # the global model
        self.weights = model.get_weights()
        self.hyper_params = hyper_params
        self.X = X
        self.y = y
        self.n = functools.reduce(operator.add, [c.sample_size() for c in self.clients])  # total samples
        self.shapes = [layer.shape for layer in self.weights]

    def _sampleClients(self):
        choices = np.random.choice(self.hyper_params['K'],
                                   max(1, int(self.hyper_params['C']*self.hyper_params['K'])),
                                   replace=False)
        return [self.clients[i] for i in choices]

    def _flatten(weights):
        return np.concatenate([layer.reshape(1, -1) for layer in weights], axis=1)

    def _unflatten(weights, shapes):
        curr_offset = 0
        result = []

        for s in shapes:
            count = np.prod(s)
            result.append(weights[0, curr_offset:curr_offset + count].reshape(s))
            curr_offset += count
        return result

# PROPOSAL: replace this client-sample-based FedAvg with a FedAvg over all clients
#           which is what I think the FedAvg algorithm actually wants.
# OLD:
#     def _fed_avg_aggregate(self, clients):
#         return ServerBase._unflatten(
#             (1/self.n) * functools.reduce(operator.add,
#                                           [c.sample_size() * ServerBase._flatten(c.get_weights()) for c in clients]),
#             self.shapes)
# NEW:
# Here we ignore the parameter clients and use self.clients, which is actually what FedAvg is supposed to do.
# TODO: If accepted, get rid of argument and clean up calls.
    def _fed_avg_aggregate(self, clients):
        return ServerBase._unflatten(
            (1 / self.n) * functools.reduce(operator.add,
                                            [c.sample_size() * ServerBase._flatten(c.get_weights()) for c in
                                             self.clients]),
            self.shapes)

    def get_weights(self):
        return self.weights

    @abstractmethod
    def run(self):
        pass
