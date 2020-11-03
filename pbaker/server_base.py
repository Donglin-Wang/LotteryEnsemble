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
        return self.clients

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

    def _fed_avg_aggregate(self, clients):
        return ServerBase._unflatten(
            (1 / self.n) * functools.reduce(operator.add,
                                            [c.sample_size() * ServerBase._flatten(c.get_weights()) for c in clients]),
            self.shapes)

    def get_weights(self):
        return self.weights

    @abstractmethod
    def run(self):
        pass
