import numpy as np
from client_base import ClientBase
from global_settings import VERBOSE
from prune import flatten_prunable, unflatten_prunable


class Client_LotteryFL(ClientBase):
    def __init__(self, k, model_creator, initial_weights, hyper_params, X, y):
        super().__init__(k, model_creator(), hyper_params, X, y)
        self.initial_weights = initial_weights
        self.mask = np.ones(flatten_prunable(initial_weights).shape)
        self.curr_prune_rate = 0

    def prune(weights, mask, r_p):
        alive = weights[np.nonzero(weights * mask)]  # flattened array of nonzero values
        percentile_value = np.percentile(abs(alive), r_p * 100)
        return np.where(abs(weights) < percentile_value, 0, mask)

    def get_mask(self):
        return self.mask

    # PROPOSAL 2: supports replacing simple FedAvg with one more appropriate for LotteryFL
    #             Also see server_LotteryFL.py
    # This returns a longer mask that includes 1's for all non-prunable parameters (e.g. bias)
    def get_mask_extended(self):
        curr_offset = 0
        result = []

        for layer in self.model.get_weights():
            if layer.ndim > 1:
                count = np.prod(layer.shape)
                result.extend(self.mask[0, curr_offset:curr_offset + count])
                curr_offset += count
            else:
                result.extend(np.ones(layer.size, dtype=int))
        return np.array(result)

    def get_weights(self):
        weights = self.model.get_weights()
        return unflatten_prunable(flatten_prunable(weights) * self.mask, weights)

    def client_update(self, weights):
        self.model.set_weights(weights)
        acc = self.model.evaluate(self.X, self.y, return_dict=True, verbose=VERBOSE)['accuracy']

        print(f'Client: accuracy {acc}, prune rate {self.curr_prune_rate}')
        if acc > self.hyper_params['acc_threshold'] and self.curr_prune_rate < self.hyper_params['r_target']:
            print('Client is pruning.')
            self.mask = Client_LotteryFL.prune(flatten_prunable(weights), self.mask, self.hyper_params['r_p'])
            self.curr_prune_rate = (self.mask.size - np.count_nonzero(self.mask)) / self.mask.size

            new_weights = unflatten_prunable(flatten_prunable(self.initial_weights) * self.mask, self.initial_weights)
            self.model.set_weights(new_weights)

        self.model.fit(self.X, self.y,
                       batch_size=self.hyper_params['B'], epochs=self.hyper_params['E'],
                       verbose=VERBOSE)
