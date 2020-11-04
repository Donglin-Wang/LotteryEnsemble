import functools, operator
from server_base import ServerBase
from utils import log
from prune import flatten_prunable, unflatten_prunable


class Server_LotteryFL(ServerBase):
    def __init__(self, clients, model, hyper_params, X, y):
        super().__init__(clients, model, hyper_params, X, y)

    def run(self):
        for r in range(self.hyper_params['R']):
            log(f'Server: round {r}')
            clients = self._sampleClients()
            for c in clients:
                log(f'Server: round {r} - client{c.k}')
                weights = unflatten_prunable(flatten_prunable(self.weights) * c.get_mask(), self.weights)
                c.client_update(weights)

            # PROPOSAL 2: replace this simple FedAvg with one more appropriate for LotteryFL
            #             Also see client_LotteryFL.py
            # OLD:
            # self.weights = self._fed_avg_aggregate(clients) # revisit this approach to aggregation for LotteryFL
            # NEW:
            # TODO: If accepted, clean up the code.
            sizes_weights = []
            flat_g_weights = ServerBase._flatten(self.get_weights())
            for c in self.clients:
                flat_c_weights = ServerBase._flatten(c.get_weights())
                mask = c.get_mask_extended()
                sizes_weights.append((c.sample_size(), mask * flat_c_weights + (1 - mask) * flat_g_weights))

            self.weights = ServerBase._unflatten((1 / self.n) * functools.reduce(operator.add,
                                                                                 [c[0] * c[1] for c in sizes_weights]),
                                                 self.shapes)