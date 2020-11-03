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

            self.weights = self._fed_avg_aggregate(clients)  # revisit this approach to aggregation for LotteryFL