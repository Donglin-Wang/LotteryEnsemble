from server_base import ServerBase
from utils import log


class Server_FedAvg(ServerBase):
    def __init__(self, clients, model, hyper_params, X, y):
        super().__init__(clients, model, hyper_params, X, y)

    def run(self):
        for r in range(self.hyper_params['R']):
            log(f'Server: round {r}')
            clients = self._sampleClients()
            for c in clients:
                log(f'Server: round {r} - client{c.k}')
                c.client_update(self.weights)

            self.weights = self._fed_avg_aggregate(clients)
