from client_base import ClientBase
from global_settings import VERBOSE


class Client_FedAvg(ClientBase):
    def __init__(self, k, model_creator, hyper_params, X, y):
        super().__init__(k, model_creator(), hyper_params, X, y)
        self.model_creator = model_creator


    def client_update(self, weights):
        self.model.set_weights(weights)
        self.model.fit(self.X, self.y,
                       batch_size=self.hyper_params['B'], epochs=self.hyper_params['E'],
                       verbose=VERBOSE)