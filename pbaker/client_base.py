class ClientBase:
    def __init__(self, k, model, hyper_params, X, y):
        self.k              = k
        self.model          = model
        self.model.client   = self
        self.hyper_params   = hyper_params
        self.X              = X
        self.y              = y


    def get_weights(self):
        return self.model.get_weights()


    def sample_size(self):
        return self.X.shape[0]