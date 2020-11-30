import torch
import torch.nn as nn
import numpy as np
torch.manual_seed(0)
np.random.seed(0)

class MLP(nn.Module):
    def __init__(self, num_classes=10):
        super(MLP, self).__init__()
        # self.classifier = nn.Sequential(
        #     nn.Linear(28*28, 64),
        #     nn.Dropout(),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(64, num_classes),
        #     nn.Softmax(dim=1)
        # )
        self.classifier = nn.Sequential(
            nn.Linear(28*28, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, num_classes),
        )
        
    def forward(self, x):
        x = torch.flatten(x, 1)
        return self.classifier(x)

