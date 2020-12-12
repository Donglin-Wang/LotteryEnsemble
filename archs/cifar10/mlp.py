import torch
import torch.nn as nn
import torch.nn.functional as F

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
        # self.classifier = nn.Sequential(
        #     nn.Linear(32*32*3, 100),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(100, 100),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(100, num_classes),
        # )

        # self.linear1 = nn.Linear(32*32*3, 1024)
        # self.linear2 = nn.Linear(1024, 512)
        # self.linear3 = nn.Linear(512, 64)
        # self.linear4 = nn.Linear(64, 64)
        # self.linear5 = nn.Linear(64, num_classes)

        self.classifier = nn.Sequential(
            nn.Linear(32*32*3, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )


        
    def forward(self, x):
        #x = torch.flatten(x, 1)
        #return self.classifier(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

