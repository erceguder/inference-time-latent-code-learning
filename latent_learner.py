from torch import nn

class LatentLearner(nn.Module):
    def __init__(self, init_size=1024):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(init_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
        )
    def forward(self, data):
        return self.layers(data)