import torch 

class AdditionNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Sequential(
            torch.nn.Linear(2, 4),
            torch.nn.ReLU(),
            torch.nn.Linear(4, 1),
        )

    def forward(self, x):
        return self.layer(x)
    

class AnomalyDetectionNet(torch.nn.Module):
    # This is a simple autoencoder for anomaly detection. It takes an input of size 10, compresses it to a latent space of size 8, and then reconstructs it back to the original size.
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Sequential(
            torch.nn.Linear(10, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 1),
            torch.nn.ReLU(),
            torch.nn.Linear(1, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 10),
        )

    def forward(self, x):
        return self.layer(x)