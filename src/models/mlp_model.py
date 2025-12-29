import torch.nn as nn


class MLP3(nn.Module):
    """
    Simple multilayer perceptron for tabular data.

    Architecture:
        input_dim → 256 → 128 → 64 → num_classes

    Uses ReLU activations and no explicit regularization.
    Intended as a lightweight baseline model.
    """
    def __init__(self, input_dim, num_classes=100):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)

class MLP4(nn.Module):
    """
    Deeper multilayer perceptron with dropout regularization.

    Architecture:
        input_dim → 512 → 256 → 128 → 64 → num_classes

    Dropout is applied after each hidden layer to reduce overfitting.
    Intended as a higher-capacity model for tabular learning.
    """
    def __init__(self, input_dim, dropout, num_classes=100):
        super().__init__()
        self.net = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, num_classes)
                )

    def forward(self, x):
        return self.net(x)
