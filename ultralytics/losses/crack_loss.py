import torch
import torch.nn as nn

class CompositeLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5):
        super(CompositeLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.mse_loss = nn.MSELoss()

    def forward(self, predictions, targets):
        bce = self.bce_loss(predictions, targets)
        mse = self.mse_loss(predictions, targets)
        return self.alpha * bce + self.beta * mse
# Example usage
# loss_fn = CompositeLoss()
# loss = loss_fn(predictions, targets)
