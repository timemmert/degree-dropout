import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class DegreeDropout(nn.Module):
    def __init__(self, p, alpha=0.006):
        """

        :param p: Probability to drop out nodes
        :param alpha: Weights Threshold to remove them for the degree cpmputation
        """
        super(DegreeDropout, self).__init__()
        if p < 0 or p >= 1:
            raise ValueError("Dropout probability must be in the range [0, 1)")

        self.p = p  # Probability of dropping out a neuron
        self.alpha = alpha

    def forward(self, x, following_weights):
        # sum(sum(abs(following_weights) > 0.006)) / (following_weights.shape[0] * following_weights.shape[1])
        # Drop 15 % of the weights

        if self.training:
            mask = abs(following_weights) > self.alpha
            degrees = torch.count_nonzero(mask, 0)
            probabs = torch.nn.functional.softmax(degrees, dim=0, dtype=torch.float32)

            if self.p != 0:
                indices = torch.multinomial(
                    probabs, int(self.p * following_weights.shape[1]), replacement=False
                )
            else:
                indices = torch.tensor([], dtype=torch.int64)

            mask = torch.ones_like(x)
            mask[:, indices] = 0
            x = x * mask

            x /= 1 - self.p

            return x
        else:
            # During inference, simply return the input as is
            return x
