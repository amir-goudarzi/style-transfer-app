from torch import nn
import torch
class GramMatrix(nn.Module):
    def forward(self, x):
        batch, channels, height, width = x.size()
        features = x.view(batch, channels, height * width)
        gram_matrix = torch.bmm(features, features.transpose(1, 2))
        return gram_matrix
gramMatrix = GramMatrix()