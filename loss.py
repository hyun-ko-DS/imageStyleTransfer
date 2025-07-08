import torch
import torch.nn as nn
import torch.nn.functional as F

class ContentLoss(nn.Module):
    def __init__(self):
        super(ContentLoss, self).__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        loss = F.mse_loss(x, y)
        return loss


class StyleLoss(nn.Module):
    def __init__(self):
        super(StyleLoss, self).__init__()

    def gram_matrix(self, x: torch.Tensor):
        b, c, h, w = x.size()

        features = x.view(b, c, h * w) # shape: (b, c, h * w) = (b, N, M)
        features_T = features.transpose(1, 2) # shape: (b, h * w, c) = (b, M, N)
        gram_mat = torch.matmul(features, features_T) # shape: (b, N, N)

        return gram_mat.div(b * c * h * w)
    
    def forward(self, x: torch.Tensor, y: torch.Tensor):
        gram_x = self.gram_matrix(x)
        gram_y = self.gram_matrix(y)
        loss = F.mse_loss(gram_x, gram_y)
        return loss