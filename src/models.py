import torch 
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):

    def __init__(self, is_training):

        super(Model, self).__init__()
        self.is_training = is_training
        # Define the model layers or download some pretrained model
        # And do transfer learning 

        # Example layers
        # For use with MNIST DataSet
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 4, kernel_size = 3)
        self.conv2 = nn.Conv2d(in_channels = 4, out_channels = 8, kernel_size = 3)
        self.conv3 = nn.Conv2d(in_channels = 8, out_channels = 10, kernel_size = 3)
        self.avg_pool = nn.AvgPool2d(kernel_size = 4)

    def forward(self, xb):
        batch_size, _, _, _ = xb.shape

        xb = self.conv1(xb)
        xb = self.conv2(xb)
        xb = self.conv3(xb)
        xb.view(batch_size, -1)
        xb = self.avg_pool(xb)

        return xb.view(batch_size, -1)
