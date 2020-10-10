import torch
import torch.nn as nn
import torch.nn.functional as F
from .stn import STN


class PointNetBase(nn.Module):
    def __init__(self, n_channels=3, input_trans=False, feat_trans=False):
        super().__init__()

        self.stn1 = STN(k=n_channels) if input_trans else None
        self.stn2 = STN(k=64) if feat_trans else None

        self.conv1 = nn.Conv1d(n_channels, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.conv5 = nn.Conv1d(128, 1024, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)

    def forward(self, inputs):
        """
        :param inputs: [batch_size, n_channels, n_points]
        :return: end_points
        """
        end_points = {}

        if self.stn1 is not None:
            # get transform matrix for input
            input_trans_mat = self.stn1(inputs)  # [bs, n_channels, n_channels]
            end_points['input_trans_mat'] = input_trans_mat
            # apply transform
            inputs = inputs.transpose(1, 2)  # [bs, n_points, n_channels]
            inputs = torch.bmm(inputs, input_trans_mat)  # [bs, n_points, n_channels]
            inputs = inputs.transpose(1, 2)  # [bs, n_channels, n_points]

        # mlp(64, 64)
        inputs = F.relu(self.bn1(self.conv1(inputs)))  # [bs, 64, n_points]
        inputs = F.relu(self.bn2(self.conv2(inputs)))  # [bs, 64, n_points]

        if self.stn2 is not None:
            # get transform matrix for features
            feat_trans_mat = self.stn2(inputs)  # [bs, 64, 64]
            end_points['feat_trans_mat'] = feat_trans_mat
            # apply transform
            inputs = inputs.transpose(1, 2)  # [bs, n_points, 64]
            inputs = torch.bmm(inputs, feat_trans_mat)  # [bs, n_points, 64]
            inputs = inputs.transpose(1, 2)  # [bs, 64, n_points]
        end_points['local_feat'] = inputs

        # mlp(64, 128, 1024)
        inputs = F.relu(self.bn3(self.conv3(inputs)))  # [bs, 64, n_points]
        inputs = F.relu(self.bn4(self.conv4(inputs)))  # [bs, 128, n_points]
        inputs = F.relu(self.bn5(self.conv5(inputs)))  # [bs, 1024, n_points]

        # max pool
        inputs, _ = torch.max(inputs, dim=2)  # [bs, 1024]
        end_points['global_feat'] = inputs

        return end_points
