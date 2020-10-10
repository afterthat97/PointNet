import torch
import torch.nn as nn
import torch.nn.functional as F


class STN(nn.Module):
    def __init__(self, k):
        super().__init__()

        self.k = k
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, inputs):
        """
        :param inputs: [batch_size, k, n_points]
        :return: a K*K transform matrix
        """
        inputs = F.relu(self.bn1(self.conv1(inputs)))  # [batch_size, 64, n_points]
        inputs = F.relu(self.bn2(self.conv2(inputs)))  # [batch_size, 128, n_points]
        inputs = F.relu(self.bn3(self.conv3(inputs)))  # [batch_size, 1024, n_points]

        inputs, _ = torch.max(inputs, 2)  # [batch_size, 1024]

        inputs = F.relu(self.bn4(self.fc1(inputs)))  # [batch_size, 512]
        inputs = F.relu(self.bn5(self.fc2(inputs)))  # [batch_size, 256]
        inputs = self.fc3(inputs)  # [batch_size, K*K]

        eye = torch.eye(self.k, dtype=torch.float32).view(-1, self.k * self.k)  # [1, K*K]
        if inputs.is_cuda:
            eye = eye.cuda()
        inputs = inputs + eye
        inputs = inputs.view(-1, self.k, self.k)  # [batch_size, K, K]

        return inputs
