import torch
import torch.nn as nn
import torch.nn.functional as F
from .pointnet_cls import calc_cls_loss, calc_accuracy
from .pointnet2_layers import PointNet2SetAbstraction


class PointNet2Cls(nn.Module):
    def __init__(self, n_classes, cfgs):
        super().__init__()

        self.sa1 = PointNet2SetAbstraction(
            n_samples=cfgs.sa1.n_samples,
            radius=cfgs.sa1.radius,
            n_points_per_group=cfgs.sa1.n_points_per_group,
            in_channels=3,
            mlp_out_channels=cfgs.sa1.mlp_out_channels
        )
        self.sa2 = PointNet2SetAbstraction(
            n_samples=cfgs.sa2.n_samples,
            radius=cfgs.sa2.radius,
            n_points_per_group=cfgs.sa2.n_points_per_group,
            in_channels=3 + self.sa1.out_channels(),
            mlp_out_channels=cfgs.sa2.mlp_out_channels
        )
        self.sa3 = PointNet2SetAbstraction(
            n_samples=cfgs.sa3.n_samples,
            radius=cfgs.sa3.radius,
            n_points_per_group=cfgs.sa3.n_points_per_group,
            in_channels=3 + self.sa2.out_channels(),
            mlp_out_channels=cfgs.sa3.mlp_out_channels
        )

        self.fc1 = nn.Linear(self.sa3.out_channels(), cfgs.fc1.out_channels)
        self.fc2 = nn.Linear(cfgs.fc1.out_channels, cfgs.fc2.out_channels)
        self.fc3 = nn.Linear(cfgs.fc2.out_channels, n_classes)

        self.bn1 = nn.BatchNorm1d(cfgs.fc1.out_channels)
        self.bn2 = nn.BatchNorm1d(cfgs.fc2.out_channels)

        self.loss, self.acc = torch.Tensor([0]), 0

    def forward(self, inputs, target=None):
        """
        :param inputs: [batch_size, 3, n_points]
        :param target: [batch_size]
        :return: outputs: [batch_size, n_classes]
        """
        batch_size = inputs.shape[0]
        inputs = inputs.transpose(1, 2).contiguous()

        l1_xyz, l1_features = self.sa1(inputs)
        l2_xyz, l2_features = self.sa2(l1_xyz, l1_features)
        l3_xyz, l3_features = self.sa3(l2_xyz, l2_features)

        features = l3_features.view([batch_size, -1])
        features = F.relu(self.bn1(self.fc1(features)))  # [bs, 512]
        features = F.dropout(features, p=0.5, training=self.training)
        features = F.relu(self.bn2(self.fc2(features)))  # [bs, 256]
        features = F.dropout(features, p=0.5, training=self.training)
        outputs = self.fc3(features)  # [bs, n_classes]

        if target is not None:
            self.loss = calc_cls_loss(outputs, target)
            self.acc = calc_accuracy(outputs, target)

        return outputs

    def get_loss(self):
        return self.loss

    def get_accuracy(self):
        return self.acc

    def get_metrics(self):
        return {'loss': self.loss.item(), 'acc': self.acc}

    def get_log_string(self, metrics=None):
        if metrics is None:
            metrics = self.get_metrics()
        return 'loss: %.2f, acc: %.1f%%' % (metrics['loss'], metrics['acc'])

    @staticmethod
    def is_better(curr_metrics, best_metrics):
        if best_metrics is None:
            return True
        return curr_metrics['acc'] > best_metrics['acc']
