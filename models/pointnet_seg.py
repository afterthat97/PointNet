import torch
import torch.nn as nn
import torch.nn.functional as F
from .pointnet_base import PointNetBase
from .pointnet_cls import calc_cls_loss, calc_accuracy


class PointNetSeg(nn.Module):
    def __init__(self, n_classes, n_channels):
        super().__init__()

        self.base = PointNetBase(n_channels, input_trans=False, feat_trans=False)

        self.conv1 = nn.Conv1d(1024 + 64, 512, 1)
        self.conv2 = nn.Conv1d(512, 256, 1)
        self.conv3 = nn.Conv1d(256, 128, 1)
        self.conv4 = nn.Conv1d(128, 128, 1)
        self.conv5 = nn.Conv1d(128, n_classes, 1)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(128)

        self.loss, self.acc = torch.Tensor([0]), 0

    def forward(self, inputs, target=None):
        """
        :param inputs: [batch_size, n_channels, n_points]
        :param target: [batch_size, n_points]
        :return: outputs: [batch_size, n_classes, n_points]
        """
        end_points = self.base(inputs)

        local_feat = end_points['local_feat']  # [bs, 64, n_points]
        n_points = local_feat.size()[-1]

        global_feat = end_points['global_feat']  # [bs, 1024]
        global_feat = global_feat[:, :, None].expand(-1, -1, n_points)  # [bs, 1024, n_points]

        inputs = torch.cat([local_feat, global_feat], dim=1)  # [bs, 1088, n_points]
        end_points['concat_feat'] = inputs

        # mlp(512, 256, 128)
        inputs = F.relu(self.bn1(self.conv1(inputs)))  # [bs, 512, n_points]
        inputs = F.relu(self.bn2(self.conv2(inputs)))  # [bs, 256, n_points]
        inputs = F.relu(self.bn3(self.conv3(inputs)))  # [bs, 128, n_points]

        # mlp(128, n_classes)
        inputs = F.relu(self.bn4(self.conv4(inputs)))  # [bs, 128, n_points]
        outputs = self.conv5(inputs)  # [bs, n_classes, n_points]

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
