import torch
import torch.nn as nn
import torch.nn.functional as F
from .pointnet_base import PointNetBase


def calc_cls_loss(logits, target):
    return F.cross_entropy(logits, target)


def calc_reg_loss(feat_trans_mat, reg_weight=0.001):
    k = feat_trans_mat.size()[1]
    eye = torch.eye(k)[None, :, :].to(device=feat_trans_mat.device)
    diff = torch.bmm(feat_trans_mat, feat_trans_mat.transpose(2, 1)) - eye
    reg_loss = torch.mean(torch.norm(diff, dim=[1, 2]))
    return reg_loss * reg_weight


@torch.no_grad()
def calc_accuracy(logits, target):
    pred = torch.argmax(logits, dim=1)
    correct_n = pred.eq(target).sum()
    return 100.0 * correct_n / target.nelement()


class PointNetCls(nn.Module):
    def __init__(self, n_classes):
        super().__init__()

        self.base = PointNetBase(input_trans=True, feat_trans=True)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, n_classes)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)

        self.cls_loss, self.reg_loss, self.acc = torch.Tensor([0]), torch.Tensor([0]), 0

    def forward(self, inputs, target=None):
        """
        :param inputs: [batch_size, 3, n_points]
        :param target: [batch_size]
        :return: outputs: [batch_size, n_classes]
        """
        end_points = self.base(inputs)

        # mlp(512, 256, n_classes)
        features = end_points['global_feat']
        features = F.relu(self.bn1(self.fc1(features)))  # [bs, 512]
        features = F.dropout(features, p=0.5, training=self.training)
        features = F.relu(self.bn2(self.fc2(features)))  # [bs, 256]
        features = F.dropout(features, p=0.5, training=self.training)
        outputs = self.fc3(features)  # [bs, n_classes]

        if target is not None:
            self.cls_loss = calc_cls_loss(outputs, target)
            self.reg_loss = calc_reg_loss(end_points['feat_trans_mat'])
            self.acc = calc_accuracy(outputs, target)

        return outputs

    def get_loss(self):
        return self.cls_loss + self.reg_loss

    def get_accuracy(self):
        return self.acc

    def get_metrics(self):
        return {
            'cls_loss': self.cls_loss.item(),
            'reg_loss': self.reg_loss.item(),
            'acc': self.acc
        }

    def get_log_string(self, metrics=None):
        if metrics is None:
            metrics = self.get_metrics()
        return 'loss: %.2f, acc: %.1f%%' % (metrics['cls_loss'] + metrics['reg_loss'], metrics['acc'])

    @staticmethod
    def is_better(curr_metrics, best_metrics):
        if best_metrics is None:
            return True
        return curr_metrics['acc'] > best_metrics['acc']
