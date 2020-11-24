import torch
import torch.nn as nn
from .pointnet_cls import calc_cls_loss, calc_accuracy
from .pointnet2_layers import PointNet2FeaturePropagation, PointNet2SetAbstraction


class PointNet2Seg(nn.Module):
    def __init__(self, n_classes, n_features, cfgs):
        super().__init__()

        self.sa1 = PointNet2SetAbstraction(
            n_samples=cfgs.sa1.n_samples,
            radius=cfgs.sa1.radius,
            n_points_per_group=cfgs.sa1.n_points_per_group,
            in_channels=3 + n_features,
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
        self.sa4 = PointNet2SetAbstraction(
            n_samples=cfgs.sa4.n_samples,
            radius=cfgs.sa4.radius,
            n_points_per_group=cfgs.sa4.n_points_per_group,
            in_channels=3 + self.sa3.out_channels(),
            mlp_out_channels=cfgs.sa4.mlp_out_channels
        )

        self.fp4 = PointNet2FeaturePropagation(
            in_channels=self.sa4.out_channels() + self.sa3.out_channels(),
            mlp_out_channels=cfgs.fp4.mlp_out_channels
        )
        self.fp3 = PointNet2FeaturePropagation(
            in_channels=self.fp4.out_channels() + self.sa2.out_channels(),
            mlp_out_channels=cfgs.fp3.mlp_out_channels
        )
        self.fp2 = PointNet2FeaturePropagation(
            in_channels=self.fp3.out_channels() + self.sa1.out_channels(),
            mlp_out_channels=cfgs.fp2.mlp_out_channels
        )
        self.fp1 = PointNet2FeaturePropagation(
            in_channels=self.fp2.out_channels() + n_features,
            mlp_out_channels=cfgs.fp1.mlp_out_channels
        )

        self.conv = nn.Conv1d(self.fp1.out_channels(), n_classes, 1)

        self.loss, self.acc = torch.Tensor([0]), 0

    def forward(self, inputs, target=None):
        """
        :param inputs: [batch_size, 3 + n_features, n_points]
        :param target: [batch_size, n_points]
        :return: outputs: [batch_size, n_classes, n_points]
        """
        input_xyz = inputs[:, :3, :].transpose(1, 2).contiguous()
        input_features = inputs[:, 3:, :].transpose(1, 2).contiguous()

        s1_xyz, s1_features = self.sa1(input_xyz, input_features)
        s2_xyz, s2_features = self.sa2(s1_xyz, s1_features)
        s3_xyz, s3_features = self.sa3(s2_xyz, s2_features)
        s4_xyz, s4_features = self.sa4(s3_xyz, s3_features)

        f3_features = self.fp4(s4_xyz, s4_features, s3_xyz, s3_features, k=3)
        f2_features = self.fp3(s3_xyz, f3_features, s2_xyz, s2_features, k=3)
        f1_features = self.fp2(s2_xyz, f2_features, s1_xyz, s1_features, k=3)
        f0_features = self.fp1(s1_xyz, f1_features, input_xyz, input_features, k=3)

        outputs = self.conv(f0_features.transpose(1, 2))  # [bs, n_classes, n_points]

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
