# PyTorch implementation of PointNet

本项目包含了对论文 PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation 的复现（基于 PyTorch），支持点云分类、点云语义分割两种任务。

## 特点

* 支持使用 CPU、单卡、或多卡进行训练
* 支持混合精度加速（需要显卡支持，且 PyTorch 版本新于 1.6.0）
* 使用 `hydra` 管理配置文件，可通过命令行修改默认参数

## 环境配置

使用 `conda` 创建虚拟环境：

```
conda create -n point-net python=3.7
conda activate point-net
```

安装最新版本的 PyTorch：

```
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch  # GPU
conda install pytorch torchvision cpuonly -c pytorch  # CPU
```

安装其他依赖：

```
pip install hydra-core tensorboard open3d
```

## 准备数据集

### ModelNet-40

下载数据集：[https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip](https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip)

转换数据集格式（从 `.txt` 转换到 `.npz`，以加速训练数据读取），在 `--zip_path` 参数中填入上一步下载的 `modelnet40_normal_resampled.zip` 的所在路径：

```
python prepare_modelnet40.py --zip_path modelnet40_normal_resampled.zip
```

以上代码会自动解压，并将转换好的数据集保存在 `datasets/modelnet40` 目录下。执行 `python visualize_modelnet40.py` 可以将 ModelNet-40 数据集的内容可视化。

### S3DIS

下载数据集（选择 Stanford3dDataset_v1.2_Aligned_Version.zip）：[http://buildingparser.stanford.edu/dataset.html](http://buildingparser.stanford.edu/dataset.html)

转换数据集格式（从 `.txt` 转换到 `.npz`，以加速训练数据读取），在 `--zip_path` 参数中填入上一步下载的 `Stanford3dDataset_v1.2_Aligned_Version.zip` 的所在路径：

```
python prepare_s3dis.py --zip_path Stanford3dDataset_v1.2_Aligned_Version.zip
```

以上代码会自动解压，并将转换好的数据集保存在 `datasets/s3dis` 目录下。执行 `python visualize_s3dis.py` 可以将 S3DIS 数据集的内容可视化。

> 注意：在 Area_5/hallway_6 中出现了一个额外字符，需要手动处理。

## 训练

### 训练点云分割模型（使用 ModelNet-40 数据集）

执行 `train.py`，并指定数据集和模型：

```
python train.py model=pointnet_cls dataset=modelnet40
```

> ModelNet-40 的数据预处理不太复杂，指定 4~8 个 worker 即可。

### 训练点云语义分割模型（使用 S3DIS 数据集）

执行 `train.py`，并指定数据集和模型：

```
python train.py model=pointnet_seg dataset=s3dis
```

由于 S3DIS 采用交叉验证的评估方式，需要训练六次（每次拿其中一个 Area 做测试，其余 Area 做训练），可以利用 hydra的 `multirun` 功能执行六次训练：

```
python train.py -m model=pointnet_seg dataset=s3dis dataset.test_area=6,1,2,3,4,5
```

> S3DIS 的所有数据将被预先读入内存。单卡模式下（16 个 worker），预计占用 10~15GB 的内存；双卡模式下（每张卡 8 个 worker），预计占用 80GB 内存。预处理比较费时，在内存足够的情况下，请尽可能多安排 worker 以避免输入瓶颈。

## 评估

### ModelNet-40

执行 `eval.py`，并指定模型保存点的路径，即可开始评估：

```
python eval.py model=pointnet_cls model.resume_path=outputs/2020-10-08/20-47-08/ckpts/epoch-200.pt dataset=modelnet40
```

评估指标为 Top-1 Accuracy。使用默认参数训练得到的精度为 86.3%，相比原文的 89.2% 还有差距，需要改进。

### S3DIS

执行 `eval.py`，并指定模型保存点的路径、测试区域编号，即可开始评估：

```
python eval.py model=pointnet_seg model.resume_path=multirun/2020-10-09/18-04-09/5/ckpts/epoch-200.pt dataset=s3dis dataset.test_area=5
```

评估指标包括 OA (Overall Accuracy) 和 mIoU (Mean IoU)。使用默认参数训练得到的精度如下：

| Metrics | Area1 | Area2 | Area3 | Area4 | Area5 | Area6 |
| ------- | ----- | ----- | ----- | ----- | ----- | ----- |
| OA      | 83.04 | 69.85 | 86.17 | 77.29 | 80.87 | 86.74 |
| mIoU    | 58.40 | 33.48 | 61.93 | 43.59 | 45.60 | 66.99 |

此外，语义分割的可视化结果（`.ply` 格式）也存放在对应的日志目录下，使用 MeshLab 即可查看。

## 查看性能指标变化曲线

找到并进入对应的日志目录：

```
cd outputs/2020-09-28/11-26-49
```

使用 Tensorboard 进行可视化：

```
tensorboard --bind_all --logdir .
```
