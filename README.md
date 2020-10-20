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

## 准备数据集和数据预处理

### ModelNet-40

下载数据集：[https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip](https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip)

数据预处理：执行 `prepare_modelnet40.py` 以转换数据集格式（从 `.txt` 转换到 `.npz`，以加速训练数据读取）。请在 `--zip_path` 参数中填入 `modelnet40_normal_resampled.zip` 的所在路径：

```
python prepare_modelnet40.py --zip_path modelnet40_normal_resampled.zip
```

以上代码会自动解压指定的压缩包，并将转换好的数据集保存在 `datasets/modelnet40` 目录下。执行 `python visualize_modelnet40.py` 可以将 ModelNet-40 数据集的内容可视化。

### S3DIS

下载数据集（选择 Stanford3dDataset_v1.2_Aligned_Version.zip）：[http://buildingparser.stanford.edu/dataset.html](http://buildingparser.stanford.edu/dataset.html)

数据预处理：执行 `prepare_s3dis.py`，它将会完成以下处理：

* 转换数据集格式（从 `.txt` 转换到 `.npz`，以加速训练数据读取）
* 将每个 Room 划分为若干个 Block
* 将较小的 Block 和其相邻且较大的 Block 合并
* 对每个 Block 中的点云进行均匀采样

一些关键的参数包括：

* `--zip_path`： `Stanford3dDataset_v1.2_Aligned_Version.zip` 的所在路径
* `--block_size`：每个 Block 的长、宽（只在 X、 Y 轴进行划分，不在 Z 轴划分）
* `--grid_size`：对 Block 内的点云进行均匀采样时，所使用的单位格子的长、宽、高

执行示例：

```
python prepare_s3dis.py --zip_path Stanford3dDataset_v1.2_Aligned_Version.zip
```

> 注意：在 Area_5/hallway_6 中出现了一个额外字符，需要手动处理。

以上代码会自动解压，并将转换好的数据集保存在 `datasets/s3dis` 目录下。

预处理结果可视化：

* `visualize_s3dis.py`：可视化模型输入（即 DataLoader 的输出）
* `visualize_s3dis_block.py`：可视化每个 Room，通过不同颜色来展示 Block 划分结果
* `visualize_s3dis_room.py`：可视化每个 Room，对比原始点云和均匀重采样后的结果

## 训练

### 训练点云分割模型（使用 ModelNet-40 数据集）

执行 `train.py`，并指定数据集和模型：

```
python train.py model=pointnet_cls dataset=modelnet40
```

> ModelNet-40 的数据预处理不太复杂，指定 4~8 workers 即可。

### 训练点云语义分割模型（使用 S3DIS 数据集）

执行 `train.py`，并指定数据集和模型：

```
python train.py model=pointnet_seg dataset=s3dis
```

由于 S3DIS 采用交叉验证的评估方式，需要训练六次（每次拿其中一个 Area 做测试，其余 Area 做训练）。可以利用 hydra 的 `multirun` 功能执行六次训练：

```
python train.py -m model=pointnet_seg dataset=s3dis dataset.test_area=6,1,2,3,4,5
```

另外，有两种 Block 划分方式可供选择（通过 `dataset.block_type=static/dynamic` 指定）：

* 静态划分（默认）：使用在预处理时划分好的 Block 进行训练，`block_size` 需要在预处理时指定。此模式相比于动态划分，较为节省内存，且 `DataLoader` 的负载较小；但容易过拟合。
* 动态划分：在训练时动态、随机地划分 Block，可以通过 `dataset.block_size=1.0` 指定 Block 长宽。此模式相比于静态划分，不易过拟合；但内存开销较大（因为需要将全部数据集预先读入内存，避免磁盘瓶颈），且 DataLoader 的负载较高。

> 对于动态划分：单卡模式下（16 workers），预计占用 10~15GB 的内存；双卡模式下（每张卡 8 workers），预计占用 90GB 内存。

## 评估

### ModelNet-40

执行 `eval.py`，并指定模型保存点的路径，即可开始评估：

```
python eval.py dataset=modelnet40 model=pointnet_cls model.resume_path=path-to-ckpt/best.pt
```

评估指标为 Top-1 Accuracy。使用默认参数训练得到的精度为 86.3%，相比原文的 89.2% 还有差距，需要改进。

### S3DIS

执行 `eval.py`，并指定模型保存点的路径、测试区域编号，即可开始评估：

```
python eval.py dataset=s3dis dataset.test_area=5 model=pointnet_seg model.resume_path=path-to-ckpt/best.pt
```

评估指标包括 OA (Overall Accuracy) 和 mIoU (Mean IoU)，其中后者更常用。使用默认参数训练得到的精度如下：

| Metrics | Area1 | Area2 | Area3 | Area4 | Area5 | Area6 |
| ------- | ----- | ----- | ----- | ----- | ----- | ----- |
| OA      | 84.01 | 68.16 | 86.00 | 79.08 | 80.77 | 88.62 |
| mIoU    | 59.45 | 31.81 | 62.57 | 44.12 | 45.68 | 69.77 |

此外，在执行 `eval.py` 时如果追加 `+visualize=` 参数，可以将语义分割的可视化结果（`.ply` 格式）存放到对应的日志目录下，使用 MeshLab 即可查看。

## 查看性能指标变化曲线

找到并进入对应的日志目录：

```
cd outputs/2020-09-28/11-26-49
```

使用 Tensorboard 进行可视化：

```
tensorboard --bind_all --logdir .
```

## Ablation Study

### S3DIS：不同 Block 划分方式下的训练效果对比

第一组实验（静态划分）：

```
python train.py -m model=pointnet_seg dataset=s3dis dataset.test_area=6,1,2,3,4,5 dataset.block_type=static
```

第二组实验（动态划分）：

```
python train.py -m model=pointnet_seg dataset=s3dis dataset.test_area=6,1,2,3,4,5 dataset.block_type=dynamic
```

两组实验，仅划分方式不同，其余参数全部一致。测试精度如下表所示（指标为 mIoU ）：

| Block Type | Area1     | Area2     | Area3     | Area4     | Area5     | Area6     |
| ---------- | --------- | --------- | --------- | --------- | --------- | --------- |
| Dynamic    | **59.84** | **32.28** | 61.21     | 43.52     | 44.45     | 69.25     |
| Static     | 59.45     | 31.81     | **62.57** | **44.12** | **45.68** | **69.77** |

由此可见，两种划分方式区别不大，静态划分总体更优（也许是因为更好的划分、归并策略）。