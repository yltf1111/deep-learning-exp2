## 一、环境配置

### 创建 conda 环境

```bash
conda create --name openmmlab python=3.8 -y
conda activate openmmlab
```

### 安装 pytorch

```bash
conda install pytorch torchvision -c pytorch
```

### 安装 mm 库

```bash
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
```

### 安装其余库

```bash
cd mmdetection
pip install -v -e .
```

## 二、数据集准备

本实验采用 COCO 2017 和 VOC 2012 数据集，将数据集下载并放置在 `data` 目录下，本实验对 VOC 2012 重新进行了划分并转化为 COCO 格式以统一训练，进入`data` 目录并运行：

```bash
python dataset_trans.py
python xml2json.py
```

## 三、模型训练和测试

本实验使用 COCO train 的前 10000 条数据进行训练，使用 val 的 5000 条数据测试。voc 使用划分后的 11987 条数据训练，1713 条测试。

### 3.1 在 COCO 数据集上进行训练和测试

#### 3.1.1 Fast R-CNN

通过预训练的 RPN（下载权重到对应目录）先导出数据集的 proposal：

```bash
python tools/test.py configs/rpn/rpn_r50_fpn_1x_coco_dump_train_proposals.py checkpoints/rpn_r50_fpn_1x_coco_20200218-5525fa2e.pth
python tools/test.py configs/rpn/rpn_r50_fpn_1x_coco_dump_val_proposals.py checkpoints/rpn_r50_fpn_1x_coco_20200218-5525fa2e.pth
```

训练：

```bash
python tools/train.py configs/fast_rcnn/fast-rcnn_r50_fpn_1x_coco.py --work-dir work_dirs/fast_rcnn_r50_fpn_1x_coco_bs16 --cfg-options train_dataloader.batch_size=16 train_dataloader.dataset.indices=10000 train_cfg.val_interval=100
```

测试：

```bash
python tools/test.py configs/fast_rcnn/fast-rcnn_r50_fpn_1x_coco.py work_dirs/fast_rcnn_r50_fpn_1x_coco_bs16/epoch_12.pth --work-dir work_dirs/fast_rcnn_r50_fpn_1x_coco_bs16
```

#### 3.1.2 Faster R-CNN

训练：

```bash
python tools/train.py configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py --work-dir work_dirs/faster_rcnn_r50_fpn_1x_coco_bs16 --cfg-options train_dataloader.batch_size=16 train_dataloader.dataset.indices=10000 train_cfg.val_interval=100
```

测试：

```bash
python tools/test.py configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py work_dirs/faster_rcnn_r50_fpn_1x_coco_bs16/epoch_12.pth --work-dir work_dirs/faster_rcnn_r50_fpn_1x_coco_bs16
```

更换不同的 backbone ResNet-101：

训练：

```bash
python tools/train.py configs/faster_rcnn/faster-rcnn_r101_fpn_1x_coco.py --work-dir work_dirs/faster_rcnn_r101_fpn_1x_coco_bs16 --cfg-options train_dataloader.batch_size=16 train_dataloader.dataset.indices=10000 train_cfg.val_interval=100
```

测试：

```bash
python tools/test.py configs/faster_rcnn/faster-rcnn_r101_fpn_1x_coco.py work_dirs/faster_rcnn_r101_fpn_1x_coco_bs16/epoch_12.pth --work-dir work_dirs/faster_rcnn_r101_fpn_1x_coco_bs16
```

#### 3.1.3 YOLO v3

训练：

```bash
python tools/train.py configs/yolo/yolov3_d53_8xb8-ms-416-273e_coco.py --work-dir work_dirs/yolov3_d53_416_coco_bs16 --cfg-options train_dataloader.batch_size=16 train_dataloader.dataset.indices=10000 train_cfg.val_interval=500 optim_wrapper.optimizer.lr=0.00025
```

测试：

```bash
python tools/test.py configs/yolo/yolov3_d53_8xb8-ms-416-273e_coco.py work_dirs/yolov3_d53_416_coco_bs16/epoch_231.pth --work-dir work_dirs/yolov3_d53_416_coco_bs16
```

#### 3.1.4 SSD

训练：

```bash
python tools/train.py configs/ssd/ssd300_coco.py --work-dir work_dirs/ssd300_coco_bs16 --cfg-options train_dataloader.batch_size=16 train_dataloader.dataset.dataset.indices=10000 train_cfg.val_interval=100 optim_wrapper.optimizer.lr=0.0005
```

测试：

```bash
python tools/test.py configs/ssd/ssd300_coco.py work_dirs/ssd300_coco_bs16/epoch_24.pth --work-dir work_dirs/ssd300_coco_bs16
```

采用 batch size=8，learning rate=0.002：

训练：

```bash
python tools/train.py configs/ssd/ssd300_coco.py --work-dir work_dirs/ssd300_coco_bsx --cfg-options train_dataloader.dataset.dataset.indices=10000 train_cfg.val_interval=100
```

测试：

```bash
python tools/test.py configs/ssd/ssd300_coco.py work_dirs/ssd300_coco_bsx/epoch_24.pth --work-dir work_dirs/ssd300_coco_bsx
```

### 3.2 在 VOC 数据集上进行训练和测试

#### 3.2.1 Fast R-CNN

通过预训练的 RPN（下载权重到对应目录）先导出数据集的 proposal：

```bash
python tools/test.py configs/rpn/rpn_r50_fpn_1x_voc2012_coco_dump_train.py checkpoints/rpn_r50_fpn_1x_coco_20200218-5525fa2e.pth --work-dir work_dirs/rpn_voc2012_coco_dump_train
python tools/test.py configs/rpn/rpn_r50_fpn_1x_voc2012_coco_dump_val.py checkpoints/rpn_r50_fpn_1x_coco_20200218-5525fa2e.pth --work-dir work_dirs/rpn_voc2012_coco_dump_val
python tools/test.py configs/rpn/rpn_r50_fpn_1x_voc2012_coco_dump_test.py checkpoints/rpn_r50_fpn_1x_coco_20200218-5525fa2e.pth --work-dir work_dirs/rpn_voc2012_coco_dump_test
```

训练：

```bash
python tools/train.py configs/fast_rcnn/fast-rcnn_r50_fpn_1x_voc2012_coco.py --work-dir work_dirs/fast_rcnn_r50_fpn_1x_voc2012_coco_bs16 --cfg-options train_dataloader.batch_size=16 train_cfg.val_interval=100
```

测试：

```bash
python tools/test.py configs/fast_rcnn/fast-rcnn_r50_fpn_1x_voc2012_coco.py work_dirs/fast_rcnn_r50_fpn_1x_voc2012_coco_bs16/epoch_12.pth --work-dir work_dirs/fast_rcnn_r50_fpn_1x_voc2012_coco_bs16
```

#### 3.2.2 Faster R-CNN

训练：

```bash
python tools/train.py configs/faster_rcnn/faster-rcnn_r50_fpn_1x_voc2012_coco.py --work-dir work_dirs/faster_rcnn_r50_fpn_1x_voc2012_coco_bs16 --cfg-options train_dataloader.batch_size=16 train_cfg.val_interval=100
```

测试：

```bash
python tools/test.py configs/faster_rcnn/faster-rcnn_r50_fpn_1x_voc2012_coco.py work_dirs/faster_rcnn_r50_fpn_1x_voc2012_coco_bs16/epoch_12.pth --work-dir work_dirs/faster_rcnn_r50_fpn_1x_voc2012_coco_bs16
```

#### 3.2.3 YOLO v3

训练：

```bash
python tools/train.py configs/yolo/yolov3_d53_8xb8-ms-416_voc2012_coco.py --work-dir work_dirs/yolov3_d53_416_voc2012_coco_bs16 --cfg-options train_dataloader.batch_size=16 train_cfg.val_interval=500 optim_wrapper.optimizer.lr=0.00025
```

测试：

```bash
python tools/test.py configs/yolo/yolov3_d53_8xb8-ms-416_voc2012_coco.py work_dirs/yolov3_d53_416_voc2012_coco_bs16/epoch_231.pth --work-dir work_dirs/yolov3_d53_416_voc2012_coco_bs16
```

#### 3.2.4 SSD

训练：

```bash
python tools/train.py configs/ssd/ssd300_voc2012_coco.py --work-dir work_dirs/ssd300_voc2012_coco_bs16 --cfg-options train_dataloader.batch_size=16 train_cfg.val_interval=100 optim_wrapper.optimizer.lr=0.0005
```

测试：

```bash
python tools/test.py configs/ssd/ssd300_voc2012_coco.py work_dirs/ssd300_voc2012_coco_bs16/epoch_24.pth --work-dir work_dirs/ssd300_voc2012_coco_bs16
```

采用 batch size=8，learning rate=0.002：

训练：

```bash
python tools/train.py configs/ssd/ssd300_voc2012_coco.py --work-dir work_dirs/ssd300_voc2012_coco_bsx --cfg-options train_cfg.val_interval=100
```

测试：

```bash
python tools/test.py configs/ssd/ssd300_voc2012_coco.py work_dirs/ssd300_voc2012_coco_bsx/epoch_24.pth --work-dir work_dirs/ssd300_voc2012_coco_bsx
```
