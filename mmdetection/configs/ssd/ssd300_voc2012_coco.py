_base_ = './ssd300_coco.py'

classes = (
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
)

data_root = 'data/VOCdevkit/VOC2012_coco/'
metainfo = dict(classes=classes)

model = dict(
    bbox_head=dict(num_classes=20)
)

train_dataloader = dict(
    dataset=dict(
        dataset=dict(
            data_root=data_root,
            metainfo=metainfo,
            ann_file='annotations/instances_train.json',
            data_prefix=dict(img='train/')
        )
    )
)

val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/instances_val.json',
        data_prefix=dict(img='val/')
    )
)

test_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/instances_test.json',
        data_prefix=dict(img='test/')
    )
)

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_val.json',
    metric='bbox'
)

test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_test.json',
    metric='bbox'
)
