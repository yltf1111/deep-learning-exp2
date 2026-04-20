_base_ = './rpn_r50_fpn_1x_voc2012_coco.py'

data_root = 'data/VOCdevkit/VOC2012_coco/'

test_dataloader = dict(
    dataset=dict(
        ann_file='annotations/instances_test.json',
        data_prefix=dict(img='test/')
    )
)

test_evaluator = dict(
    _delete_=True,
    type='DumpProposals',
    output_dir=data_root + 'proposals/',
    proposals_file='rpn_test.pkl'
)
