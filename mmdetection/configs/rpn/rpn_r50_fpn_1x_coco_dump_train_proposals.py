_base_ = './rpn_r50_fpn_1x_coco.py'

data_root = 'data/coco/'

# Run RPN on COCO train2017 and dump proposals for Fast R-CNN training.
test_dataloader = dict(
    dataset=dict(
        ann_file='annotations/instances_train2017.json',
        data_prefix=dict(img='train2017/')))

test_evaluator = dict(
    _delete_=True,
    type='DumpProposals',
    output_dir=data_root + 'proposals/',
    proposals_file='rpn_r50_fpn_1x_train2017.pkl')
