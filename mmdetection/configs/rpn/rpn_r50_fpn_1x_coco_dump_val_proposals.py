_base_ = './rpn_r50_fpn_1x_coco.py'

data_root = 'data/coco/'

# Run RPN on COCO val2017 and dump proposals for Fast R-CNN evaluation.
test_evaluator = dict(
    _delete_=True,
    type='DumpProposals',
    output_dir=data_root + 'proposals/',
    proposals_file='rpn_r50_fpn_1x_val2017.pkl')
