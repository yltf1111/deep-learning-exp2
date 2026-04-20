import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description='Plot and save loss curves from an MMDetection json log.')
    parser.add_argument('json_log', help='Path to the json log file.')
    parser.add_argument(
        '--out',
        default=None,
        help='Output image path. Defaults to <json_log_dir>/faster_rcnn_r50_coco_loss_curve.png'
    )
    parser.add_argument(
        '--metrics',
        nargs='+',
        default=None,
        help=('Metric keys to plot, for example: loss loss_cls loss_bbox. '
              'If omitted, all keys starting with "loss" in the log are used.'))
    parser.add_argument(
        '--x',
        choices=['step', 'epoch'],
        default='step',
        help='Field used on x-axis. Default: step')
    parser.add_argument(
        '--title',
        default='faster_rcnn_r50_coco_loss_curve',
        help='Figure title.')
    parser.add_argument(
        '--dpi',
        type=int,
        default=200,
        help='Saved figure DPI. Default: 200')
    return parser.parse_args()


def load_records(json_log):
    records = []
    metric_names = set()

    with open(json_log, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            records.append(record)
            metric_names.update(
                key for key in record.keys() if key.startswith('loss'))

    if not records:
        raise ValueError(f'No valid log records were found in {json_log}.')

    return records, sorted(metric_names)


def load_points(records, metric_key, x_key):
    xs = []
    ys = []

    for record in records:
        if metric_key not in record or x_key not in record:
            continue
        xs.append(record[x_key])
        ys.append(record[metric_key])

    if not xs:
        raise ValueError(
            f'No valid "{metric_key}" and "{x_key}" pairs were found in the log.')

    return xs, ys


def main():
    args = parse_args()
    json_log = Path(args.json_log)
    out_path = Path(args.out) if args.out else json_log.parent / f'{args.title}.png'
    records, available_metrics = load_records(json_log)
    metrics = args.metrics if args.metrics else available_metrics

    if not metrics:
        raise ValueError(f'No loss metrics were found in {json_log}.')

    missing_metrics = [metric for metric in metrics if metric not in available_metrics]
    if missing_metrics:
        raise ValueError(
            f'Metrics not found in log: {missing_metrics}. '
            f'Available loss metrics: {available_metrics}')

    plt.figure(figsize=(10, 6))

    for metric in metrics:
        xs, ys = load_points(records, metric, args.x)
        linewidth = 2.0 if metric == 'loss' else 1.2
        plt.plot(xs, ys, linewidth=linewidth, label=metric)

    plt.xlabel(args.x)
    plt.ylabel('loss')
    plt.title(args.title)
    plt.grid(True, linestyle='--', alpha=0.35)
    plt.legend()
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=args.dpi, bbox_inches='tight')
    print(f'Saved loss curve to: {out_path}')


if __name__ == '__main__':
    main()
