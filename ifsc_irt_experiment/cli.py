from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='IFSC bouldering IRT expansion runner')
    parser.add_argument('--data', type=str, default='data/ifsc_full/matrix.csv')
    parser.add_argument('--missing', type=str, default='zero', choices=['zero', 'nan'],
                        help='未出場の扱い: zero=0, nan=欠損値')
    parser.add_argument('--round', type=str, default='final',
                        choices=['final', 'semifinal', 'qualification', 'all'])
    parser.add_argument('--output', type=str, default='results/')
    parser.add_argument('--epochs', type=int, default=4000)
    parser.add_argument('--lr', type=float, default=0.03)
    parser.add_argument('--min_attempts', type=int, default=5)
    parser.add_argument('--use-kagglehub', action='store_true',
                        help='kagglehubでmxmlnv/ifsc-competition-climbingを取得して使用')
    return parser


def main() -> None:
    args = build_parser().parse_args()
    from .pipeline import run_experiment

    run_experiment(
        data_path=Path(args.data),
        output_dir=Path(args.output),
        missing_mode=args.missing,
        round_filter=args.round,
        epochs=args.epochs,
        learning_rate=args.lr,
        min_attempts=args.min_attempts,
        use_kagglehub=args.use_kagglehub,
    )
