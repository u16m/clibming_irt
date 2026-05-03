from __future__ import annotations

import argparse
from pathlib import Path

from .ifsc_scraper import scrape_all_to_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='IFSC component API scraper')
    parser.add_argument('--raw-dir', type=str, default='data/ifsc_raw',
                        help='スクレイピング時の生JSONキャッシュ保存先')
    parser.add_argument('--scrape-output', type=str, default='data/ifsc_full_scrape.json',
                        help='スクレイピング全件統合JSONの保存先')
    parser.add_argument('--scrape-sleep', type=float, default=1.5,
                        help='リクエスト間スリープ秒数')
    parser.add_argument('--scrape-retries', type=int, default=5,
                        help='スクレイピング時の最大リトライ回数')
    return parser


def main() -> None:
    args = build_parser().parse_args()
    scrape_all_to_file(
        raw_dir=Path(args.raw_dir),
        out_file=Path(args.scrape_output),
        sleep_seconds=args.scrape_sleep,
        max_retries=args.scrape_retries,
    )


if __name__ == '__main__':
    main()
