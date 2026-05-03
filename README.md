# クライミングIRT

## これは何
項目反応理論でクライマーと課題のレベルを推定する

## 環境構築
```bash
mise install
uv sync
```

## IFSC公開APIを全件スクレイピング
以下のAPIフローでデータを段階的に取得します。

1. `api=index` で season / league 一覧
2. `api=season_leagues_results` で league内の大会一覧
3. `api=event_results` で大会別カテゴリ・種目リザルト
4. `api=event_full_results` で予選/準決勝/決勝を含む詳細リザルト

```bash
make scrape \
  RAW_DIR=data/ifsc_raw \
  SCRAPE_OUTPUT=data/ifsc_full_scrape.json \
  SCRAPE_SLEEP=1.5 \
  SCRAPE_RETRIES=5
```

実装上の方針:
- リクエスト間にsleepを入れる
- 取得済みJSONは`--raw-dir`へキャッシュして再取得しない
- 失敗時は指数バックオフでリトライする
- `User-Agent` を明示する

## IRT実験の使い方
データを配置。`--data` にCSVを指定する（`athlete,event,year,round,problem,topped`列を含む）。

```shell
uv run python -m ifsc_irt_experiment.cli \
  --data data/IFSC_worldcup.csv \
  --round final \
  --missing zero \
  --output results/
```
