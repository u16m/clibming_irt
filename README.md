# クライミングIRT

## これは何
項目反応理論でクライマーと課題のレベルを推定する

## 環境構築
```bash
mise install
uv sync
```

データを配置。データの場所はmain.pyに書く。
formatは
`NAME, Problem_0, Problem_1, ...`
という形で記載。
トライ数にかかわらず、完登で1, 未登で0とする。

## 使い方
```shell
uv run python main.py
```