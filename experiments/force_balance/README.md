# 凍結モデルの引力・反発バランス実験

既存ブランチを変更せず、`experiments/force-balance-rescue` の独立worktreeで実装。
既存runは読み取り専用。新しい実データや上流再学習への置換は行わない。

```sh
# 新しいコードのworktreeで実行。source-rootは実成果物がある既存UMAPing。
PYTHONPATH=src /実環境のpython -m umaping.force_balance \
  --source-root /home/suzuki/Learn/UMAPing \
  --output-root "$PWD" \
  --protocol experiments/force_balance/20260917T124700Z/configs/protocol.json \
  --device cpu --threads 1
```

この依頼の絶対期限は **2026-09-18 00:47 UTC (09:47 JST)**。
再開時にprotocolの開始・期限を書き換えない。9時間で新規条件投入停止、10.5時間で計算終了。
GPUを選ぶ場合は、先に読み取り専用の `nvidia-smi` で使用状況を確認する。
実行器もGPU使用状況をprovenanceへ保存する。既存ジョブを停止しない。
実行プロセスはforegroundで、絶対期限のsignalとquery内の期限確認を使用する。

必須入力は監査CSVへ出力する。Uniform/FitGridの上流SHA256を比較し、temporal seedをmetadataで検証する。
seed 1は反発ネットのみの学習seedで、上流をseed 0と共有する。
実時間群に従って選択512・確認最大4000・転用最大2000のIDを評価前に固定する。
実環境の最大100点pilotから予算を更新する。予算削減時も確認用は全条件共通。

A: Uniform/FitGrid × 7比率およびA/Rの係数1条件。
選択結果を保存してからB: 確認集合の現行/選択/端点、利用可能ならseed 1。
C: 同じwを既存splitへ適用。各条件で同じquery IDを使う。
D/Eは自動投入しない。A–Cと図表が完了した後に残り予算から判断する。

25queryごとと中断時に途中埋め込み・実測時間を保存する。同じrun IDで再開できる。
cacheはquery ID、上流/model hash、推論コードhash、係数を照合する。
`execution.lock` が残っている場合は記載PIDの稼働状況を確認する。稼働中なら並行起動しない。
`selection.json` は再開で異なる値に書き換えない。

評価は既存のRecall@15、binary NDCG@15、density log distortionの定義を再利用。
主指標macro Recallは時間群を等重みとする。2000回の時間群内paired cell bootstrap。
バッチ推論は導入していないため、query間相互作用もない。
初期座標・検索近傍は同じ凍結モデルから決定的に計算し、参照半径をメモリ内で再利用する。
保存済み他条件の評価値や通常UMAPは現在の実行器には混ぜない。

## 検証

```sh
PYTHONPATH=src python -m pytest tests/test_force_balance.py tests/test_inference.py tests/test_umap_forces.py -q
```

synthetic fixtureは実装検証専用。実データの実験結果には使わない。
実データでの現行再現確認も、探索前に同一3queryで旧A+R式と比較する。

2026-09-17のローカル監査では実データの必要入力43件が不足。
`Final_analysis/force_balance_20260917T124700Z/README_ja.md` と
`99_provenance/input_inventory.csv` を参照。実データ実験A–Cは未実行。
