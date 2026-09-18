# 引力・反発バランス改善実験

状態: **complete_A_B_C**。未完了条件を成功とは扱いません。

ブランチ: `experiments/force-balance-rescue`。基準commit: `c4dc3fb926eee5d2038771e3aef5b27f8f20ff42`。
実行コードcommit: `d03c8e7872ffdc24347ad67fc7d476e9dae58d3f`。
開始: 2026-09-17T12:47:00Z。終了: 2026-09-17T16:12:29.244106+00:00。
着手からの経過秒: 12329.24412727356。絶対終了: 2026-09-18T00:47:00Z。

## 実行コマンド
```sh
python -m umaping.force_balance --source-root /home/suzuki/Learn/UMAPing --output-root /home/suzuki/Learn/UMAPing --protocol /home/suzuki/Learn/UMAPing-force-balance-rescue/experiments/force_balance/20260917T124700Z/configs/protocol.json --device cpu --threads 1
```

## 実装上の事実
`A` はpairwise成分clipping後に近傍重みを掛けて足した引力です。`R` は学習済み反発ネット出力に既存のnegative_sample_rateとquery近傍重み総和を掛けた値です。係数の二重適用はありません。
`v = 2*((1-w)*A + w*R)` の後に現行と同じ成分別合力clippingと学習率を適用します。w=0.5はA+Rを再現します。比率は係数比であり、実効ベクトルノルム比ではありません。端点2A/2Rと係数1のA/Rを区別します。
参照軌道は固定です。時間tは最適化時間であり生物学的時間ではありません。FitGridの場を直接推論に使用せず、学習済みB_phiだけを使用します。
主指標は実際のheld-out時間群のRecall@15の等重み平均です。NDCGは既存の最終埋め込み評価と同じbinary NDCG@15です。densityは既存query_metricsの、入力空間の正解近傍を共通基準とした対数半径比歪みです。
95%区間は時間群内のpaired cell bootstrapです。training seed間変動とは別で、生物学的反復ではありません。seed 1は反発ネットの学習seedのみで、上流は共通です。今回の確認集合は研究全体で完全未使用の独立テストではありません。

## 選択
{"w": 0.5, "attraction_coefficient": 1.0, "repulsion_coefficient": 1.0, "objective": 0.11204427083333333, "scores": {"0.0": 0.103125, "0.2": 0.1052734375, "0.3333333333333333": 0.10794270833333333, "0.5": 0.11204427083333333, "0.6666666666666666": 0.11022135416666667, "0.8": 0.08424479166666667, "1.0": 0.0014322916666666668}, "rule": "mean teacher macro Recall; exact ties nearest 0.5 then smaller w", "scope": "best among seven discrete candidates; confirmation not used"}
選択規則: 7候補それぞれについてUniform/FitGridのmacro Recall平均を最大化。完全同点なら0.5に近い候補、さらに同点なら小さいw。確認集合や別seed/splitで再選択しません。

## 確認・転用結果
| split | teacher | w | scale | n | macro Recall | micro Recall | NDCG | density | zero Recall |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
|temporal0|Uniform|0.5|2|4000|0.11068|0.11068|0.11566|0.28422|0.26750|
|temporal0|Uniform|0|1|4000|0.10298|0.10298|0.10648|0.27672|0.29900|
|temporal0|Uniform|0|2|4000|0.10473|0.10473|0.10930|0.28183|0.30425|
|temporal0|Uniform|1|1|4000|0.00350|0.00350|0.00326|3.43275|0.97625|
|temporal0|Uniform|1|2|4000|0.00293|0.00293|0.00260|4.45686|0.98475|
|temporal0|FitGrid|0.5|2|4000|0.11043|0.11043|0.11639|0.28572|0.27575|
|temporal0|FitGrid|0|1|4000|0.10298|0.10298|0.10648|0.27672|0.29900|
|temporal0|FitGrid|0|2|4000|0.10473|0.10473|0.10930|0.28183|0.30425|
|temporal0|FitGrid|1|1|4000|0.00240|0.00240|0.00208|3.03761|0.98275|
|temporal0|FitGrid|1|2|4000|0.00213|0.00213|0.00172|5.13483|0.98475|
|temporal1|Uniform|0.5|2|4000|0.10928|0.10928|0.11423|0.28448|0.27350|
|temporal1|FitGrid|0.5|2|4000|0.11040|0.11040|0.11650|0.28259|0.27025|
|existing|Uniform|0.5|2|2000|0.08883|0.08883|0.09305|0.26815|0.34900|
|existing|FitGrid|0.5|2|2000|0.08930|0.08930|0.09343|0.26955|0.35050|

## 完了・未完了
完了条件数: 32。詳細は99_provenance/execution.json。

D: not run: A–C priority。E: not run: A–C priority。
Standard UMAPは今回同じ調整予算で探索していません。比較しない場合に優越性を主張しません。

## 結果の解釈と未検証事項
Dまたは改善不成立: 両teacherの確認用改善を再現できていない。
主seedの差は図表の95%区間と併読してください。density/NDCGとのトレードオフ、別seed、別splitの確認が揃わない範囲では採用を限定します。
係数が非ゼロであることだけでは実質的寄与の証拠にしません。力のノルム・寄与割合・clipping率は04_force_contributionsのCSVと図に記録します。寄与割合の分母0は欠測として明示します。
既存splitの学習seedがmetadataに無い場合は不明とします。確認用の改善、seed間再現性、split間転用はそれぞれ区別して判断します。

## 発表で使える結論（日本語3文）
1. Dまたは改善不成立: 両teacherの確認用改善を再現できていない。
2. 細胞bootstrapの区間は固定モデル下の評価であり、生物学的反復や学習seed間の不確実性とは異なる。
3. この比率は離散候補から選んだもので、参照軌道は固定し、他のseedとsplitで再調整していない。
