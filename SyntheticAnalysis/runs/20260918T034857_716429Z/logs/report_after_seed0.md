# 既知構造の合成データ：通常UMAP transformとUMAPing



作成: 2026-09-18T03:56:40.201482+00:00。run開始から 0.128 時間。実行済み学習seed: [np.int64(0)]。

全手法をreferenceのみで学習し、queryは未使用の点として配置した。通常UMAPはreferenceにfitしてqueryをtransformした結果であり、全点fitではない。島の面積や全体拡大は改善指標にしていない。



## 目的と事前固定

局所近傍、相対局所スケール、巻き・枝・橋の接続を評価する。無ノイズの2次元平面を50次元へ等長線形写像した機構確認であり、高難度・実データでの優位性は主張しない。

生成と評価規則は学習前に固定（`logs/pretraining_frozen_manifest.json`）。データ生成条件・重み・学習回数をtest成績に合わせて変更していない。全点ID・正解座標・構造ID・枝ID・進行位置・splitをdataに保存。



## 設定

- 内部6,000点（A:2,001、B:1,998、C:2,001）、reference 4,000／query 2,000。橋AB・BCは各240点、計480点（reference 320／query 160）追加。内部点・ID・splitは両条件と全seedで完全共通。

- A: 2.5周、半径1.2–5.6、全幅0.36。B: 7枝、全幅0.36、枝の交差なし。C: 半径4×3の楕円、正規化横位置uに対して密度∝exp(1.2u)。橋全幅0.11。

- 座標標準化・ノイズ・非線形変換なし。x=z Aᵀ、AᵀA=I。学習APIの入力のみfloat32、正解/PCA/評価はfloat64。validationとearly stoppingは使わず固定step。

- k設定15、Euclidean、min_dist=0.1、spread=1、negative_sample_rate=5。通常UMAPは500 epochs、transformは166 epochs。umap-learnのfit近傍数15はself込み（実質14）、既存UMAPingはself除外15。既存の規約差を保持して記録する。

- Retriever [512,256]→128、2,000 steps、batch256。Spectral [512,256]→3、2,000 steps、edge batch4096、初期較正scale10。reference trajectory 200 steps、MC64、alpha1、成分clip4。

- 反発ネットhidden256、4 residual blocks、time embedding32、4,000 steps、batch512、jitter0.1、Adam lr0.001。Uniform MC64／既存FitGrid256×256。両teacherで上流trajectory・Retriever・Spectral・初期state_dict・TrainingQueries列・学習条件を共有。FitGridの場自体は推論で使わず蒸留ネットを使用。

- force-balanceの保存済み採用値 w=0.5, scale=2（係数1の引力＋係数1の反発）を使用。`configs/force_balance_selection_source.json`に元結果を保存。今回のquery情報による再探索なし。

- CPU、PyTorch threads=2。seedはモデル全体の再学習seedで、同じ固定データとsplitを使用。3 seedsでもデータ生成分布の反復ではない。



## 評価定義

Recall@5/15は正解zと埋め込みyにおける同じreference候補集合へのkNNの共通要素数/k。reference行はselfを除く。withinは候補referenceを同じ構造（橋は各橋）に制限し、島分離の影響を分ける。

半径rᶻᵢ,ₖとrʸᵢ,ₖは同じ候補集合から各空間で独立に選ぶk番目の距離。s=exp(median_{i∈reference} log(rᶻᵢ,₁₅/rʸᵢ,₁₅))を手法・条件・seedにつき一つだけ求め、Eᵢ,ₖ=log(s rʸᵢ,ₖ/rᶻᵢ,ₖ)。負は圧縮、正は膨張、0は倍率補正後一致。k=5/15とwithinにも同じsを使用。群別・query別の再較正なし。相対局所スケール指標で、絶対島サイズは評価しない。

誤近傍は埋め込みkNN15のうち、正解距離が点自身の正解r15の2倍超、かつAでは角度差>π、Bでは異なる枝の辺と定義。共有分岐点から双方0.6以内の隣接枝は除外。全近傍15本を分母に率を算出する。正解kNNに存在しない非局所接続のみを数えるため、近傍境界の微小な入替えは誤接続としない。

bridge edge lossは、正解kNN15で少なくとも片端が橋である辺のうち埋め込みkNN15から消えた割合。cross bridge edge lossはさらに構造IDの異なる端点接続辺に限定。分母0は欠測。bridge_nearはこれら正解辺に関係するsource点で、橋の点だけでなく接続端の内部点も含む。



## 入力段階の健全性

|scope|components|without_bridges|unexpected_contacts|wrong_turn|wrong_branch|
|---|---|---|---|---|---|
|disconnected_all|3|3|0|0|0|
|disconnected_reference|3|3|0|0|0|
|sparse_bridge_all|1|3|0|0|0|
|sparse_bridge_reference|1|3|0|0|0|

全点グラフとreferenceグラフの双方で、disconnectedは3成分、sparse_bridgeは1成分。橋を除くと3成分。意図した端点以外の構造間接触と、事前規則で検出される巻き・枝間近道は0。橋は入力段階で切れていない。実際のUMAPing/umap-learnの学習グラフもmodelsに保存した。

PCA回復最大絶対誤差: disconnected=3.55e-14, sparse_bridge=2.84e-14。PCAはreferenceのみでfitし、回転・平行移動の較正もreferenceだけで計算。

![正解](figures/01_ground_truth_geometry.png)

![入力グラフ](figures/02_input_reference_graph.png)



## 主比較：共通内部query 2,000点、実行seed平均

|condition|method|recall5|recall15|within_recall15|log_radius15|abs_log_radius15|wrong_turn15|wrong_branch15|
|---|---|---|---|---|---|---|---|---|
|disconnected|fitgrid|0.6571|0.7869|0.7869|-0.1061|0.3346|0.0036|0.0000|
|disconnected|pca|1.0000|1.0000|1.0000|0.0000|0.0000|0.0000|0.0000|
|disconnected|umap|0.6388|0.8223|0.8223|-0.0304|0.2733|0.0000|0.0012|
|disconnected|uniform|0.6564|0.7868|0.7868|-0.1071|0.3352|0.0036|0.0000|
|sparse_bridge|fitgrid|0.6548|0.7929|0.7965|-0.0385|0.3282|0.0000|0.0000|
|sparse_bridge|pca|1.0000|1.0000|1.0000|0.0000|0.0000|0.0000|0.0000|
|sparse_bridge|umap|0.6362|0.8154|0.8158|-0.0181|0.2720|0.0005|0.0000|
|sparse_bridge|uniform|0.6556|0.7930|0.7965|-0.0387|0.3283|0.0000|0.0000|



## 観測事実：五つの問いへの回答



### disconnected

uniform − 通常UMAP: 内部Recall@15差 -0.0356、半径絶対log誤差差 +0.0619（後者は負が改善）。内部近傍は改善しなかった。島が大きいという視覚印象だけではこの結論を変更しない。

fitgrid − 通常UMAP: 内部Recall@15差 -0.0354、半径絶対log誤差差 +0.0613（後者は負が改善）。内部近傍は改善しなかった。島が大きいという視覚印象だけではこの結論を変更しない。

FitGrid − Uniform: 内部Recall@15差 +0.0002、半径誤差差 -0.0006。以下に全seedと構造別の悪化も含めて示す。



### sparse_bridge

uniform − 通常UMAP: 内部Recall@15差 -0.0193、半径絶対log誤差差 +0.0563（後者は負が改善）。内部近傍は改善しなかった。島が大きいという視覚印象だけではこの結論を変更しない。

fitgrid − 通常UMAP: 内部Recall@15差 -0.0193、半径絶対log誤差差 +0.0562（後者は負が改善）。内部近傍は改善しなかった。島が大きいという視覚印象だけではこの結論を変更しない。

FitGrid − Uniform: 内部Recall@15差 +0.0000、半径誤差差 -0.0001。以下に全seedと構造別の悪化も含めて示す。



### 橋追加による共通内部点の変化

同じIDを対応付けた sparse_bridge − disconnected。within Recallでは候補referenceも同じ構造内の同一ID集合なので、橋を候補に加える直接効果を除いて比較できる。各条件で再学習するため、差には上流構造の変化も含む。倍率sは条件ごとのreferenceから求める単一値であり、半径差にもこの全体較正差が含まれる。

|method|split|delta_recall15|delta_within_recall15|delta_abs_log_radius15|
|---|---|---|---|---|
|fitgrid|query|0.0060|0.0096|-0.0064|
|fitgrid|reference|0.0047|0.0091|-0.0017|
|pca|query|0.0000|0.0000|0.0000|
|pca|reference|0.0000|0.0000|0.0000|
|umap|query|-0.0070|-0.0065|-0.0013|
|umap|reference|-0.0062|-0.0057|0.0038|
|uniform|query|0.0062|0.0098|-0.0069|
|uniform|reference|0.0047|0.0091|-0.0017|

[全seed・構造別の対応差](metrics/paired_bridge_effect.csv)



### reference構造とquery配置の差

UniformとFitGridのreference座標は完全に同じなので、両者のreference指標差は0。両者のquery差は、この共有上流の下でteacher変更に伴う反発ネットの違いとして比較できる。通常UMAPとの比較はreference配置自体も異なる。reference→queryの平均差は異なる点集合間の記述比較であり、配置手順の因果効果を単独同定するものではない。

|condition|method|split|within_recall15|log_radius15|abs_log_radius15|
|---|---|---|---|---|---|
|disconnected|fitgrid|query|0.7869|-0.1061|0.3346|
|disconnected|fitgrid|reference|0.7983|-0.0688|0.3253|
|disconnected|pca|query|1.0000|0.0000|0.0000|
|disconnected|pca|reference|1.0000|0.0000|0.0000|
|disconnected|umap|query|0.8223|-0.0304|0.2733|
|disconnected|umap|reference|0.8253|-0.0052|0.2753|
|disconnected|uniform|query|0.7868|-0.1071|0.3352|
|disconnected|uniform|reference|0.7983|-0.0688|0.3253|
|sparse_bridge|fitgrid|query|0.7965|-0.0385|0.3282|
|sparse_bridge|fitgrid|reference|0.8074|-0.0012|0.3236|
|sparse_bridge|pca|query|1.0000|0.0000|0.0000|
|sparse_bridge|pca|reference|1.0000|0.0000|0.0000|
|sparse_bridge|umap|query|0.8158|-0.0181|0.2720|
|sparse_bridge|umap|reference|0.8197|0.0097|0.2791|
|sparse_bridge|uniform|query|0.7965|-0.0387|0.3283|
|sparse_bridge|uniform|reference|0.8074|-0.0012|0.3236|



### 接続の保持

|method|split|wrong_turn15|wrong_branch15|bridge_edge_loss15|bridge_cross_edge_loss15|bridge_near_recall15|
|---|---|---|---|---|---|---|
|fitgrid|query|0.0000|0.0000|0.1122|0.3458|0.8856|
|fitgrid|reference|0.0000|0.0000|0.1193|0.3593|0.8751|
|pca|query|0.0000|0.0000|0.0000|0.0000|1.0000|
|pca|reference|0.0000|0.0000|0.0000|0.0000|1.0000|
|umap|query|0.0005|0.0000|0.0438|0.2336|0.9470|
|umap|reference|0.0000|0.0000|0.0493|0.2667|0.9390|
|uniform|query|0.0000|0.0000|0.1122|0.3458|0.8856|
|uniform|reference|0.0000|0.0000|0.1193|0.3593|0.8751|

橋を切って島を離すだけで改善としない。橋周辺や端点での正解辺欠落を上表で併記する。入力は連結なので、ここでの欠落は入力グラフの橋切れとは区別される。



## 構造別比較（悪化も含む、seed平均）

|condition|method|group|within_recall15|log_radius15|abs_log_radius15|wrong_turn15|wrong_branch15|
|---|---|---|---|---|---|---|---|
|disconnected|fitgrid|A_spiral|0.8871|-0.0411|0.2703|0.0109|0.0000|
|disconnected|fitgrid|B_tree|0.7442|-0.0094|0.3074|0.0000|0.0000|
|disconnected|fitgrid|C_ellipse|0.7294|-0.2676|0.4259|0.0000|0.0000|
|disconnected|pca|A_spiral|1.0000|0.0000|0.0000|0.0000|0.0000|
|disconnected|pca|B_tree|1.0000|-0.0000|0.0000|0.0000|0.0000|
|disconnected|pca|C_ellipse|1.0000|-0.0000|0.0000|0.0000|0.0000|
|disconnected|umap|A_spiral|0.9253|-0.1344|0.1788|0.0000|0.0000|
|disconnected|umap|B_tree|0.7491|0.1718|0.3362|0.0000|0.0035|
|disconnected|umap|C_ellipse|0.7924|-0.1282|0.3048|0.0000|0.0000|
|disconnected|uniform|A_spiral|0.8869|-0.0411|0.2704|0.0109|0.0000|
|disconnected|uniform|B_tree|0.7442|-0.0099|0.3075|0.0000|0.0000|
|disconnected|uniform|C_ellipse|0.7291|-0.2701|0.4277|0.0000|0.0000|
|sparse_bridge|fitgrid|A_spiral|0.9068|-0.0579|0.2663|0.0000|0.0000|
|sparse_bridge|fitgrid|B_tree|0.7462|0.1113|0.3401|0.0000|0.0000|
|sparse_bridge|fitgrid|C_ellipse|0.7364|-0.1687|0.3781|0.0000|0.0000|
|sparse_bridge|fitgrid|bridge_AB|0.9217|0.1516|0.3223|0.0000|0.0000|
|sparse_bridge|fitgrid|bridge_BC|0.9250|-0.4231|0.4511|0.0000|0.0000|
|sparse_bridge|pca|A_spiral|1.0000|0.0000|0.0000|0.0000|0.0000|
|sparse_bridge|pca|B_tree|1.0000|-0.0000|0.0000|0.0000|0.0000|
|sparse_bridge|pca|C_ellipse|1.0000|0.0000|0.0000|0.0000|0.0000|
|sparse_bridge|pca|bridge_AB|1.0000|-0.0000|0.0000|0.0000|0.0000|
|sparse_bridge|pca|bridge_BC|1.0000|-0.0000|0.0000|0.0000|0.0000|
|sparse_bridge|umap|A_spiral|0.9089|-0.1051|0.1829|0.0015|0.0000|
|sparse_bridge|umap|B_tree|0.7520|0.2000|0.3230|0.0000|0.0000|
|sparse_bridge|umap|C_ellipse|0.7864|-0.1490|0.3102|0.0000|0.0000|
|sparse_bridge|umap|bridge_AB|0.9767|-0.1438|0.1599|0.0000|0.0000|
|sparse_bridge|umap|bridge_BC|0.9675|-0.4100|0.4168|0.0000|0.0000|
|sparse_bridge|uniform|A_spiral|0.9069|-0.0578|0.2661|0.0000|0.0000|
|sparse_bridge|uniform|B_tree|0.7461|0.1110|0.3403|0.0000|0.0000|
|sparse_bridge|uniform|C_ellipse|0.7364|-0.1691|0.3784|0.0000|0.0000|
|sparse_bridge|uniform|bridge_AB|0.9217|0.1517|0.3225|0.0000|0.0000|
|sparse_bridge|uniform|bridge_BC|0.9242|-0.4236|0.4516|0.0000|0.0000|

![構造別・橋なし](figures/30_disconnected_structure_metrics.png)

![構造別・橋あり](figures/30_sparse_bridge_structure_metrics.png)



## 実行した全seedの主指標

|condition|seed|method|recall15|within_recall15|abs_log_radius15|
|---|---|---|---|---|---|
|disconnected|0|pca|1.0000|1.0000|0.0000|
|disconnected|0|umap|0.8223|0.8223|0.2733|
|disconnected|0|uniform|0.7868|0.7868|0.3352|
|disconnected|0|fitgrid|0.7869|0.7869|0.3346|
|sparse_bridge|0|pca|1.0000|1.0000|0.0000|
|sparse_bridge|0|umap|0.8154|0.8158|0.2720|
|sparse_bridge|0|uniform|0.7930|0.7965|0.3283|
|sparse_bridge|0|fitgrid|0.7929|0.7965|0.3282|

seedを結果の良し悪しで選択していない。固定データ上の学習seed変動であり、3反復の有意差や一般化を強く主張しない。全点値は各condition/seedの `*_points.csv.gz`、全枝・reference/query集計は [all_summary.csv](metrics/all_summary.csv)。



## 主要図



### disconnected、seed 0（全seedは同じ構成で保存）

![10_structure](figures/disconnected/seed_0/10_structure.png)

![11_branch](figures/disconnected/seed_0/11_branch.png)

![12_progress](figures/disconnected/seed_0/12_progress.png)

![13_density](figures/disconnected/seed_0/13_density.png)

![14_error](figures/disconnected/seed_0/14_error.png)

![15_recall](figures/disconnected/seed_0/15_recall.png)

![20_progress_spiral_zoom](figures/disconnected/seed_0/20_progress_spiral_zoom.png)

![21_progress_branch_zoom](figures/disconnected/seed_0/21_progress_branch_zoom.png)



### sparse_bridge、seed 0（全seedは同じ構成で保存）

![10_structure](figures/sparse_bridge/seed_0/10_structure.png)

![11_branch](figures/sparse_bridge/seed_0/11_branch.png)

![12_progress](figures/sparse_bridge/seed_0/12_progress.png)

![13_density](figures/sparse_bridge/seed_0/13_density.png)

![14_error](figures/sparse_bridge/seed_0/14_error.png)

![15_recall](figures/sparse_bridge/seed_0/15_recall.png)

![20_progress_spiral_zoom](figures/sparse_bridge/seed_0/20_progress_spiral_zoom.png)

![21_progress_branch_zoom](figures/sparse_bridge/seed_0/21_progress_branch_zoom.png)

![22_progress_bridge_zoom](figures/sparse_bridge/seed_0/22_progress_bridge_zoom.png)



## 原因に関する仮説と限界

入力グラフが健全でも、Spectral初期化・reference trajectory・学習Retrieverの近傍誤差・蒸留場の誤差・推論更新の違いが最終配置に影響し得る。これらは原因候補であり、この比較だけで引力過多とは断定しない。UniformとFitGridの差はteacherを変えた比較だが、通常UMAPとの差には別の初期化・reference最適化も含まれる。

このデータの真の支持集合は2次元線形部分空間で、PCAはほぼ完全に回復する。従ってUMAP系の情報損失は測定できるが、高次元非線形データでの汎化や優位性は未検証。橋ありはreference点数が320多いという差もあり、橋のトポロジーだけの因果効果ではない。



## 時間・再開・完了範囲

現在までのwall time 0.128 h、記録された計算stage合計 0.095 h。stage別の実測は [timing.csv](metrics/timing.csv)。初回2条件×seed0完了後の拡張判断は `logs/seed_expansion_decision.json`（実施時）。

完成した上流stage、teacher別モデル、100 queryごとの埋め込みを再利用。反発学習は250 stepsごとにoptimizer・乱数状態を保存。Retriever/Spectralは完成stage単位で再開し、中断中のstageのみ同一seedから再計算する。絶対11.5時間で学習停止し、既存成果物を削除しない。

完了: disconnected/seed_0, sparse_bridge/seed_0。必須比較・評価CSV・PNG/PDF・本報告を保存。

今回はノイズ・非線形写像・追加橋密度・広範な設定探索は実施していない。mainへのmergeなし。



## 再実行

手順と評価式の詳細は [README](../../README.md)。作図は学習済みembeddingだけで再実行可能。