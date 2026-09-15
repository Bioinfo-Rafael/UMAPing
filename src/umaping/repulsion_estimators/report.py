"""保存済み数値から日本語reportとmatplotlib PNG/SVGを生成。"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yaml
from umaping.comparison.reporting import markdown_table
from .benchmark import write_json


def render(run, metadata, estimator_rows, trained_rows, destination=None):
    run = Path(run)
    dest = Path(destination) if destination else run
    if destination:
        dest.mkdir(parents=True, exist_ok=False)
        (dest/'figures').mkdir()
    frame, trained = pd.DataFrame(estimator_rows), pd.DataFrame(trained_rows)
    training_config = yaml.safe_load((run/'config/training.yaml').read_text())
    good = frame[frame.status == 'success'] if len(frame) else frame
    models = trained[trained.status == 'success'] if len(trained) else trained
    skips = []
    def save(fig, name):
        fig.tight_layout()
        for ext in ('png','svg'):
            path = dest/'figures'/f'{name}.{ext}'
            if path.exists():
                raise FileExistsError(f'No overwrite: {path}')
            fig.savefig(path, dpi=220, bbox_inches='tight')
        plt.close(fig)
    def bars(data, column, name, ylabel, log=False):
        if column not in data or not len(data) or not data[column].notna().any():
            skips.append(name+': measured values unavailable')
            return
        valid = data[data[column].notna()]
        fig, ax = plt.subplots(figsize=(max(7, .8*len(valid)),4.6))
        ax.bar(valid.method, valid[column], color='#416b8c')
        ax.tick_params(axis='x', rotation=45)
        for label in ax.get_xticklabels():
            label.set_ha('right')
        ax.set_ylabel(ylabel)
        if log and (valid[column] > 0).all():
            ax.set_yscale('log')
        save(fig, name)
    bars(good,'rmse','estimator_rmse_vs_method','Exact-field RMSE (lower is better)',True)
    stochastic = good[good.stochastic == True] if 'stochastic' in good else good.iloc[:0]
    bars(stochastic,'variance','estimator_variance_vs_method','Teacher variance (finite repetitions)',True)
    bars(good,'cosine','estimator_cosine_vs_method','Cosine similarity (nonzero vectors only)')
    bars(good,'magnitude_error','estimator_magnitude_error','Absolute magnitude error',True)
    bars(good,'ess','importance_ess','Effective sample size out of 64')
    def pareto(data, time_column, name, xlabel):
        if len(data) == 0 or time_column not in data:
            skips.append(name+': measured values unavailable')
            return
        fig, ax = plt.subplots(figsize=(8,5))
        valid = data[(data[time_column] > 0) & data.rmse.notna()]
        for row in valid.to_dict('records'):
            ax.scatter(row[time_column], row['rmse'], label=row['method'])
        ax.legend(loc='upper left', bbox_to_anchor=(1.02,1), fontsize=8, frameon=False)
        ax.set_xscale('log')
        if len(valid) and (valid.rmse > 0).all():
            ax.set_yscale('log')
        ax.set(xlabel=xlabel, ylabel='Exact-field RMSE (lower is better)')
        save(fig,name)
    pareto(good,'runtime_seconds_per_query','estimator_runtime_vs_rmse','Teacher seconds/query (build cost reported separately)')
    # 同一networkのforward差はnoiseが支配的なので主Paretoは学習wall-clock費用。
    pareto(models,'train_seconds','final_runtime_accuracy_pareto','Training seconds, excluding validation and teacher build')
    weight_paths = sorted((run/'estimator_benchmark').glob('*_weights.npy'))
    if weight_paths:
        fig, ax = plt.subplots(figsize=(8,4.6))
        for path in weight_paths:
            values = np.load(path)
            positive = values[np.isfinite(values) & (values > 0)]
            if len(positive):
                ax.hist(np.log10(positive), bins=60, density=True, histtype='step', label=path.stem.removesuffix('_weights'))
        ax.set(xlabel='log10 importance weight (all saved samples)', ylabel='Density')
        ax.legend(fontsize=8)
        save(fig,'importance_weight_distribution')
    else:
        skips.append('importance_weight_distribution: no successful IS weights')
    for column, name, ylabel in [('loss','training_loss_curves','Training loss (noisy teacher)'),
                                  ('rolling_std','training_loss_rolling_std','Rolling training loss std'),
                                  ('rmse','exact_validation_rmse','Exact validation RMSE'),
                                  ('cosine','exact_validation_cosine','Exact validation cosine'),
                                  ('magnitude_error','exact_validation_magnitude_error','Exact validation magnitude error')]:
        fig, ax = plt.subplots(figsize=(8,4.6))
        found = False
        plotted_values = []
        for path in sorted((run/'repulsion_training').glob('*/'+('losses.csv' if column in ('loss','rolling_std') else 'validation.csv'))):
            try:
                history = pd.read_csv(path)
            except pd.errors.EmptyDataError:
                continue
            if column in history and history[column].notna().any():
                ax.plot(history.step, history[column], label=path.parent.name, alpha=.8)
                plotted_values.extend(history[column].dropna().tolist())
                found = True
        if found:
            ax.set(xlabel='Training step (same query sequence and initialization)', ylabel=ylabel)
            if column != 'cosine' and min(plotted_values) > 0 and max(plotted_values)/min(plotted_values) >= 20:
                ax.set_yscale('log')
            ax.legend(fontsize=8)
            save(fig,name)
        else:
            plt.close(fig)
            skips.append(name+': training not completed')
    lines = ['# Embryo repulsion teacher比較', '',
             f"dataset: `{metadata['dataset']}`、N={metadata['n_reference']}、入力次元={metadata['input_dim']}、trajectory shape={metadata['trajectory_shape']}。",
             '設定・凍結入力のSHA-256・計算環境はconfig/manifest.json、元設定はconfig/source_config.yaml、今回の共通学習設定はconfig/training.yamlに保存。',
             f"checkpoint数={len(metadata['trajectory_times'])}、seed={metadata['seed']}、device={metadata['device']}。",
             'RepulsionField・共通学習設定: '+json.dumps(training_config['repulsion'], ensure_ascii=False),
             '既存kernelの設定: '+json.dumps(metadata['force'], ensure_ascii=False),
             '## 推定器単体', '', markdown_table(frame[[c for c in ('method','status','bias','variance','mse','rmse','cosine','magnitude_error','runtime_seconds_per_query','build_seconds','score_seconds_per_query','ess','max_weight','reason') if c in frame]]), '',
             'biasは有限反復の平均による推定で、真のbiasとは異なる。bias_squared_debiasedは反復分散補正値（負値も保持）。決定論的手法のvariance=0は反復乱数がないことだけを意味する。',
             '## Learned B_phi', '', markdown_table(trained[[c for c in ('method','status','rmse','cosine','magnitude_error','best_validation_rmse','reason') if c in trained]]), '',
             '学習安定性・収束・費用:', '', markdown_table(trained[[c for c in ('method','train_seconds','rolling_std_mean','late_loss_cv','first_step_half_initial_rmse','seconds_to_half_initial_rmse') if c in trained]]), '',
             '同一モデル初期値、stepごとに再現可能なanchor・連続t・Gaussian jitter、同一optimizer/settingsを使用。exact oracleは評価とteacher variant選択だけに使用。学習targetには使用していない。',
             'lossは既存どおり座標平均MSE（row-mass有効時は元の重み付き和）。報告のexact MSEはベクトル誤差ノルム二乗の平均であり、2Dでは座標平均MSEの2倍。',
             '## 解釈', '']
    reference = good[good.method == 'uniform_mc'] if len(good) else good
    if len(reference):
        base = reference.iloc[0]
        lines.append(f"1. Uniform-64: RMSE={base.rmse:.6g}、variance={base.variance:.6g}。exact場のRMS={base.exact_signal_rms:.6g}、RMSE/信号RMS={base.normalized_rmse:.6g}（1なら誤差が信号RMSと同規模）。")
        for number, method in [(2,'dual_raw_is'),(3,'dual_hub_is'),(4,'dual_topl')]:
            row = good[good.method == method]
            comparator = good[good.method == 'dual_raw_is'] if number == 3 else reference
            if len(row) and len(comparator):
                value, comparison = row.iloc[0], comparator.iloc[0]
                lines.append(f"{number}. {method}: variance={value.variance:.6g}、{comparison.method}との差={value.variance-comparison.variance:.6g}。RMSE={value.rmse:.6g}。")
                if method == 'dual_topl':
                    for baseline in ('dual_raw_is','dual_hub_is'):
                        compare = good[good.method == baseline]
                        if len(compare):
                            other = compare.iloc[0]
                            lines.append(f"Top-L−{baseline}: variance差={value.variance-other.variance:.6g}、RMSE差={value.rmse-other.rmse:.6g}。名目64-forceは同じだがscore費用も含む秒/queryは{value.runtime_seconds_per_query:.6g} / {other.runtime_seconds_per_query:.6g}。")
            else:
                lines.append(f'{number}. {method}: 比較不能。失敗理由を表に保持。')
    for number, family in [(5,'barnes_hut'),(6,'fit_grid')]:
        selected = metadata.get('selected',{}).get(family)
        row = good[good.method == selected['name']] if selected and len(good) else pd.DataFrame()
        lines.append(f"{number}. {family}: "+(f"選択={selected['name']}、RMSE={row.iloc[0].rmse:.6g}、秒/query={row.iloc[0].runtime_seconds_per_query:.6g}。" if len(row) else '利用可能な測定なし。'))
        if len(row) and len(reference):
            lines.append(f"Uniform比のruntime={row.iloc[0].runtime_seconds_per_query/reference.iloc[0].runtime_seconds_per_query:.6g}倍。構築費用は別途表に記録し、同一64-force費用とは見なさない。")
    if len(models):
        best = models.loc[models.rmse.idxmin()]
        lines.append(f"7. 同じvalidation上の最小final RMSEは{best.method}: {best.rmse:.6g}。単一seedの比較であり一般的優越性は未検証。")
        frontier = models[[not ((models.train_seconds <= r.train_seconds) & (models.rmse <= r.rmse) & ((models.train_seconds < r.train_seconds) | (models.rmse < r.rmse))).any() for r in models.itertuples()]]
        lines.append('8. 学習時間と最終RMSEのPareto候補: '+', '.join(frontier.method)+'。次の全pipeline候補は最小RMSEの手法だが、OOS性能改善はこのfield評価だけでは確定しない。')
    else:
        lines += ['7. 実データのB_phi学習比較は未完了。最良teacherは未判定。','8. 次の全pipelineではUniform-64を維持し、実測完了後に判断する。']
    lines += ['', '## 制約・異常',
              'BH/gridはcheckpoint間の場の線形補間。厳密oracleは点位置を補間してから非線形forceを計算するため、theta→0でも連続tの時間近似誤差は残る。',
              'BHは近接cellを再帰し、遠方cellでは質量×centroidのclip済みforce。個別点clipの完全再現は葉だけ。GridはCIC・線形FFT・双線形補間による近似で、範囲外queryは明示的な失敗としwrapしない。',
              'theta/gridの選択とネットワーク評価に同じ固定validationを使う探索実験。独立test、複数学習seed、OOS embedding評価は今後の課題。',
              'モデルforward時間は同一architectureのbatch計測であり、query単発latencyとは異なる。teacher初期構築、Dual Q/K cache、hubness構築は別計測。',
              'Low ESS/巨大weightが有限なら主比較に残す。support underflow、NaN、範囲外queryは失敗として記録する。']
    lines += ['- '+issue for issue in metadata.get('phase_a_issues',[])]
    if len(trained) and 'reason' in trained:
        lines += ['- '+str(row.method)+': '+str(row.reason) for row in trained[trained.status != 'success'].itertuples()]
    lines += ['- '+skip for skip in skips]
    (dest/'report.md').write_text('\n\n'.join(lines)+'\n')
    write_json(dest/'figure_status.json', dict(skipped=skips))


def main():
    p = argparse.ArgumentParser(description='保存値だけから新しい出力先へ図・日本語reportを再生成')
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    meta = json.loads((args.run_dir/'config/manifest.json').read_text())
    rows = json.loads((args.run_dir/'metrics/estimator_metrics.json').read_text())
    path = args.run_dir/'metrics/trained_fields.json'
    trained = json.loads(path.read_text()) if path.exists() else []
    render(args.run_dir, meta, rows, trained, args.output_dir)


if __name__ == '__main__':
    main()
