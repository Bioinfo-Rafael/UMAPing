# UMAPing

An **inductive, single-query approximation of UMAP**: given a fixed
reference dataset, embed one previously unseen point at a time, without
ever building a graph over query points or letting query points interact
with each other.

## 1. The idea

UMAP is powerful but transductive in spirit: `umap-learn`'s own
`.transform()` for new points still needs the fitted reference index and
re-runs a (cheap, but graph-shaped) optimization against it. This project
asks: how much of UMAP's *behavior* can be captured by a handful of small
neural networks that each amortize one specific, well-understood
non-parametric piece of the algorithm, so that embedding one new point
becomes a fixed, closed-form computation?

The answer here is a decomposition, not a black box:

```
x*
  ├── Neighborhood Retriever  ──>  N_hat(x*), mu_{*->j}      (learned)
  └── Spectral Encoder        ──>  y*_0                       (learned)

dy*/dt  =  analytic UMAP attraction   +   learned expected UMAP repulsion
        =  g_plus(...)                   B_phi(y*, t)
```

integrated forward to obtain the final `y*`. Nothing here is a generic
`x -> UMAP(x)` regression network: the retriever's only job is candidate
recall (final edge weights are always recomputed analytically from exact
distances); the spectral encoder's only job is a good initialization
(calibrated via frozen, reference-only linear algebra); attraction is
never learned at all (it's the same closed-form UMAP gradient umap-learn
uses); and the repulsion network's only job is the one non-parametric
quantity that's expensive to compute exactly at inference time -- the mean
force from the *whole* reference set pushing a point away.

See **[docs/method.md](docs/method.md)** for the full mathematical
treatment (four algorithms, every equation, exact conventions).

### Why point-wise at inference

Everything above the dashed line is fit once, offline, using **all** of the
fixed reference set $X$. At inference:

- `embed_one(x*)` is the only contract that matters: its result never
  depends on any other query point.
- No graph is ever built over query points, and no two query points ever
  interact (directly or via attention/message-passing).
- The reference trajectory (Section 4 below) is frozen: query inference
  only *reads* it, never rebuilds or perturbs it.
- "Batch" inference exists only as a Python loop over independent
  `embed_one` calls, purely for convenience/timing -- never as a shortcut
  that lets queries see each other.

### What is trained vs. what stays analytic

| Component | Trained? | Role |
|---|---|---|
| Neighborhood Retriever (dual encoder) | yes | candidate recall only; never regresses final edge weights |
| Smooth-kNN weights $\rho_*, \sigma_*, \mu_{*\to j}$ | no | recomputed analytically from exact reranked distances |
| Spectral Encoder (MLP) | yes | initialization $y_*(0)$ |
| Whitening / projection / calibration | no | closed-form linear algebra, fit once on reference data |
| Attraction $g_+$ | no | closed-form UMAP gradient |
| Repulsion field $B_\phi$ | yes | the only learned part of the force field: the mean repulsion from all of $X$ |
| Reference trajectory | no (frozen after one offline integration) | read-only memory at inference |

## 2. Datasets

All three are implemented; see [docs/method.md](docs/method.md) for the
underlying equations and `RUNTIME_CHECKS.md` for exactly how the official
sources were verified.

### COIL-20 / COIL-100

Columbia CAVE object image libraries (1,440 / 7,200 images, 20 / 100
objects, 72 poses each, grayscale / RGB, 128x128). Object id and pose are
parsed from filenames. Every 4th pose (by rotational order) is held out as
the query set, the rest is the fixed reference $X$ (~75% / 25%), so every
object class appears on both sides. Ground-truth query neighbors are always
searched within the reference set only. Features are PCA'd (default 256
dims, fit on reference only; raw flattened pixels are a config option).

### Pancreas scRNA-seq

The pancreas integration-benchmark dataset used in the scvi-tools /
scArches / scIB reference-mapping tutorials (~16,382 cells x ~19,093 genes;
`obs['tech']`, `obs['celltype']`; raw counts in `layers['counts']`). Query
technologies: `smartseq2`, `celseq2`; reference: every other technology
(`inDrop1-4`, `smarter`, `celseq`, `fluidigmc1`). Preprocessing -- HVG
selection (2,000 genes, batch-aware) and PCA (50 components) -- is fit on
the reference split only and merely *applied* to the query split; there is
no query leakage into any stage of training.

## 3. Outputs and directory structure

```
runs/<dataset>/<run_name>/
    config.yaml                  # exact config used for this run
    metadata.json                # package versions, timestamps, completed stages
    checkpoints/{retriever,spectral_encoder,repulsion_field}.pt
    memory/                      # frozen reference-only inference state
        reference_features.npy
        retriever_keys.npy
        graph_directed.npz       # Mu
        graph_symmetric.npz      # W
        spectral_calibration.npz
        reference_trajectory.npz
    cache/                       # resumability cache (prepared dataset, calibrated
                                  # reference embedding) -- not "reference memory"
                                  # in the sense above, just avoids redoing work
    metrics/
        {retriever,spectral,field,embedding}.json, per_query.csv
        {retriever,spectral,repulsion}_training.json   # loss curves, read by `analyze`
    figures/*.png
```

`runs/`, `data/`, checkpoints, and all other generated artifacts are
git-ignored -- nothing here is meant to be committed.

## 4. Baselines and metrics

Baselines/ablations (`baselines.py`, orchestrated in `pipeline.py`):
standard `umap-learn` (fit reference, `.transform()` query), spectral-only
(no flow integration), ours with oracle (exact) neighbors instead of the
learned retriever, the full method, a no-repulsion ablation, and a
repulsion-oracle diagnostic (a large Monte-Carlo repulsion estimate in
place of $B_\phi$, on a configurable query subset) that isolates field
approximation error. An optional transductive UMAP (fit on reference+query
jointly) is included for visualization only and is clearly not the
inductive target.

Metrics (`evaluation/`): retriever Recall@k / candidate Recall@M / fuzzy
weighted recall / NDCG@k / latency; spectral Rayleigh energy / orthogonality
error / projected eigenvalues / principal angles vs. an exact `scipy.eigsh`
solution; repulsion field vector MSE/RMSE/cosine similarity/magnitude error
plus vector-field plots at $t=0, 0.5, 1$; and the primary embedding metric,
query-to-reference neighborhood recall (searched against the reference
embedding only), alongside reference trustworthiness, reference-label kNN
accuracy, and query latency. Any coordinate-level comparison between two
independently-fit 2D embeddings is Procrustes-aligned first.

This README intentionally does not include a results table: metrics are
only meaningful once generated by an actual run on a real machine, not
invented ahead of time.

## 5. Expected compute behavior

No numbers are asserted here -- only where the cost actually goes and what
scales with what, so you know what to expect on your own hardware.

- **Preprocessing/graph construction** costs one exact chunked kNN pass over
  the reference set (`O(N^2 d)` in the current exact backend); this is the
  dataset-size-dependent step most likely to dominate wall-clock on
  Pancreas or COIL-100 if run on CPU only.
- **Retriever / spectral / repulsion training** are standard small-MLP
  training loops (a few thousand steps by default); a GPU helps but none of
  these networks are large. `--device auto` picks CUDA when available.
  `--device cpu` works throughout; it is simply slower. Repulsion field
  training's teacher generation runs its trajectory lookups through a
  device-resident `dynamics.TorchTrajectoryView` specifically so it doesn't
  round-trip large position arrays through host memory every step -- for
  small datasets/models, a GPU with high per-op dispatch overhead (or one
  shared with other jobs) can still be slower than CPU regardless; if you
  see that, `--device cpu` for the smaller datasets (COIL-20/100) is a
  reasonable choice.
- **Reference mean dynamics** (`flow.n_steps`, default 200) is a vectorized
  scatter/gather over graph edges and small Monte-Carlo negative samples --
  it never forms an `N x N` matrix and scales with edge count, not `N^2`.
  It runs on CPU or GPU.
- **Query inference** (`embed_one`) is deliberately cheap per point: one
  small forward pass, a chunked inner-product search, an exact rerank over
  at most `M` candidates, and `flow.n_steps` Euler updates over a 2D vector
  -- independent of how large the reference set is beyond the kNN search
  itself.

Every expensive stage checkpoints to `<run-dir>/{checkpoints,memory}/` and
is skipped on a subsequent `train`/`pipeline` call against the same
`--run-dir` (pass `--resume` to continue a partially-completed run).

## 6. Quickstart

```bash
git clone https://github.com/Bioinfo-Rafael/UMAPing.git
cd UMAPing

python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

pip install -e ".[all]"              # editable install + scRNA extras (anndata/scanpy) + dev/test deps
```

### Full worked example: COIL-20

```bash
umaping download --dataset coil20

umaping train    --config configs/coil20.yaml --run-dir runs/coil20/main --device auto --seed 0
umaping evaluate --run-dir runs/coil20/main --device auto
umaping analyze  --run-dir runs/coil20/main --device auto

# or, chained end to end:
umaping pipeline --config configs/coil20.yaml --run-dir runs/coil20/main --device auto --seed 0

# Metrics:  runs/coil20/main/metrics/{retriever,spectral,field,embedding}.json, per_query.csv
# Figures:  runs/coil20/main/figures/*.png
```

### COIL-100

```bash
umaping download --dataset coil100
umaping pipeline --config configs/coil100.yaml --run-dir runs/coil100/main --device auto --seed 0
```

### Pancreas

```bash
umaping download --dataset pancreas
umaping pipeline --config configs/pancreas.yaml --run-dir runs/pancreas/main --device auto --seed 0
```

### All experiments, sequentially

```bash
for name in coil20 coil100 pancreas; do
  umaping download --dataset "$name"
  umaping pipeline --config "configs/${name}.yaml" --run-dir "runs/${name}/main" --device auto --seed 0
done
```

### All experiments, in the background (detached from the terminal)

Useful on a remote machine where you want download + training + evaluation +
analysis for all three datasets to keep running after you disconnect:

```bash
cd /path/to/UMAPing && source .venv/bin/activate && mkdir -p logs && \
nohup bash -c 'set -e; for name in coil20 coil100 pancreas; do
  umaping download --dataset "$name"
  umaping pipeline --config "configs/${name}.yaml" --run-dir "runs/${name}/main" --device auto --seed 0 --resume
done' > logs/all_experiments.log 2>&1 &
disown
echo "Started (PID $!). Monitor with: tail -f $(pwd)/logs/all_experiments.log"
```

Check progress any time with `tail -f logs/all_experiments.log`, confirm it's
still alive with `ps -p <PID>` (the PID the command above prints, or `pgrep -f
"umaping pipeline"`), and find results the same way as the foreground
commands above (`runs/<dataset>/main/{metrics,figures}`). `--resume` is
included so this is always safe to re-run as-is if the machine reboots or the
process is killed partway through: `download` is idempotent (skips a dataset
already on disk), and each pipeline stage is independently checkpointed, so
already-completed datasets/stages are skipped rather than redone.

### Resuming an interrupted run

```bash
umaping train --config configs/coil20.yaml --run-dir runs/coil20/main --resume
```

Stages (preprocessing, graph, retriever, spectral, reference dynamics,
repulsion field) are each checkpointed independently; a run-dir that
already contains completed stages is refused unless `--resume` is passed,
so you never silently overwrite or half-mix two experiments.

### Local smoke test, no download and no real training

`configs/mock.yaml` uses a synthetic Gaussian-blob-mixture dataset
(`data/mock.py`) instead of any real data, with tiny step counts throughout,
so the entire pipeline runs to completion in a few seconds on a laptop CPU:

```bash
umaping pipeline --config configs/mock.yaml --run-dir runs/mock/main --device cpu
```

This exercises every stage end to end -- including `evaluate` and
`analyze`, which real runs only reach after training finishes -- so it is
the fastest way to confirm a change didn't break anything before running
it against real data. It is a code-correctness check only: with so few
training steps, the produced embedding/metrics/figures are not meant to be
accurate, just non-crashing.

## 7. Project layout

```
configs/{coil20,coil100,pancreas}.yaml
docs/method.md
src/umaping/
    cli.py                # umaping {download,train,evaluate,analyze,pipeline}
    config.py             # YAML-backed dataclasses
    graph.py               # directed Mu / symmetric W construction
    umap_forces.py         # Phi, g_plus, g_minus, find_ab_params
    dynamics.py            # reference mean UMAP dynamics + ReferenceTrajectory
    inference.py           # InferenceEngine.embed_one
    baselines.py           # baseline/ablation embedding runners
    pipeline.py            # stage orchestration + checkpointing/resume
    data/{coil,pancreas,preprocessing}.py
    models/{mlp,retriever,spectral,repulsion}.py
    training/{retriever,spectral,flow}.py
    evaluation/{retrieval,spectral,field,embedding,plotting}.py
    utils/{seed,io,device}.py
tests/                    # written for remote execution -- see RUNTIME_CHECKS.md
```

`baselines.py` and `pipeline.py` are the two intentional additions beyond a
literal reading of the original file tree: they keep `cli.py` a thin
argument-parsing layer while giving baseline runners and
checkpointing/resume logic their own homes.

## 8. Development notes

- Central seeding (`utils/seed.set_seed`) covers Python, NumPy, and Torch.
- Float32 throughout; sparse (`scipy.sparse`) graphs; no `N x N` dense
  matrices for retriever training or the reference dynamics.
- See `RUNTIME_CHECKS.md` for everything that can only be validated by
  actually running this on a machine with the real dependencies and data.
