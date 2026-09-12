# Method

UMAPing amortizes specific, well-understood non-parametric pieces of the
UMAP algorithm into small neural networks, so that a single previously
unseen point can be embedded **independently of every other query point**,
in time roughly proportional to one forward pass plus a small number of
Euler steps. It is not a generic black-box network trained to imitate
`x -> UMAP(x)`; every learned component has a specific, narrow job, and the
rest of the computation is the same closed-form UMAP mathematics umap-learn
itself uses.

Throughout, $X = \{x_1, \dots, x_N\}$ is the fixed reference set. Training
uses all of $X$; a query $x_*$ is a previously unseen point processed alone.

## Notation: two different graphs

UMAP's fuzzy simplicial set construction produces a **directed** local
membership and then symmetrizes it. This codebase keeps the two explicitly
separate and never conflates them:

- **Directed $\mathrm{Mu}$** ($\mu_{i \to j}$): the retrieval teacher.
  Asymmetric, one row per point, used only to train the Neighborhood
  Retriever (Algorithm 2).
- **Symmetric $W$**: $W = \mathrm{Mu} + \mathrm{Mu}^\top - \mathrm{Mu} \odot
  \mathrm{Mu}^\top$ (elementwise product). Used only for the Spectral
  Encoder (Algorithm 3) and the reference mean dynamics (Algorithm 4).

`graph.py` builds both from one exact kNN pass and never lets $W$ leak into
retriever training, nor $\mathrm{Mu}$ leak into spectral/dynamics code.

For a query, the analogous *query-side* directed weights $\mu_{* \to j}$
are recomputed from scratch from exact distances at inference time (see
Algorithm 2) -- they are never symmetrized, and there is no $(N+1)$-point
graph.

## Algorithm 1 -- Overview: training and single-point embedding

**Offline (uses all of $X$):**

1. Build the kNN graph on $X$: directed $\mathrm{Mu}$, symmetric $W$
   (`graph.py`).
2. Train the Neighborhood Retriever $f_\theta, g_\psi$ against $\mathrm{Mu}$
   (Algorithm 2).
3. Train the Spectral Encoder against $W$; calibrate it into a frozen
   `x -> y0` map (Algorithm 3).
4. Integrate the reference mean UMAP dynamics from
   $Y_{\text{ref}}(0) = \text{SpectralEncoder}(X)$ to build a frozen
   reference trajectory (Algorithm 4).
5. Distill the Monte-Carlo mean repulsive field along that trajectory into
   $B_\phi(y, t)$ (Algorithm 4).

**Online, for one unseen $x_*$ (`inference.py::InferenceEngine.embed_one`):**

1. Retrieve candidates with $f_\theta$, rerank exactly, recompute
   $\mu_{* \to j}$ analytically (Algorithm 2).
2. $y_*(0) = \text{SpectralEncoder}(x_*)$ via the frozen calibration
   (Algorithm 3).
3. For each step $e$: read the (frozen) reference trajectory positions of
   the $k$ retrieved neighbors at that step's time, apply analytic
   attraction + $B_\phi$-based learned repulsion, take one Euler step.

No stage of single-point inference builds a graph, and no stage reads
another query point. The only thing "batch inference" ever does is run
this same per-point procedure in a Python loop with independent timing --
see `baselines.py::embed_all_queries`.

## Algorithm 2 -- Neighborhood Retriever

**Teacher.** For each $x_i \in X$, find its $k$ nearest reference
neighbors $K_i$ *excluding $x_i$ itself*, and define

$$\mu_{i \to j} = \exp\!\left(-\frac{\max(0,\, d(x_i, x_j) - \rho_i)}{\sigma_i}\right), \quad j \in K_i,$$

with $\rho_i, \sigma_i$ from the standard UMAP smooth-kNN construction
($\rho_i$ = nearest-neighbor distance under `local_connectivity`; $\sigma_i$
from a binary search matching $\sum_j \exp(-\max(0, d_{ij}-\rho_i)/\sigma_i)
= \log_2 k$). This is $\mathrm{Mu}$ -- the retrieval teacher, never the
symmetrized $W$.

**Model.** A DPR-style dual encoder: $q_i = f_\theta(x_i)$,
$k_j = g_\psi(x_j)$, both L2-normalized, scored by $s_{ij} = q_i^\top k_j /
\tau$.

**Training** (scalable -- never an $N \times N$ score matrix): sample a
minibatch of queries $i$; for each, sample one positive
$j^+ \sim \mu_{i, \cdot} / \sum_j \mu_{i,j}$; score against in-batch
positives of the other sampled queries plus optional random negatives;
mask out any candidate that *is* the query point itself, and (by default)
any candidate that is a true-but-unsampled $\mathrm{Mu}$-neighbor of that
query (so a real neighbor is never treated as a hard negative just because
it wasn't the one sampled this step). InfoNCE / DPR loss:

$$\mathcal{L}_{\text{ret}} = -\log \frac{\exp(s_{i,j^+})}{\sum_{l} \exp(s_{i,l})}.$$

Because $j^+$ is resampled proportional to $\mu_{i,\cdot}$ across many
steps, this converges to the desired fuzzy-weighted multi-positive
objective without ever materializing it explicitly. See
`training/retriever.py`.

**Inference.** All reference key embeddings are stored once. Candidate
retrieval is a deterministic, chunked exact inner-product search over those
stored keys (`models/retriever.py::ExactChunkedIndex`; an ANN backend such
as pynndescent is a documented, non-default drop-in behind the same
`NeighborIndex` interface -- FAISS/HNSW are never required). Default pool
size $M = \max(4k, 64)$. The $M$ candidates are **reranked by exact
Euclidean distance in the original preprocessed space**, and $\rho_*,
\sigma_*, \mu_{* \to j}$ are recomputed analytically from just those $k$
distances -- the same smooth-kNN formula as the reference graph, applied to
a single query row (`graph.py::query_fuzzy_weights`; this mirrors
`UMAP.transform()`'s own query-side semantics). The retriever therefore
only needs high **candidate recall**; it never needs to regress the final
fuzzy weights.

## Algorithm 3 -- Spectral Encoder

A SpectralNet-style pointwise encoder trained on the symmetric graph $W$.
Network output has $r = \text{embedding\_dim} + 1$ raw dimensions (2 + 1 = 3
for the main experiments). With degree $D_{ii} = \sum_j W_{ij}$ and
normalized Laplacian $L_{\text{sym}} = I - D^{-1/2} W D^{-1/2}$, the training
objective is the graph Dirichlet energy

$$\tfrac{1}{2}\sum_{ij} W_{ij} \left\lVert \frac{z_i}{\sqrt{D_{ii}}} - \frac{z_j}{\sqrt{D_{jj}}} \right\rVert^2 = \operatorname{tr}(Z^\top L_{\text{sym}} Z),$$

estimated over sampled edge minibatches (gathering only the unique nodes
touched by a batch before the MLP forward pass), plus a soft orthogonality
penalty $\lVert Z^\top Z / N - I \rVert_F^2$ on the same batch's raw outputs
so training does not collapse to a degenerate solution. Eigenvectors are
never used as regression labels.

**Reference-only post-hoc calibration**, computed once after training
(`training/spectral.py::compute_calibration`):

1. Evaluate the raw network on all of $X$ -> $R \in \mathbb{R}^{N \times r}$.
2. Exactly whiten: $T$ such that $Z = RT$ satisfies $Z^\top Z / N = I$
   (the soft penalty above only encourages this during training; this step
   makes it exact).
3. Form the small projected operator $H = Z^\top L_{\text{sym}} Z \in
   \mathbb{R}^{r \times r}$ and eigendecompose it.
4. Discard the lowest (trivial) mode, keep the next `embedding_dim` modes
   as a projection $P \in \mathbb{R}^{r \times \text{embedding\_dim}}$.
5. Center and scale $Z P$ using only reference statistics, to a scale
   suitable for the force equations below (Section 4).

The whole chain $x \to \text{network} \to T \to P \to \text{center/scale}$
is frozen and reused verbatim for a query: `x* -> neural network -> stored
linear transforms -> y*_0`, with **no reference-graph access at query
time**. `evaluation/spectral.py` compares this subspace against an exact
`scipy.sparse.linalg.eigsh` solution via principal angles -- for evaluation
only, never for training.

## Algorithm 4 -- Reference mean UMAP dynamics + repulsion-field distillation

**Low-dimensional kernel** (`umap_forces.py`), fit the same way umap-learn
fits it:

$$\Phi(y, z) = \frac{1}{1 + a\lVert y-z\rVert^{2b}} = \frac{1}{1+aq^b}, \quad q = \lVert y-z \rVert^2,$$

with $(a, b)$ from the same `spread`/`min_dist` curve-fit umap-learn uses.
Analytic directions (verified against the current umap-learn source, see
RUNTIME_CHECKS.md):

$$g_+(y,z) = \nabla_y \log \Phi(y,z) = -\frac{2ab\, q^{b-1}}{1+aq^b}(y-z), \qquad g_-(y,z) = \nabla_y \log\bigl(1-\Phi(y,z)\bigr) \approx \frac{2b}{(q+\varepsilon)(1+aq^b)}(y-z).$$

**Reference mean dynamics.** Rather than replaying stochastic per-edge SGD,
`dynamics.py` integrates the *expected* field over the whole fixed
reference set, starting from $Y_{\text{ref}}(0) = \text{SpectralEncoder}(X)$.
With $s_i = \sum_j W_{ij}$:

$$A_i(t) = \sum_j W_{ij}\, g_+\bigl(y_i(t), y_j(t)\bigr) \quad\text{(scatter-add over sparse edges, never an $N\times N$ matrix)},$$

$$B_i(t) \approx \frac{1}{M_{\text{neg}}}\sum_{c} g_-\bigl(y_i(t), y_c(t)\bigr) \quad\text{($M_{\text{neg}}$ reference points, freshly sampled per step, shared across all $i$)},$$

$$F_i(t) = A_i(t) + \text{negative\_sample\_rate} \cdot s_i \cdot B_i(t), \qquad y_i^{e+1} = y_i^e + \alpha_e F_i^e, \qquad \alpha_e = \alpha_0\left(1 - \tfrac{e}{n_{\text{steps}}}\right).$$

Step $e \in \{0,\dots,n_{\text{steps}}\}$ maps to normalized time
$t_e = e/n_{\text{steps}} \in [0,1]$ -- used consistently as the alpha
schedule's time variable, the repulsion field's time input, and the
reference-trajectory index, everywhere in the codebase. All (or regularly
spaced) checkpoints are stored as fixed reference memory
(`dynamics.py::ReferenceTrajectory`); query inference only ever *reads* from
it.

**What is learned here:** only the *global mean repulsive field*

$$B_X(y,t) = \mathbb{E}_{c \sim X}\bigl[g_-(y, y_c(t))\bigr],$$

distilled into a residual MLP $B_\phi(y,t)$ (Fourier time embedding + a few
residual blocks) via regression against freshly-sampled Monte-Carlo teacher
targets $B_{\text{target}} = \text{mean}_c\, g_-(y,t)$ at jittered
off-trajectory points, with a stop-gradient MSE loss. Attraction is never
learned.

**Single-query inference** (`inference.py::embed_one`) mirrors this exactly,
but with a single point instead of the whole reference set, and reads
(never recomputes) the frozen reference trajectory for its $k$ retrieved
neighbors:

$$v_{\text{attr}} = \sum_{j} \mu_{* \to j}\, g_+\bigl(y_*, y_j(t)\bigr), \qquad s_* = \sum_j \mu_{*\to j}, \qquad v_{\text{rep}} = \text{negative\_sample\_rate}\cdot s_* \cdot B_\phi(y_*, t),$$

$$y_*^{e+1} = y_*^e + \alpha_e\,(v_{\text{attr}} + v_{\text{rep}}),$$

using the identical $\alpha_e$ schedule. The
$\text{negative\_sample\_rate} \times s_*$ factor is applied exactly once,
at the point where the repulsion term is combined with attraction -- never
folded into $B_\phi$'s training target and also applied again at inference.
There is no query-query interaction anywhere in this loop: the only reads
are the frozen reference trajectory (indexed by the retrieved neighbor ids)
and $B_\phi$'s own weights.
