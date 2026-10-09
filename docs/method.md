# The method

KMNet is a **discrete-time** survival model. A neural network maps covariates to one logit per
time bin; the logits define a survival curve; and the network is trained on a weighted sum of a
**censored likelihood** and a **Kaplan&ndash;Meier inspired rank loss**.

<div class="km-pipeline">
<svg viewBox="0 0 900 250" role="img" aria-label="KMNet pipeline: features to network to logits, which feed the survival curve, the likelihood loss and the rank loss">
  <path class="flow" d="M130 127 H175"/>
  <path class="flow" d="M325 127 H370"/>
  <path class="flow" d="M500 115 C530 115 525 47 555 47"/>
  <path class="flow" d="M500 139 C530 139 525 207 555 207"/>
  <path class="flow" d="M725 47 H760"/>
  <path class="flow" d="M725 207 H760"/>
  <path class="flow" d="M827 79 V175"/>
  <g class="node"><rect x="10" y="95" width="120" height="64" rx="12"/>
    <text x="70" y="122" text-anchor="middle">Features x</text><text class="sub" x="70" y="143" text-anchor="middle">(n, p)</text></g>
  <g class="node"><rect x="175" y="95" width="150" height="64" rx="12"/>
    <text x="250" y="122" text-anchor="middle">Network f_θ</text><text class="sub" x="250" y="143" text-anchor="middle">any nn.Module</text></g>
  <g class="node n-accent"><rect x="370" y="95" width="130" height="64" rx="12"/>
    <text x="435" y="122" text-anchor="middle">Logits φ</text><text class="sub" x="435" y="143" text-anchor="middle">(n, T bins)</text></g>
  <g class="node"><rect x="555" y="15" width="170" height="64" rx="12"/>
    <text x="640" y="42" text-anchor="middle">Survival S(t | x)</text><text class="sub" x="640" y="63" text-anchor="middle">∏ σ(φ_k)</text></g>
  <g class="node"><rect x="555" y="175" width="170" height="64" rx="12"/>
    <text x="640" y="202" text-anchor="middle">ℒ NLL</text><text class="sub" x="640" y="223" text-anchor="middle">censored likelihood</text></g>
  <g class="node"><rect x="760" y="15" width="135" height="64" rx="12"/>
    <text x="827" y="42" text-anchor="middle">ℒ rank</text><text class="sub" x="827" y="63" text-anchor="middle">pairwise order</text></g>
  <g class="node n-accent"><rect x="760" y="175" width="135" height="64" rx="12"/>
    <text x="827" y="202" text-anchor="middle">ℒ total</text><text class="sub" x="827" y="223" text-anchor="middle">α·NLL+(1−α)λ·rank</text></g>
</svg>
</div>

## Discrete-time survival

Cut the time axis into a grid $\tau_0 < \tau_1 < \dots < \tau_{T-1}$ and let $j_i$ be the bin in
which subject $i$ was last observed, with event indicator $\delta_i \in \{0, 1\}$. The network
outputs logits $\phi(x) \in \mathbb{R}^T$, which define the **conditional survival** of each bin

$$
s_t(x) = \sigma\big(\phi_t(x)\big) = P\big(T > \tau_t \mid T > \tau_{t-1},\, x\big),
$$

and the survival curve is their running product

$$
\boxed{S(\tau_t \mid x) = \prod_{k=0}^{t} s_k(x)}, \qquad F(\tau_t \mid x) = 1 - S(\tau_t \mid x).
$$

## The likelihood term

With **exclusive** censoring a subject contributes "survived" for every bin strictly before
$j_i$ and, if it had the event, "failed" in bin $j_i$:

$$
\ell^{\text{NLL}}_i = -\sum_{t < j_i} \log s_t(x_i) \;-\; \delta_i \log\big(1 - s_{j_i}(x_i)\big)
= \sum_{t<j_i} \operatorname{softplus}(-\phi_{i,t}) + \delta_i\, \operatorname{softplus}(\phi_{i,j_i}).
$$

`base="bce"` writes the same idea as a masked binary cross-entropy on the targets
$\mathbb 1\{t < j_i\}$ over the bins each subject was at risk for.

## The rank term

The likelihood fits each subject on its own. The rank term adds what a Kaplan&ndash;Meier
comparison would check: **if subject $i$ failed before subject $j$, then $i$ should look riskier
at that time.**

### Comparable pairs

Inside each mini-batch KMNet builds (with Numba) the indicator

$$
R_{ij} = \mathbb 1\{\delta_i = 1\} \cdot \mathbb 1\{\, j_i < j_j \;\text{ or }\; (j_i = j_j \text{ and } \delta_j = 0) \,\}.
$$

### `rank_mode="full"` (default)

Compare the two **CDFs at the earlier event time**:

$$
\Delta_{ij} = F_i(\tau_{j_i}) - F_j(\tau_{j_i}) = S_j(\tau_{j_i}) - S_i(\tau_{j_i}),
\qquad
\mathcal L_{\text{rank}} = \frac{1}{n^2} \sum_{i,j} R_{ij}\; \psi\!\left(\frac{\Delta_{ij}}{\sigma}\right).
$$

### `rank_mode="conditional"`

Compare only the **conditional survival scores in bin $j_i$** (in logit or probability space),
averaged over the comparable partners $k$ of each event anchor $i$:

$$
m_{ik} = \phi_{k,j_i} - \phi_{i,j_i},
\qquad
\mathcal L_{\text{rank}} = \frac{1}{n} \sum_{i:\,\delta_i=1} \frac{\sum_k R_{ik}\, \psi(m_{ik}/\sigma)}{\max\big(1, \sum_k R_{ik}\big)}.
$$

### The penalty $\psi$

| `rank_penalty` | $\psi(z)$ | behaviour |
|---|---|---|
| `"softplus"` (default) | $\log(1 + e^{-z})$ | grows linearly for badly ordered pairs; stable gradients |
| `"exp"` | $e^{-z}$ | DeepHit-style penalty; sharper, can explode for large violations |

$\sigma$ sets how soft the comparison is. `sigma="auto"` picks
$\sigma = \max\big(0.25\cdot \operatorname{sd}(\phi_{i,j_i}),\,10^{-3}\big)$ for every batch.

## The total loss

$$
\boxed{\mathcal L = \alpha\, \mathcal L_{\text{NLL}} + (1 - \alpha)\, \lambda\, \mathcal L_{\text{rank}}}
$$

with $\alpha \in [0, 1]$ (`alpha`) and $\lambda$ (`rank_weight`) balancing the scales of the two
terms. $\alpha = 1$ recovers a plain discrete-time hazard model.

:::{admonition} Implementation notes
:class: tip
All loss kernels are `torch.jit.script` functions, so the per-batch $n \times n$ comparisons run
without Python overhead, on CPU or GPU. Predictions take the `cumprod` of the sigmoid of the
logits, and `KMNet.interpolate` wraps pycox's `InterpolateDiscrete` for piecewise-continuous
curves.
:::
