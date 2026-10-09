# Configuring the loss

Every option can be passed straight to {class}`~kmnet.KMNet`, or you can build a
{class}`~kmnet.KMLoss` yourself and hand it over with `loss=`.

::::{tab-set}

:::{tab-item} Through KMNet
```python
model = KMNet(
    net,
    duration_index=labtrans.cuts,
    alpha=0.2,
    sigma=0.1,
    base="nll",
    rank_mode="full",
    rank_penalty="softplus",
    rank_weight=1.0,
)
```
:::

:::{tab-item} With an explicit KMLoss
```python
from kmnet import KMLoss

loss = KMLoss(alpha=0.5, sigma="auto", base="bce",
              rank_mode="conditional", rank_space="prob")
model = KMNet(net, loss=loss, duration_index=labtrans.cuts)
```
:::

::::

## Options

| Argument | Default | Values | What it does |
|---|---|---|---|
| `alpha` | `0.2` in `KMNet`, `0.5` in `KMLoss` | $[0, 1]$ | Weight on the likelihood; $1-\alpha$ goes to the rank loss. |
| `sigma` | `0.1` | float $> 0$ or `"auto"` | Softness of the pairwise penalty. `"auto"` adapts it per batch. |
| `base` | `"nll"` | `"nll"`, `"bce"` | Form of the likelihood term. |
| `rank_mode` | `"full"` | `"full"`, `"conditional"` | Compare CDFs at the earlier event time, or conditional scores in that bin. |
| `rank_space` | `"logit"` | `"logit"`, `"prob"` | Score space for `rank_mode="conditional"`. |
| `rank_penalty` | `"softplus"` | `"softplus"`, `"exp"` | Shape $\psi$ of the ranking penalty. |
| `rank_weight` | `1.0` | float | Extra multiplier $\lambda$ on the rank loss to balance scales. |
| `reduction` | `"mean"` | `"mean"`, `"sum"`, `"none"` | `KMLoss` only. |

See {doc}`method` for the formulas behind each option.

## Tuning tips

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} 🎚️ Start with `alpha`
:class-card: km-card-teal
Sweep `alpha` over `{0.1, 0.2, 0.5, 0.8}`. Lower values put more weight on ordering subjects
(discrimination); higher values put more on fitting each subject's event time.
:::

:::{grid-item-card} 🌫️ Then `sigma`
:class-card: km-card-indigo
A small `sigma` approximates a hard 0/1 ordering; a large one gives smoother gradients.
`"auto"` is a good choice when you do not want to tune it.
:::

:::{grid-item-card} 📦 Batch size matters
:class-card: km-card-coral
The rank loss only sees pairs within a mini-batch, and its cost grows as $O(n^2)$ per batch.
64-256 balances pair coverage and memory.
:::

:::{grid-item-card} 🧱 Number of bins
:class-card: km-card-teal
10-50 bins is typical. Too few blurs the timing; too many leaves bins with no events.
`KMNet.label_transform(n, scheme="quantiles")` spreads bins by event-time quantiles.
:::

::::
