# Configuring the loss

Every option can be passed straight to {class}`~kmnet.KMNet`, or you can build a
{class}`~kmnet.KMLoss` yourself and hand it over with `loss=`. The defaults are the settings
used in the paper.

::::{tab-set}

:::{tab-item} Through KMNet
```python
model = KMNet(
    net,
    duration_index=labtrans.cuts,
    alpha=0.9,
    sigma="auto",
    rank_weight=0.25,
    rank_penalty="softplus",
)
```
:::

:::{tab-item} With an explicit KMLoss
```python
from kmnet import KMLoss

loss = KMLoss(alpha=0.9, sigma="auto", rank_weight=0.25, rank_space="prob")
model = KMNet(net, loss=loss, duration_index=labtrans.cuts)
```
:::

::::

## Options

| Argument | Default | Values | What it does |
|---|---|---|---|
| `alpha` | `0.9` | $[0, 1]$ | Weight on the likelihood; $1-\alpha$ goes to the rank loss. |
| `sigma` | `"auto"` | float $> 0$ or `"auto"` | Softness of the pairwise penalty. `"auto"` adapts it per batch. |
| `rank_weight` | `0.25` | float | Extra multiplier $\lambda$ on the rank loss to balance scales. |
| `rank_space` | `"logit"` | `"logit"`, `"prob"` | Space in which the conditional survival scores are compared. |
| `rank_penalty` | `"softplus"` | `"softplus"`, `"exp"` | Shape $\psi$ of the ranking penalty. |
| `reduction` | `"mean"` | `"mean"`, `"sum"`, `"none"` | `KMLoss` only. |

See {doc}`method` for the formulas behind each option.

## Tuning tips

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Start with `alpha`
Lower `alpha` gives the rank term more weight. Keep it high (0.8 or above): a strong rank term
pulls predicted survival below the observed Kaplan&ndash;Meier curve.
:::

:::{grid-item-card} Then `sigma`
A small `sigma` approximates a hard 0/1 ordering; a large one gives smoother gradients.
`"auto"` is a good choice when you do not want to tune it.
:::

:::{grid-item-card} Batch size matters
The rank loss only sees pairs within a mini-batch, and its cost grows as $O(n^2)$ per batch.
64-256 balances pair coverage and memory.
:::

:::{grid-item-card} Select on calibration too
Concordance only measures ordering, so it cannot see survival curves that sit too low. When
tuning, check the integrated Brier score as well.
:::

::::
