---
html_theme.sidebar_secondary.remove: true
---

# KMNet

<div class="km-hero">
<div class="km-hero-text">
<p class="km-eyebrow">DISCRETE-TIME SURVIVAL ANALYSIS · PYTORCH</p>
<p class="km-title">Survival models that fit the data and rank it correctly.</p>
<p class="km-lede">KMNet trains any PyTorch network on a censored likelihood plus a
Kaplan&ndash;Meier inspired rank loss. You choose the likelihood form; KMNet supplies the
loss, the training loop and the survival curves.</p>
<div class="km-cta">
  <a class="km-btn km-btn-primary" href="quickstart.html">Get started</a>
  <a class="km-btn km-btn-ghost" href="method.html">How it works</a>
</div>
<code class="km-install">pip install kmnet</code>
</div>
<figure class="km-hero-figure">
<svg class="km-plot" viewBox="0 0 480 300" role="img" aria-label="Three Kaplan-Meier survival curves; the high-risk curve is highlighted">
  <path class="axis" d="M40 20 V260 H470"/>
  <path class="c-muted draw" pathLength="1" d="M40 30 H80 V48 H140 V62 H200 V80 H270 V96 H340 V118 H410 V132 H470"/>
  <path class="c-muted draw d2" pathLength="1" d="M40 30 H70 V60 H120 V92 H180 V124 H240 V150 H310 V176 H380 V196 H470"/>
  <path class="c-accent draw d3" pathLength="1" d="M40 30 H56 V78 H96 V124 H140 V166 H190 V200 H250 V226 H330 V242 H470"/>
  <path class="tick" d="M120 54 V66 M300 90 V102 M222 118 V130 M280 220 V232"/>
  <text x="44" y="16">S(t | x)</text>
  <text x="470" y="282" text-anchor="end">time t</text>
</svg>
<figcaption>Predicted survival for three subjects. Ticks mark censored observations.</figcaption>
</figure>
</div>

## Why KMNet

::::{grid} 1 1 3 3
:gutter: 3

:::{grid-item-card} Likelihood that respects censoring
Choose the negative log-likelihood or the masked BCE form. Censored subjects count for
exactly as long as they were observed.
:::

:::{grid-item-card} Ranking, not just fitting
A smooth pairwise penalty makes the subject who failed first look riskier at that time,
which targets concordance directly.
:::

:::{grid-item-card} Drop-in and fast
TorchScript-compiled loss kernels and a Numba pair matrix, behind the familiar `pycox` /
`torchtuples` API.
:::

::::

## Thirty-second example

```python
import torch.nn as nn
from kmnet import KMNet

# X: (n, p) float32 features; time, event: (n,) arrays
labtrans = KMNet.label_transform(20)            # 20 discrete time bins
y = labtrans.fit_transform(time, event)

net = nn.Sequential(nn.Linear(X.shape[1], 32), nn.ReLU(),
                    nn.Linear(32, labtrans.out_features))

model = KMNet(net, duration_index=labtrans.cuts,
              base="nll",       # or "bce": required, no default
              alpha=0.2, sigma=0.1)
model.fit(X, y, batch_size=64, epochs=20)

surv = model.predict_surv_df(X[:5])             # rows: time grid, cols: subjects
```

## Where to next

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Quickstart
:link: quickstart
:link-type: doc

Simulate, discretise, train and plot, end to end.
:::

:::{grid-item-card} The method
:link: method
:link-type: doc

The model, the likelihood and both rank losses.
:::

:::{grid-item-card} Configuring the loss
:link: configuration
:link-type: doc

`base`, `alpha`, `sigma`, `rank_mode` and the rest.
:::

:::{grid-item-card} Tutorial notebook
:link: examples/KMNet_Demo
:link-type: doc

Training curves, survival plots, C-index and IBS.
:::

::::

## Citing KMNet

KMNet accompanies a paper under review at TMLR; see {doc}`citation` for the BibTeX entry.
KMNet is released under the MIT License.

```{toctree}
:hidden:
:caption: Guide

installation
quickstart
method
configuration
examples/KMNet_Demo
```

```{toctree}
:hidden:
:caption: Reference

api
faq
changelog
citation
```
