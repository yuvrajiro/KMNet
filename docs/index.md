---
html_theme.sidebar_secondary.remove: true
---

# KMNet

<div class="km-hero">
<img src="_static/kmnet_banner.svg" alt="KMNet: Kaplan-Meier survival curves" class="km-hero-banner">
</div>

<p class="km-eyebrow">Deep survival analysis · PyTorch · TorchScript</p>

<p class="km-lede">
<span class="km-gradient-text"><strong>KMNet</strong></span> is a discrete-time survival model that learns
from <strong>two signals at once</strong>: the usual censored likelihood, which fits each subject's
event time, and a <strong>Kaplan&ndash;Meier inspired rank loss</strong>, which makes the predicted
survival curves put subjects in the right order. Bring any PyTorch network; KMNet supplies the
loss, the training loop and the survival predictions.
</p>

<div class="km-cta">
  <a class="km-btn km-btn-primary" href="quickstart.html">Get started &rarr;</a>
  <a class="km-btn km-btn-ghost" href="method.html">How it works</a>
  <a class="km-btn km-btn-ghost" href="https://github.com/yuvrajiro/KMNet">GitHub</a>
</div>

<div class="km-install">pip install kmnet</div>

<div class="km-stats km-reveal">
  <div class="km-stat"><b>2</b><span>loss terms: likelihood + ranking</span></div>
  <div class="km-stat"><b>JIT</b><span>TorchScript-compiled loss kernels</span></div>
  <div class="km-stat"><b>any</b><span><code>nn.Module</code> as the backbone</span></div>
  <div class="km-stat"><b>S(t|x)</b><span>full survival curves per subject</span></div>
</div>

## Why KMNet

::::{grid} 1 1 3 3
:gutter: 3

:::{grid-item-card}
:class-card: km-card-teal

<div class="km-icon">📈</div>

**Likelihood that respects censoring**
^^^
An exclusive-censoring negative log-likelihood (or its BCE twin) fits each subject's
discrete hazard, using censored subjects for exactly as long as they were observed.
:::

:::{grid-item-card}
:class-card: km-card-indigo

<div class="km-icon">🏁</div>

**Ranking, not just fitting**
^^^
A smooth pairwise penalty pushes the subject who failed first to have the lower
survival at that time, which directly targets concordance.
:::

:::{grid-item-card}
:class-card: km-card-coral

<div class="km-icon">⚡</div>

**Fast and drop-in**
^^^
Loss kernels are TorchScript-compiled and the comparable-pairs matrix is built with
Numba. Training and prediction follow the familiar `pycox` / `torchtuples` API.
:::

::::

## Thirty-second example

```python
import numpy as np, torch.nn as nn
from kmnet import KMNet

# X: (n, p) float32 features; time, event: (n,) arrays
labtrans = KMNet.label_transform(20)            # 20 discrete time bins
y = labtrans.fit_transform(time, event)

net = nn.Sequential(nn.Linear(X.shape[1], 32), nn.ReLU(),
                    nn.Linear(32, labtrans.out_features))

model = KMNet(net, duration_index=labtrans.cuts, alpha=0.2, sigma=0.1)
model.fit(X, y, batch_size=64, epochs=20)

surv = model.predict_surv_df(X[:5])             # rows: time grid, cols: subjects
```

## Where to next

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} 🚀 Quickstart
:link: quickstart
:link-type: doc
:class-card: km-card-teal

Simulate data, discretise time, train, and plot survival curves, end to end.
:::

:::{grid-item-card} 🧮 The method
:link: method
:link-type: doc
:class-card: km-card-indigo

The discrete-time model, the likelihood, and both flavours of the rank loss.
:::

:::{grid-item-card} 🎛️ Configuring the loss
:link: configuration
:link-type: doc
:class-card: km-card-coral

Every `KMLoss` knob: `alpha`, `sigma`, `base`, `rank_mode`, `rank_penalty`, and more.
:::

:::{grid-item-card} 📓 Tutorial notebook
:link: examples/KMNet_Demo
:link-type: doc
:class-card: km-card-teal

A runnable walkthrough with training curves and predicted survival functions.
:::

::::

## Citing KMNet

If KMNet helps your research, please cite it (see {doc}`citation`). KMNet is released under
the MIT License.

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
