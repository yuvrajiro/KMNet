# KMNet

<div class="km-hero">
<p class="km-eyebrow"><span class="km-pill">v0.3.1</span> Deep discrete-time survival analysis · TMLR 2026 (under review)</p>
<p class="km-headline">Survival models that fit the data,<br><em>and rank it correctly.</em></p>
<p class="km-lead">KMNet trains any PyTorch network on a censored likelihood plus a
Kaplan&ndash;Meier inspired rank loss. You choose the likelihood form; KMNet supplies the
loss, the training loop and the survival curves.</p>
<div class="km-actions">
  <a class="km-btn km-btn-solid" href="quickstart.html">Get started</a>
  <a class="km-btn km-btn-outline" href="method.html">How it works</a>
  <code class="km-install"><span>$</span>pip install kmnet</code>
</div>
</div>

<div class="km-features">
<div class="km-card">
<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3 4v16h18"/><path d="M5 6h4v4h4v4h4v3h3"/></svg>
<h3>Censoring-aware likelihood</h3>
<p>A masked BCE likelihood; censored subjects count for exactly as long as they were observed.</p>
</div>
<div class="km-card">
<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 18h4V12H4z"/><path d="M10 18h4V8h-4z"/><path d="M16 18h4V4h-4z"/></svg>
<h3>Ranking, not just fitting</h3>
<p>A smooth pairwise penalty makes the subject who failed first look riskier at that time.</p>
</div>
<div class="km-card">
<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 7l-5 5 5 5"/><path d="M16 7l5 5-5 5"/><path d="M13.5 5l-3 14"/></svg>
<h3>Drop-in PyTorch</h3>
<p>Bring any <code>nn.Module</code>; TorchScript loss kernels behind the familiar pycox API.</p>
</div>
</div>

## How it works

<div class="km-steps">
<div class="km-step">
<span class="km-step-num">01</span>
<h3>Discretise time</h3>
<p>Cut follow-up into T bins with <code>KMNet.label_transform</code>.</p>
</div>
<div class="km-step">
<span class="km-step-num">02</span>
<h3>Bring a network</h3>
<p>Any <code>nn.Module</code> with T outputs gives one logit per bin.</p>
</div>
<div class="km-step">
<span class="km-step-num">03</span>
<h3>Choose the loss</h3>
<p>A censored likelihood, weighted against the rank term by <code>alpha</code>.</p>
</div>
<div class="km-step">
<span class="km-step-num">04</span>
<h3>Predict curves</h3>
<p>S(t | x) is the running product of the per-bin survival.</p>
</div>
</div>

## Quick start

```python
import torch.nn as nn
from kmnet import KMNet

# X: (n, p) float32 features; time, event: (n,) arrays
labtrans = KMNet.label_transform(20)            # 20 discrete time bins
y = labtrans.fit_transform(time, event)

net = nn.Sequential(nn.Linear(X.shape[1], 32), nn.ReLU(),
                    nn.Linear(32, labtrans.out_features))

model = KMNet(net, duration_index=labtrans.cuts,
              alpha=0.2, sigma=0.1)
model.fit(X, y, batch_size=64, epochs=20)

surv = model.predict_surv_df(X[:5])             # rows: time grid, cols: subjects
```

## What you get

<figure class="km-figure">
<svg class="km-plot" viewBox="0 0 640 300" role="img" aria-label="Schematic of three predicted survival curves, low, medium and high risk, with a dashed population Kaplan-Meier reference curve">
  <path class="axis" d="M50 20 V250 H620"/>
  <path class="ref" d="M50 30 C 180 70, 330 150, 620 186"/>
  <path class="l1" d="M50 30 H100 V46 H170 V60 H240 V76 H320 V92 H400 V110 H490 V124 H620"/>
  <path class="l2" d="M50 30 H85 V58 H150 V88 H220 V118 H290 V144 H370 V170 H450 V190 H620"/>
  <path class="l3" d="M50 30 H68 V76 H115 V120 H170 V160 H230 V192 H300 V214 H390 V230 H620"/>
  <path class="tick" d="M135 40 V52 M360 86 V98 M260 112 V124 M345 208 V220"/>
  <text x="54" y="14">S(t | x)</text>
  <text x="620" y="272" text-anchor="end">time t</text>
  <text x="44" y="34" text-anchor="end">1</text>
  <text x="44" y="254" text-anchor="end">0</text>
  <g class="key">
    <path class="l1" d="M470 40 H494"/><text x="502" y="44">low risk</text>
    <path class="l2" d="M470 58 H494"/><text x="502" y="62">medium risk</text>
    <path class="l3" d="M470 76 H494"/><text x="502" y="80">high risk</text>
    <path class="ref" d="M470 94 H494"/><text x="502" y="98">population KM</text>
  </g>
</svg>
<figcaption>Schematic: KMNet predicts a full survival curve per subject. Ticks mark censored observations.</figcaption>
</figure>

## Citation

<div class="km-cite">
<p>Deep Discrete-Time Survival Analysis with Guaranteed Monotonicity.</p>
<p class="km-cite-meta">Anonymous · Submitted to Transactions on Machine Learning Research, 2026 · under review ·
<a href="https://openreview.net/forum?id=VXIk9iY10Q">OpenReview</a> · BibTeX on the <a href="citation.html">citation page</a></p>
</div>

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
