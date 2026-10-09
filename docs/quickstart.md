# Quickstart

This page walks through the full workflow on simulated data: **simulate → discretise → define a
network → train → predict → evaluate**. Every block runs as-is.

## 1. Simulate censored survival data

The hazard depends on the first feature, and an independent exponential censoring time hides
part of the events.

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)

def make_data(n=2000):
    X = rng.standard_normal((n, 5)).astype("float32")
    T = rng.exponential(1 / (0.1 * np.exp(0.5 * X[:, 0])))   # event time
    C = rng.exponential(1 / 0.05, size=n)                     # censoring time
    time = np.minimum(T, C).astype("float32")
    event = (T <= C).astype("float32")
    return X, time, event

X, time, event = make_data()
n_train = 1500
X_train, X_test = X[:n_train], X[n_train:]
t_train, t_test = time[:n_train], time[n_train:]
e_train, e_test = event[:n_train], event[n_train:]
```

## 2. Discretise time

KMNet is a **discrete-time** model: the time axis is cut into bins and the network outputs one
logit per bin. `KMNet.label_transform` is pycox's `LabTransDiscreteTime`.

```python
from kmnet import KMNet

labtrans = KMNet.label_transform(20)                 # 20 equidistant bins
y_train = labtrans.fit_transform(t_train, e_train)   # (bin index, event)
y_test = labtrans.transform(t_test, e_test)
```

## 3. Define a network

Any `torch.nn.Module` works. Its output width must equal the number of bins.

```python
import torch.nn as nn

net = nn.Sequential(
    nn.Linear(X.shape[1], 32), nn.ReLU(),
    nn.Linear(32, 32), nn.ReLU(),
    nn.Linear(32, labtrans.out_features),
)
```

## 4. Train

```python
import torchtuples as tt

model = KMNet(
    net,
    optimizer=tt.optim.Adam(1e-3),
    duration_index=labtrans.cuts,
    base="nll",     # likelihood form: "nll" or "bce" (required, no default)
    alpha=0.2,      # weight on the likelihood; 1 - alpha goes to the rank loss
    sigma=0.1,      # softness of the pairwise ranking penalty
)

log = model.fit(
    X_train, y_train, batch_size=64, epochs=50,
    callbacks=[tt.callbacks.EarlyStopping()],
    val_data=(X_test, y_test), verbose=False,
)
log.plot();
```

:::{tip}
The rank loss compares subjects **within a mini-batch**, so very small batches give it few pairs
to work with. Batch sizes of 64-256 are a good default.
:::

## 5. Predict survival curves

```python
import matplotlib.pyplot as plt

surv = model.predict_surv_df(X_test[:5])   # index: time grid, columns: subjects

ax = surv.plot(drawstyle="steps-post", figsize=(8, 4.5))
ax.set(xlabel="time t", ylabel="S(t | x)", title="Predicted survival curves")
plt.show()
```

For smoother curves, interpolate between the bins:

```python
surv_smooth = model.interpolate(10).predict_surv_df(X_test[:5])
```

## 6. Evaluate

pycox's `EvalSurv` gives the time-dependent concordance and the integrated Brier score:

```python
from pycox.evaluation import EvalSurv
from scipy.integrate import trapezoid

surv = model.interpolate(10).predict_surv_df(X_test)
ev = EvalSurv(surv, t_test, e_test, censor_surv="km")

print("C-td :", ev.concordance_td("antolini"))
grid = np.linspace(t_test.min(), np.percentile(t_test, 90), 100)
brier = ev.brier_score(grid)                  # Brier score at each grid time
print("IBS  :", trapezoid(brier.values, brier.index) / (grid[-1] - grid[0]))
```

Next: see {doc}`method` for what the loss is optimising, and {doc}`configuration` to tune it.
