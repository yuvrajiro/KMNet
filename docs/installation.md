# Installation

KMNet is a pure-Python package on [PyPI](https://pypi.org/project/kmnet/).

::::{tab-set}

:::{tab-item} pip
```bash
pip install kmnet
```
:::

:::{tab-item} From source
```bash
git clone https://github.com/yuvrajiro/KMNet.git
cd KMNet
pip install -e ".[dev]"
```
:::

:::{tab-item} CPU-only PyTorch
If you do not have a GPU, installing the CPU build of PyTorch first keeps the download small:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install kmnet
```
:::

::::

## Requirements

| Package | Role |
|---|---|
| Python ≥ 3.8 | |
| `torch` ≥ 1.7 | networks, TorchScript-compiled losses |
| `torchtuples` | training loop, callbacks, data tuples |
| `pycox` | label transforms, interpolation, evaluation |
| `numba` | fast construction of the comparable-pairs matrix |
| `numpy`, `pandas` | arrays and survival-curve DataFrames |

## Check the install

```python
import kmnet
print(kmnet.__version__)
```

## GPU

KMNet runs wherever your network runs. Pass `device="cuda"` when constructing the model:

```python
model = KMNet(net, duration_index=labtrans.cuts, base="nll", device="cuda")
```

## Building these docs

```bash
pip install -e ".[docs]"
sphinx-build -b html docs docs/_build/html
```
