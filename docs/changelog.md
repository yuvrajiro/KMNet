# Changelog

## 0.3.0

- **Breaking:** `base` no longer defaults to `"nll"`. Pass `base="nll"` or `base="bce"`
  to `KMNet` (or `KMLoss`) explicitly; omitting it raises a `ValueError`.
- Documentation restyled with a quieter, minimal look and moved to
  [kmnet.statml.in](https://kmnet.statml.in/).
- Citation updated to the TMLR submission.

## 0.2.0

- New documentation site: quickstart, method, loss-configuration guide, tutorial notebook and a
  full API reference, built and deployed to GitHub Pages by GitHub Actions.
- Packaging moved to `pyproject.toml` with a single-source version (`kmnet.__version__`).
- Pushing a `v*` tag now builds the package, publishes it to PyPI and creates a GitHub Release.
- Docstrings for `KMNet`, `KMLoss` and the loss kernels.
- Minimum Python raised to 3.8.

## 0.1.1

- README and packaging metadata fixes.

## 0.1.0

- First release: `KMNet`, `KMLoss` and `KMNetDataset` with TorchScript-compiled losses.
