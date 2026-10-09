# Citation

The KMNet paper is currently under review. Until it is published, please cite the software:

```bibtex
@software{goswami_kmnet,
  title  = {KMNet: Discrete-Time Survival Analysis with a Kaplan-Meier Inspired Rank Loss},
  author = {Goswami, Rahul},
  url    = {https://github.com/yuvrajiro/KMNet},
  year   = {2025}
}
```

The repository includes a machine-readable `CITATION.cff`, so GitHub's
**Cite this repository** button gives the same reference.

## Acknowledgements

KMNet builds on [pycox](https://github.com/havakv/pycox) and
[torchtuples](https://github.com/havakv/torchtuples) for survival utilities and the training
loop, and on PyTorch's TorchScript for compiled loss kernels.
