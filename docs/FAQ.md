# Frequently Asked Questions (FAQ)

## General Questions

### What is KMNet?
KMNet is a deep learning framework for discrete-time survival analysis that combines neural networks with Kaplan-Meier inspired ranking losses.

### Why use KMNet over traditional methods?
- **Better performance**: Captures non-linear effects that Cox-PH misses
- **Ranking information**: Uses both likelihood and ranking constraints
- **Fast**: JIT-compiled for 1.6x speedup
- **Flexible**: Customizable loss functions and architectures

### What types of data does KMNet work with?
KMNet works with:
- Right-censored survival data
- Time-to-event data
- Any dataset where you have features (X), event times (T), and event indicators (E)

## Technical Questions

### How do I handle time-varying covariates?
Currently, KMNet handles static covariates. For time-varying covariates, consider:
- Creating separate models for different time periods
- Using recurrent architectures (LSTM/GRU) as the base network

### Can I use custom neural network architectures?
Yes! KMNet accepts any PyTorch `nn.Module` as the base network. Just ensure the output dimension matches the number of time bins.

### How do I choose the number of time bins?
- **Too few bins**: May miss important temporal patterns
- **Too many bins**: Increases computational cost, may overfit
- **Recommendation**: Start with 10-50 bins, validate on holdout set

### What's the difference between 'nll' and 'bce' base loss?
- `nll`: Negative log-likelihood (default, generally more stable)
- `bce`: Binary cross-entropy (alternative formulation, may work better in some cases)

Both are mathematically equivalent for discrete survival analysis.

## Installation & Setup

### I'm getting import errors. What should I do?
Make sure you have all dependencies installed:
```bash
pip install torch pandas numpy torchtuples pycox numba
```

### Can I use KMNet with GPU?
Yes! Simply pass `device='cuda'` when initializing the model:
```python
model = KMNet(net, device='cuda')
```

## Performance

### How can I speed up training?
1. Use GPU (`device='cuda'`)
2. Increase batch size (if memory allows)
3. Reduce number of time bins
4. Use fewer network layers

### My model is overfitting. What should I do?
- Add dropout layers to your network
- Use L2 regularization
- Reduce network complexity
- Get more training data
- Use early stopping (included in examples)

## Research & Citation

### How do I cite KMNet?
See the [README](../README.md#-citation) for BibTeX citation.

### Where can I find the paper?
The paper is currently under review. Check the GitHub repository for updates.

### Can I use KMNet in my research?
Yes! KMNet is MIT licensed. Please cite our work if you use it.
