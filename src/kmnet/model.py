
import torch
from torch import Tensor
import torch.nn.functional as F
from pycox import models
from pycox.preprocessing import label_transforms
import pandas as pd
import numpy as np
import numba
import torchtuples as tt

class SurvBase(tt.Model):
    """Base class for survival models. 
    Essentially same as torchtuples.Model, 
    """

    label_transform = label_transforms.LabTransDiscreteTime

    def __init__(self, net, loss=None, optimizer=None, device=None, duration_index=None):
        self.duration_index = duration_index
        super().__init__(net, loss, optimizer, device)
    
    @property
    def duration_index(self):
        return self._duration_index

    @duration_index.setter
    def duration_index(self, val):
        self._duration_index = val
    
    def predict_surv(self, input, batch_size=8224, numpy=None, eval_=True,
                     to_cpu=False, num_workers=0):
        """Predict the survival function for `input`.
        See `prediction_surv_df` to return a DataFrame instead.
        """
        raise NotImplementedError

    def predict_surv_df(self, input, batch_size=8224, eval_=True, num_workers=0):
        """Predict the survival function for `input` and return as a pandas DataFrame.
        """
        raise NotImplementedError

    def predict_hazard(self, input, batch_size=8224, numpy=None, eval_=True, to_cpu=False,
                       num_workers=0):
        """Predict the hazard function for `input`.
        """
        raise NotImplementedError

    def predict_pmf(self, input, batch_size=8224, numpy=None, eval_=True, to_cpu=False,
                    num_workers=0):
        """Predict the probability mass function (PMF) for `input`.
        """
        raise NotImplementedError

def pair_rank_mat(idx_durations, events, dtype='float32'):
    """Indicator matrix R with R_ij = 1{T_i < T_j and D_i = 1}."""
    idx_durations = idx_durations.reshape(-1)
    events = events.reshape(-1)
    n = len(idx_durations)
    mat = np.zeros((n, n), dtype=dtype)
    mat = _pair_rank_mat(mat, idx_durations, events, dtype)
    return mat

@numba.njit
def _pair_rank_mat(mat, idx_durations, events, dtype='float32'):
    n = len(idx_durations)
    for i in range(n):
        dur_i = idx_durations[i]
        ev_i = events[i]
        if ev_i == 0:
            continue
        for j in range(n):
            dur_j = idx_durations[j]
            ev_j = events[j]
            if (dur_i < dur_j) or ((dur_i == dur_j) and (ev_j == 0)):
                mat[i, j] = 1
    return mat

class KMNetDataset(tt.data.DatasetTuple):
    """Dataset that attaches the comparable-pairs matrix to every mini-batch.

    Each item is ``(input, (idx_durations, events, rank_mat))`` where
    ``rank_mat`` is built by :func:`pair_rank_mat` for that batch.
    """

    def __getitem__(self, index):
        input, target =  super().__getitem__(index)
        target = target.to_numpy()
        rank_mat = pair_rank_mat(*target)
        target = tt.tuplefy(*target, rank_mat).to_tensor()
        return tt.tuplefy(input, target)

# Re-exporting these for convenience if needed, though we import them from .model
# to avoid code duplication for non-optimizable parts.

@torch.jit.script
def _reduction(loss: Tensor, reduction: str = 'mean') -> Tensor:
    if reduction == 'none':
        return loss
    elif reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    raise ValueError("`reduction` is not valid. Use 'none', 'mean' or 'sum'.")

@torch.jit.script
def bce_km_loss_exclusive(
    phi: Tensor,
    idx_durations: Tensor,
    events: Tensor,
    reduction: str = 'mean'
) -> Tensor:
    """
    Censored likelihood as a masked BCE-with-logits (exclusive censoring).

    Per-bin losses are summed over each subject's at-risk bins and then
    reduced over subjects.
    """
    if phi.shape[1] <= idx_durations.max():
        raise ValueError("Network output `phi` is too small for `idx_durations`.")

    events_f = events.float()

    T = phi.shape[1]
    t = torch.arange(T, dtype=idx_durations.dtype, device=idx_durations.device)

    # alive_during[t] = 1 if t < j else 0  (exclusive)
    alive_during = (t.view(1, -1) < idx_durations.view(-1, 1)).to(phi.dtype)

    # count_mask: events -> all ones; censored -> alive_during only (stop at j)
    e_col = events_f.view(-1, 1).to(phi.dtype)
    count_mask = e_col + (1.0 - e_col) * alive_during

    # at-risk-at-start mask (left-shift alive_during with a leading 1)
    # torch.cat requires a list of tensors
    at_risk_start = torch.cat(
        [torch.ones_like(alive_during[:, :1]), alive_during[:, :-1]], dim=1
    )

    weight = count_mask * at_risk_start
    # sum over bins per subject (not a mean over all B*T entries, which would shrink
    # the loss by a factor of T and count masked-out bins in the denominator)
    loss = F.binary_cross_entropy_with_logits(phi, alive_during, weight=weight, reduction='none')
    return _reduction(loss.sum(dim=1), reduction)

@torch.jit.script
def rank_km_loss(
    phi: Tensor,
    idx_durations: Tensor,
    events: Tensor,
    rank_mat: Tensor,
    sigma: float,
    reduction: str = 'mean',
    space: str = 'logit',        # 'logit' or 'prob'
    penalty: str = 'softplus'    # 'softplus' (recommended) or 'exp'
) -> Tensor:
    """
    Kaplan–Meier inspired rank loss on the conditional survival scores.

    For each event anchor i and comparable partner k, compares the scores of
    both subjects in the anchor's event bin j_i; the penalty is averaged over
    the batch.
    """
    B, T = phi.shape
    idx = idx_durations.view(-1)

    if rank_mat.sum() == 0:
        return torch.zeros((), device=phi.device, dtype=phi.dtype)

    events_f = events.float()
    anchors = (events_f > 0).view(-1, 1)             # [B,1]

    # scores_all_at_j_i: [B,B], row i = scores of ALL subjects at j_i
    # trick: phi.T is [T,B]; indexing with idx (len B) picks rows j_i → [B,B]
    if space == 'logit':
        scores_all_at_j_i = phi.transpose(0, 1)[idx.long()]         # [B,B]
        anchor_scores = phi.gather(1, idx.view(-1, 1).long()).view(-1)  # [B]
    elif space == 'prob':
        scores_all_at_j_i = torch.sigmoid(phi).transpose(0, 1)[idx.long()]
        anchor_scores = torch.sigmoid(phi.gather(1, idx.view(-1, 1).long()).view(-1))
    else:
        raise ValueError("space must be 'logit' or 'prob'")

    margins = scores_all_at_j_i - anchor_scores.view(-1, 1)  # [B,B] = score_k(j_i) - score_i(j_i)

    if penalty == 'softplus':
        loss_mat = F.softplus(-(margins / sigma))
    elif penalty == 'exp':
        loss_mat = torch.exp(-(margins / sigma))
    else:
        raise ValueError("penalty must be 'softplus' or 'exp'")

    pair_mask = (rank_mat > 0) & anchors                       # [B,B]
    loss_mat = loss_mat * pair_mask.float()

    # average over the batch (dividing by each anchor's own pair count instead
    # made this term ~10x stronger)
    loss_per_i = loss_mat.sum(dim=1, keepdim=True) / B
    loss_per_i = loss_per_i * anchors.float()                  # only event anchors contribute

    return _reduction(loss_per_i, reduction)


class _Loss(torch.nn.Module):
    def __init__(self, reduction: str = 'mean') -> None:
        super().__init__()
        self.reduction = reduction

class _KMLoss(_Loss):
    def __init__(self, alpha: float, sigma: float, reduction: str = 'mean') -> None:
        super().__init__(reduction)
        self.alpha = alpha
        self.sigma = sigma

    @property
    def alpha(self) -> float:
        return self._alpha
    @alpha.setter
    def alpha(self, alpha: float) -> None:
        if (alpha < 0) or (alpha > 1):
            raise ValueError(f"Need `alpha` to be in [0, 1]. Got {alpha}.")
        self._alpha = alpha

    @property
    def sigma(self) -> float:
        return self._sigma
    @sigma.setter
    def sigma(self, sigma):
        # allow 'auto' (defer validation); otherwise require positive float
        if isinstance(sigma, str):
            if sigma != 'auto':
                raise ValueError(f"Unknown sigma string: {sigma!r}. Use 'auto' or a positive float.")
        else:
            if sigma <= 0:
                raise ValueError(f"Need `sigma` to be positive. Got {sigma}.")
        self._sigma = sigma


class KMLoss(_KMLoss):
    """Combined KMNet loss: censored likelihood plus Kaplan-Meier inspired rank loss.

    ``loss = alpha * likelihood + (1 - alpha) * rank_weight * rank_loss``

    Parameters
    ----------
    alpha : float, default 0.9
        Weight on the likelihood term, in ``[0, 1]``.
    sigma : float or 'auto', default 'auto'
        Softness of the pairwise ranking penalty. ``'auto'`` uses
        ``max(0.25 * std(phi[i, j_i]), 1e-3)`` for each batch.
    reduction : {'mean', 'sum', 'none'}, default 'mean'
        How per-sample losses are reduced.
    rank_space : {'logit', 'prob'}, default 'logit'
        Space in which the conditional survival scores are compared.
    rank_penalty : {'softplus', 'exp'}, default 'softplus'
        Shape of the ranking penalty.
    rank_weight : float, default 0.25
        Extra multiplier on the rank loss to balance the two scales.
    """

    def __init__(
        self,
        alpha: float = 0.9,
        sigma: float = 'auto',
        reduction: str = 'mean',
        rank_space: str = 'logit',      # 'logit' or 'prob'
        rank_penalty: str = 'softplus', # 'softplus' or 'exp'
        rank_weight: float = 0.25       # optional extra weight to balance scales
    ) -> None:
        super().__init__(alpha=alpha, sigma=sigma, reduction=reduction)
        self.rank_space = rank_space
        self.rank_penalty = rank_penalty
        self.rank_weight = rank_weight

    def forward(self, phi: Tensor, idx_durations: Tensor, events: Tensor, rank_mat: Tensor) -> Tensor:
        idx_durations = idx_durations.to(phi.device)
        events = events.to(phi.device)
        rank_mat = rank_mat.to(phi.device)

        # compute sigma if 'auto'
        if isinstance(self.sigma, str) and self.sigma == 'auto':
            with torch.no_grad():
                phi_j = phi.gather(1, idx_durations.view(-1,1).long()).squeeze(1)
                sigma_val = float((0.25 * phi_j.std()).clamp_min(1e-3).item())
        else:
            sigma_val = float(self.sigma)

        # likelihood
        base_loss = bce_km_loss_exclusive(phi, idx_durations, events, self.reduction)

        # rank loss
        rloss = rank_km_loss(
            phi, idx_durations, events, rank_mat,
            sigma=sigma_val,
            reduction=self.reduction,
            space=self.rank_space,
            penalty=self.rank_penalty
        )

        # combine with weights
        return self.alpha * base_loss + (1.0 - self.alpha) * (self.rank_weight * rloss)

class KMNet(SurvBase):
    """Discrete-time survival model trained with :class:`KMLoss`.

    The network outputs one logit ``phi_t`` per time bin; the conditional
    survival of bin ``t`` is ``sigmoid(phi_t)`` and the survival curve is
    their cumulative product. Training, callbacks and prediction follow the
    ``torchtuples.Model`` API.

    Parameters
    ----------
    net : torch.nn.Module
        Any network whose output width equals the number of time bins
        (``labtrans.out_features``).
    optimizer : torchtuples optimizer or torch.optim.Optimizer, optional
        Defaults to Adam.
    device : str or torch.device, optional
        e.g. ``'cuda'``. Defaults to CUDA if available, otherwise CPU.
    duration_index : array-like, optional
        Time grid of the bins (``labtrans.cuts``); used as the index of
        :meth:`predict_surv_df`.
    alpha, sigma, rank_space, rank_penalty, rank_weight
        Passed to :class:`KMLoss` when ``loss`` is not given.
    loss : KMLoss, optional
        A pre-built loss; overrides the loss arguments above.

    Examples
    --------
    >>> labtrans = KMNet.label_transform(20)
    >>> y = labtrans.fit_transform(time, event)
    >>> model = KMNet(net, duration_index=labtrans.cuts)
    >>> model.fit(X, y, batch_size=64, epochs=20)
    >>> surv = model.predict_surv_df(X_test)
    """
    def __init__(
        self,
        net,
        optimizer=None,
        device=None,
        duration_index=None,
        alpha: float = 0.9,
        sigma: float = 'auto',
        loss=None,
        # extra knobs for KMLoss
        rank_space: str = 'logit',      # 'logit' or 'prob'
        rank_penalty: str = 'softplus', # 'softplus' or 'exp'
        rank_weight: float = 0.25
    ):
        if loss is None:
            loss = KMLoss(
                alpha=alpha,
                sigma=sigma,
                rank_space=rank_space,
                rank_penalty=rank_penalty,
                rank_weight=rank_weight,
            )
        super().__init__(net, loss, optimizer, device, duration_index)

    def make_dataloader(self, data, batch_size, shuffle, num_workers=0):
        dataloader = super().make_dataloader(data, batch_size, shuffle, num_workers,
                                             make_dataset=KMNetDataset)
        return dataloader
    
    def make_dataloader_predict(self, input, batch_size, shuffle=False, num_workers=0):
        dataloader = super().make_dataloader(input, batch_size, shuffle, num_workers)
        return dataloader

    def predict_surv_df(self, input, batch_size=8224, eval_=True, num_workers=0, is_dataloader=None):
        """Predict survival curves as a DataFrame (rows: time grid, columns: subjects)."""
        surv = self.predict_surv(input, batch_size, True, eval_, True, num_workers, is_dataloader)
        return pd.DataFrame(surv.transpose(), self.duration_index)

    def predict_surv(self, input, batch_size=8224, numpy=None, eval_=True, to_cpu=False,
                     num_workers=0, is_dataloader=None):
        """Predict survival curves ``S(t | x)`` as an array of shape ``(n, T)``."""
        return np.cumprod(self.predict(input, batch_size, numpy, eval_, False, to_cpu, num_workers,
                            is_dataloader, torch.sigmoid), axis=1)

    def interpolate(self, sub=10, scheme='const_pdf', duration_index=None):
        """Wrap the model in pycox's ``InterpolateDiscrete`` for smoother curves.

        ``sub`` is the number of sub-steps inserted between consecutive bins.
        """
        if duration_index is None:
            duration_index = self.duration_index
        return models.interpolation.InterpolateDiscrete(self, scheme, duration_index, sub)
