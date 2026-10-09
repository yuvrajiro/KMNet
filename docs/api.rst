API reference
=============

.. currentmodule:: kmnet

Model
-----

.. autoclass:: KMNet
   :members: fit, predict, predict_surv, predict_surv_df, interpolate
   :show-inheritance:

Loss
----

.. autoclass:: KMLoss
   :members: forward
   :show-inheritance:

Data
----

.. autoclass:: KMNetDataset
   :show-inheritance:

Loss kernels
------------

TorchScript-compiled building blocks used by :class:`KMLoss`.

.. currentmodule:: kmnet.model

.. autofunction:: nll_discrete_surv_exclusive
.. autofunction:: bce_km_loss_exclusive
.. autofunction:: rank_km_loss
.. autofunction:: pair_rank_mat
