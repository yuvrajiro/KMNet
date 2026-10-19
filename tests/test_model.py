
import unittest
import torch
import numpy as np
import torchtuples as tt
from kmnet.model import KMNet, KMLoss, bce_km_loss_exclusive, pair_rank_mat, rank_km_loss

class TestKMNet(unittest.TestCase):
    def setUp(self):
        self.batch_size = 32
        self.time_bins = 10
        self.input_dim = 5
        
        # Create dummy data
        self.phi = torch.randn(self.batch_size, self.time_bins, requires_grad=True)
        self.idx_durations = torch.randint(0, self.time_bins, (self.batch_size,))
        self.events = torch.randint(0, 2, (self.batch_size,)).float()
        
        # Create rank matrix (dummy)
        self.rank_mat = torch.randint(0, 2, (self.batch_size, self.batch_size)).float()
        
    def test_loss_forward(self):
        # Test that the loss function runs and returns a scalar
        loss_fn = KMLoss(alpha=0.5, sigma=0.1)
        loss = loss_fn(self.phi, self.idx_durations, self.events, self.rank_mat)
        
        self.assertTrue(torch.is_tensor(loss))
        self.assertEqual(loss.dim(), 0)
        print(f"Loss value: {loss.item()}")

    def test_model_prediction(self):
        # Simple Net
        net = torch.nn.Linear(self.input_dim, self.time_bins)
        
        # Instantiate model
        model = KMNet(net, loss=KMLoss())
        
        # Test prediction
        input_data = torch.randn(self.batch_size, self.input_dim)
        pred = model.predict(input_data)
        
        self.assertEqual(pred.shape, (self.batch_size, self.time_bins))
        print("Model prediction shape check passed.")

    def test_fit_and_predict_surv(self):
        rng = np.random.default_rng(0)
        n = 200
        X = rng.standard_normal((n, self.input_dim)).astype('float32')
        time = rng.exponential(10, n).astype('float32')
        event = (rng.random(n) < 0.7).astype('float32')

        labtrans = KMNet.label_transform(self.time_bins)
        y = labtrans.fit_transform(time, event)
        net = torch.nn.Linear(self.input_dim, labtrans.out_features)

        for kwargs in ({}, {'sigma': 0.1, 'rank_space': 'prob'}):
            model = KMNet(net, duration_index=labtrans.cuts, **kwargs)
            log = model.fit(X, y, batch_size=32, epochs=2, verbose=False)
            self.assertTrue(np.isfinite(log.to_pandas()['train_loss']).all())

        surv = model.predict_surv_df(X[:4])
        self.assertEqual(surv.shape, (labtrans.out_features, 4))
        self.assertTrue(((surv.values >= 0) & (surv.values <= 1)).all())
        # survival curves are non-increasing in time
        self.assertTrue((np.diff(surv.values, axis=0) <= 1e-7).all())


    def test_likelihood_is_per_subject_sum(self):
        # survived every bin before j, plus failed in bin j for events;
        # summed over bins per subject, then averaged over subjects
        sp = torch.nn.functional.softplus
        per_subject = torch.stack([
            sp(-self.phi[i, :j]).sum() + self.events[i] * sp(self.phi[i, j])
            for i, j in enumerate(self.idx_durations.tolist())
        ])
        for red, want in (('none', per_subject), ('sum', per_subject.sum()), ('mean', per_subject.mean())):
            got = bce_km_loss_exclusive(self.phi, self.idx_durations, self.events, red)
            self.assertTrue(torch.allclose(got, want, atol=1e-5), red)

    def test_conditional_rank_is_batch_mean(self):
        idx, ev = self.idx_durations.numpy(), self.events.numpy()
        R = torch.tensor(pair_rank_mat(idx, ev))
        sigma, B = 0.1, self.batch_size
        got = rank_km_loss(self.phi, self.idx_durations, self.events, R, sigma,
                           'mean', 'logit', 'softplus')
        total = 0.0
        for i in range(B):
            if ev[i] == 0:
                continue
            for k in range(B):
                if R[i, k] > 0:
                    m = self.phi[k, idx[i]] - self.phi[i, idx[i]]
                    total += torch.nn.functional.softplus(-m / sigma).item()
        self.assertAlmostEqual(got.item(), total / B / B, places=5)

    def test_defaults(self):
        loss = KMLoss()
        model = KMNet(torch.nn.Linear(self.input_dim, self.time_bins))
        for l in (loss, model.loss):
            self.assertEqual((l.alpha, l.sigma, l.rank_weight), (0.9, 'auto', 0.25))


if __name__ == '__main__':
    unittest.main()
