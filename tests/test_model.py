
import unittest
import torch
import numpy as np
import torchtuples as tt
from kmnet.model import KMNet, KMLoss

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

if __name__ == '__main__':
    unittest.main()
