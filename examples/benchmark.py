
import time
import torch
import torch.nn as nn
import torchtuples as tt
import numpy as np
import pandas as pd
from kmnet.model import KMNet

def make_synthetic_data(n_samples=2000):
    X = np.random.randn(n_samples, 20).astype('float32')
    h = 0.1 * np.exp(0.5 * X[:, 0])
    T = np.random.exponential(1/h)
    C = np.random.exponential(1/0.05, size=n_samples)
    time = np.minimum(T, C)
    event = (T <= C).astype('float32')
    return X, time, event

def benchmark_model(model_class, X, y, duration_index, epochs=5, batch_size=256, name="Model"):
    in_features = X.shape[1]
    out_features = len(duration_index)
    
    net = nn.Sequential(
        nn.Linear(in_features, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, out_features)
    )
    
    model = model_class(net, duration_index=duration_index)
    
    start_time = time.time()
    model.fit(X, y, batch_size, epochs, verbose=False)
    end_time = time.time()
    
    duration = end_time - start_time
    print(f"{name} Training Time ({epochs} epochs): {duration:.4f} seconds")
    return duration

def main():
    print("Generating data...")
    X, time_evt, event = make_synthetic_data(n_samples=5000)
    
    # Preprocessing
    num_durations = 50
    labtrans = KMNet.label_transform(num_durations)
    get_target = lambda df: (df['duration'].values, df['event'].values)
    df = pd.DataFrame({'duration': time_evt, 'event': event})
    y = labtrans.fit_transform(*get_target(df))
    
    print(f"Benchmarking with {len(X)} samples, {num_durations} time bins.")
    
    # Benchmark KMNet
    t_opt = benchmark_model(KMNet, X, y, labtrans.cuts, epochs=10, name="KMNet")
    
    print(f"\nTraining complete.")

if __name__ == "__main__":
    main()
