
import torch
import torch.nn as nn
import torchtuples as tt
from pycox.evaluation import EvalSurv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from kmnet.model import KMNet

def make_synthetic_data(n_samples=1000):
    # Simple synthetic data generation
    X = np.random.randn(n_samples, 5).astype('float32')
    # Hazard depends on X[:, 0]
    h = 0.1 * np.exp(0.5 * X[:, 0])
    T = np.random.exponential(1/h)
    C = np.random.exponential(1/0.05, size=n_samples)
    time = np.minimum(T, C)
    event = (T <= C).astype('float32')
    return X, time, event

def main():
    # 1. Generate Data
    X, time, event = make_synthetic_data()
    
    # 2. Preprocessing
    # Discretize time
    num_durations = 20
    labtrans = KMNet.label_transform(num_durations)
    get_target = lambda df: (df['duration'].values, df['event'].values)
    
    df = pd.DataFrame({'duration': time, 'event': event})
    y = labtrans.fit_transform(*get_target(df))
    
    # 3. Define Network
    in_features = X.shape[1]
    out_features = labtrans.out_features
    net = nn.Sequential(
        nn.Linear(in_features, 32),
        nn.ReLU(),
        nn.Linear(32, out_features)
    )
    
    # 4. Train Model
    model = KMNet(net, duration_index=labtrans.cuts)
    batch_size = 64
    epochs = 10
    
    # We need to pass the data as a tuple of (input, (duration_idx, event))
    # KMNetDataset handles the rank matrix creation internally
    train_data = tt.tuplefy(X, y)
    
    print("Training KMNet...")
    log = model.fit(X, y, batch_size, epochs, verbose=True)
    
    # 5. Prediction & Visualization
    surv = model.predict_surv_df(X[:5]) # Predict for first 5 samples
    
    plt.figure(figsize=(10, 6))
    for i, col in enumerate(surv.columns):
        plt.step(surv.index, surv[col], where="post", label=f"Sample {i}")
    
    plt.ylabel("Survival Probability")
    plt.xlabel("Time")
    plt.title("KMNet Survival Predictions")
    plt.legend()
    plt.grid(True)
    
    output_file = "survival_plot.png"
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")

if __name__ == "__main__":
    main()
