import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

wine_path = "/Users/daniele/Desktop/Code/Machine Learning/Books/dlwpt-code/data/p1ch4/tabular-wine/winequality-white.csv"
wineq_numpy = np.loadtxt(wine_path, dtype=np.float32, delimiter=";", skiprows=1)
wine = torch.tensor(wineq_numpy)

n_samples = wine.shape[0]
shuffled_indices = torch.randperm(n_samples)
n_val = int(0.2 * n_samples)

train_idx = shuffled_indices[ : -n_val ]
val_idx = shuffled_indices[-n_val : ]

X = wine[train_idx]
validation = wine[val_idx]
X_train = X[:,:-1]
y_train = X[:,-1].unsqueeze(1)
X_test = validation[:,:-1]
y_test = validation[:,-1].unsqueeze(1)

model = nn.Sequential(
            nn.Linear(11, 100),
            nn.ReLU(),
            nn.Linear(100, 1)
            )
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

def training_loop(n_epochs, optimizer, loss_fn, X_train, y_train, X_test, y_test):
    for epoch in range(1, n_epochs + 1):
        y_predict = model(X_train)
        loss = loss_fn(y_predict, y_train)

        with torch.no_grad():
            val_predict = model(X_test)
            val_loss = loss_fn(val_predict, y_test)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch < 3) or (epoch % 100 == 0):
            print(f"Epoch {epoch} Training loss: {loss.item():.4f}",
                    f" Validation loss: {val_loss.item():.4f}")

training_loop(1000, optimizer, loss_fn, X_train, y_train, X_test, y_test)
