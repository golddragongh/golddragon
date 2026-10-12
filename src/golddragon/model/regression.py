import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt

class LinearRegression(nn.Module):

    # a.)
    def __init__(self, learning_rate=0.01, epochs=1000):
        super().__int__()
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        # Tells model that this should be tracked as a parameter
        self.w0 = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        self.w1 = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        self.loss = nn.MSELoss()
        self.w0_history = []
        self.w1_history = []
        self.loss_history = []
        self.test_r2 = None
        self.fitted = False

    # b.)
    def forward(self, x):
        return self.w0 + self.w1 * x

    # c.)
    def fit(self, x_train, y_train, x_test, y_test, device="gpu"):
        # Convert to torch tensors if not already
        if not isinstance(x_train, torch.Tensor):
            x_train = torch.tensor(x_train, dtype=torch.float64)
        if not isinstance(y_train, torch.Tensor):
            y_train = torch.tensor(y_train, dtype=torch.float64)
        if not isinstance(x_test, torch.Tensor):
            x_test = torch.tensor(x_test, dtype=torch.float64)
        if not isinstance(y_test, torch.Tensor):
            y_test = torch.tensor(y_test, dtype=torch.float64)

        # Train
        for epoch in range(self.epochs):
            self.optimizer.zero_grad()  # Zero the gradients
            y_pred = self.forward(x_train)  # Forward pass
            loss = self.loss(y_pred, y_train)  # Compute loss
            loss.backward()  # Backward pass
            self.optimizer.step()  # Update parameters

            # Store history for analysis
            self.w0_history.append(self.w0.item())
            self.w1_history.append(self.w1.item())
            self.loss_history.append(loss.item())

        # Evaluate on test set
        with torch.no_grad():
            y_test_pred = self.forward(x_test)
            ss_total = torch.sum((y_test - torch.mean(y_test)) ** 2)
            ss_residual = torch.sum((y_test - y_test_pred) ** 2)
            self.test_r2 = 1 - (ss_residual / ss_total)

        self.fitted = True

    # d.)
    def predict(self, data):
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction.")
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float64)
        return self.forward(data)

    # e.)
    def analysis_plot(self, x_train, y_train, x_test, y_test):

        if not self.fit:
            raise ValueError("Model must be fitted before analysis.")

        # Plot Original Data
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.scatter(x_train.numpy(), y_train.numpy(), color='blue', label='Training Data')
        plt.title('Original Data')
        plt.xlabel('x_train')
        plt.ylabel('y_train')

        # Plot Fitted Line
        plt.subplot(1, 3, 2)
        plt.scatter(x_test.numpy(), y_test.numpy(), color='blue', label='Test Data')
        plt.plot(x_test.numpy(), self.predict(x_test).detach().numpy(), color='red', label='Fitted Line')
        plt.title('Fitted Line')
        plt.xlabel('x_test')
        plt.ylabel('y_test')

        # Plot loss history
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.plot(self.loss_history)
        plt.title('Loss History')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')

        # Plot w0 history
        plt.subplot(1, 3, 2)
        plt.plot(self.w0_history)
        plt.title('w0 History')
        plt.xlabel('Epochs')
        plt.ylabel('w0')

        # Plot w1 history
        plt.subplot(1, 3, 3)
        plt.plot(self.w1_history)
        plt.title('w1 History')
        plt.xlabel('Epochs')
        plt.ylabel('w1')

        plt.tight_layout()
        plt.show()