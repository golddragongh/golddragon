import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt

class LinearRegression(nn.Module):

    # a.)
    def __init__(self, learning_rate=0.01, epochs=1000):
        super().__init__()
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


class CauchyRegression(nn.Module):

    # a.) Initialize the model with learning rate and epochs
    def __init__(self, learning_rate=0.01, epochs=1000, c=1.0):
        super().__init__()
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.c = float(c)

        # Initialze weights for the linear model
        self.w0 = nn.Parameter(torch.tensor(0.0, dtype = torch.float64))
        self.w1 = nn.Parameter(torch.tensor(0.0, dtype = torch.float64))
        self.w2 = nn.Parameter(torch.tensor(0.0, dtype = torch.float64))
        self.w3 = nn.Parameter(torch.tensor(0.0, dtype = torch.float64))
        self.w4 = nn.Parameter(torch.tensor(0.0, dtype = torch.float64))

        # Initialize optimizer and loss function
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        self.loss_history = []
        self.weight_history = []
        self.test_r2 = None
        self.fitted = False


    # b.) Model Equation: y = w0 + w1*x + w2*x^2 + w3*x^3 + w4*x^4
    def forward(self, x1, x2, x3, x4):
        return self.w0 + self.w1 * x1 + self.w2 * x2 + self.w3 * x3 + self.w4 * x4

    # c.) Cauchy Loss Function
    def cauchy_loss(self, y_pred, y):
        residual = y - y_pred
        loss = ((self.c ** 2) / 2) * torch.log1p((residual / self.c)**2)
        return loss.mean()

    # d.) Fit the model to the training data
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

        # Standardize training features
        self.x_mean = x_train.mean(dim=0)
        self.x_std = x_train.std(dim=0, unbiased=False).clamp_min(1e-12)
        x_train = (x_train - self.x_mean) / self.x_std

        # Initialize intercept near the mean target value
        with torch.no_grad():
            self.w0.copy_(y_train.mean())

        # Train
        for epoch in range(self.epochs):
            self.optimizer.zero_grad()  # Zero the gradients
            y_pred = self.forward(x_train[:, 0], x_train[:, 1], x_train[:, 2], x_train[:, 3])  # Forward pass
            loss = self.cauchy_loss(y_pred, y_train)  # Compute loss
            loss.backward()  # Backward pass
            self.optimizer.step()  # Update parameters

            # Store history for analysis
            self.loss_history.append(loss.item())
            self.weight_history.append([self.w0.item(), self.w1.item(), self.w2.item(), self.w3.item(), self.w4.item()])
        self.fitted = True

        # Evaluate on test set
        with torch.no_grad():
            y_test_pred = self.predict(x_test)
            ss_total = torch.sum((y_test - torch.mean(y_test)) ** 2)
            ss_residual = torch.sum((y_test - y_test_pred) ** 2)
            self.test_r2 = 1 - (ss_residual / ss_total)

    # e.) Predict using the fitted model
    def predict(self, data):
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction.")
        data = torch.tensor(data, dtype=torch.float64)
        # Apply the same standardization
        data = (data - self.x_mean) / self.x_std
        with torch.no_grad():
            return self.forward(data[:, 0], data[:, 1], data[:, 2], data[:, 3])


class LogisticRegression(nn.Module):

    # a.) Initialize model
    def __init__(self, learning_rate=0.01, epochs=100, batch_size=256):
        super().__init__()

        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)

        # Four input features and one intercept
        self.w = nn.Parameter(torch.zeros(4))
        self.w0 = nn.Parameter(torch.tensor(0.0))
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)

        self.loss_history = []
        self.fitted = False

    # b.) Linear model (logits)
    def forward(self, x):
        return self.w0 + x @ self.w

    # c.) Fit using mini-batch stochastic gradient descent
    def fit(self, x_train, y_train):

        x_train = torch.as_tensor(x_train, dtype=torch.float32)
        y_train = torch.as_tensor(np.asarray(y_train), dtype=torch.float32).reshape(-1)

        # Account for class imbalance
        positives = y_train.sum()
        negatives = len(y_train) - positives

        loss_fn = nn.BCEWithLogitsLoss(pos_weight=negatives / positives)

        for epoch in range(self.epochs):
            indices = torch.randperm(len(y_train))
            total_loss = 0.0

            for batch in indices.split(self.batch_size):
                self.optimizer.zero_grad()

                logits = self.forward(x_train[batch])
                loss = loss_fn(logits, y_train[batch])

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item() * len(batch)

            self.loss_history.append(total_loss / len(y_train))

        self.fitted = True
        return self

    # d.) Predict probabilities
    def predict_proba(self, x):
        if not self.fitted:
            raise ValueError("Model must be fitted first.")

        x = torch.as_tensor(x, dtype=torch.float32)

        with torch.no_grad():
            return torch.sigmoid(self.forward(x))

    # e.) Predict classes
    def predict(self, x):
        return (self.predict_proba(x) >= 0.5).int()

