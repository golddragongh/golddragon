import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt

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