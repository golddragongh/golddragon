import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler


class TorchNet(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.layer1 = nn.Linear(input_dim, 4)
        self.layer2 = nn.Linear(4, 3)
        self.layer3 = nn.Linear(3, output_dim)

        self.relu = nn.ReLU()

        self.scaler = StandardScaler()
        self.loss_history = []


    # Forward propagation
    def forward(self, x):

        z1 = self.layer1(x)
        a1 = self.relu(z1)

        z2 = self.layer2(a1)
        a2 = self.relu(z2)

        z3 = self.layer3(a2)

        return z3


    # Standardize data
    def standardize(self, x, fit=False):

        if fit:
            return self.scaler.fit_transform(x)

        return self.scaler.transform(x)


    # Train the neural network
    def fit_model(self, X_train, Y_train,
                  epochs=1000, lr=0.01):

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)

        self.loss_history = []
        self.train()

        for epoch in range(epochs):

            optimizer.zero_grad()

            y_pred = self(X_train)

            loss = criterion(y_pred, Y_train)

            loss.backward()

            optimizer.step()

            self.loss_history.append(loss.item())

            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch+1}, Loss: {loss.item():.6f}")

        return self.loss_history


    # Plot training loss
    def plot_loss(self):

        plt.figure(figsize=(8, 6))

        plt.plot(
            range(1, len(self.loss_history) + 1),
            self.loss_history
        )

        plt.xlabel("Epoch")
        plt.ylabel("Binary Cross-Entropy Loss")
        plt.title("Neural Network Training Loss")
        plt.grid()
        plt.show()


    # Save model in ONNX format
    def save_onnx(self, filename="diabetes_model.onnx"):

        self.eval()

        dummy_input = torch.zeros(
            1, self.input_dim, dtype=torch.float32
        )

        torch.onnx.export(
            self,
            dummy_input,
            filename,
            input_names=["input"],
            output_names=["output"],
            opset_version=17,
            dynamo=False
        )

        print("Model successfully exported to ONNX")


    # Load ONNX model
    def load_onnx(self, filename="diabetes_model.onnx"):

        import onnxruntime as ort

        self.session = ort.InferenceSession(
            filename,
            providers=["CPUExecutionProvider"]
        )

        print("ONNX model successfully loaded")


    # Predict using ONNX model
    def predict_onnx(self, x):

        x_standard = self.standardize(x)

        x_standard = x_standard.astype(np.float32)

        input_name = self.session.get_inputs()[0].name

        logits = self.session.run(
            None,
            {input_name: x_standard}
        )[0]

        probabilities = torch.sigmoid(
            torch.from_numpy(logits)
        ).numpy()

        predictions = (probabilities >= 0.5).astype(int)

        return probabilities, predictions
