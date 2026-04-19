import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ml_da.data.dataclasses import AssimDataBundle
from ml_da.models.base_model import BaseAssimilationModel
from ml_da.models.EnKF import EnKF
from ml_da.tools.config import DataCoreConfig, ModelConfig


class NeuralModel:
    def __init__(self, state_dim, lr=1e-3, device="cpu"):
        self.device = device

        self.archi = ((24, 5), (37, 5))
        self.use_bilin = True
        self.batchnorm = True
        self.epochs = 1
        self.batch_size = 256

        layers = []
        in_channels = 1

        for out_ch, ksize in self.archi:
            layers.append(nn.Conv1d(in_channels, out_ch, kernel_size=ksize, padding=ksize // 2))

            if self.batchnorm:
                layers.append(nn.BatchNorm1d(out_ch))

            layers.append(nn.ReLU())
            in_channels = out_ch

        layers.append(nn.Conv1d(in_channels, 1, kernel_size=1))

        self.conv = nn.Sequential(*layers).to(device)

        if self.use_bilin:
            self.bilin = nn.Bilinear(state_dim, state_dim, state_dim).to(device)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def parameters(self):
        params = list(self.conv.parameters())
        if self.use_bilin:
            params += list(self.bilin.parameters())
        return params

    def forward(self, x):
        x_cnn = x.unsqueeze(1)
        out = self.conv(x_cnn).squeeze(1)

        if self.use_bilin:
            out = out + self.bilin(x, x)

        return out

    def predict(self, x_np):
        self.conv.eval()
        x = torch.tensor(x_np, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            y = self.forward(x)

        return y.cpu().numpy()

    def train_model(self, X_np, Y_np):
        self.conv.train()

        X = torch.tensor(X_np, dtype=torch.float32, device=self.device)
        Y = torch.tensor(Y_np, dtype=torch.float32, device=self.device)

        dataset = torch.utils.data.TensorDataset(X, Y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for _ in range(self.epochs):
            for xb, yb in loader:
                pred = self.forward(xb)
                loss = self.loss_fn(pred, yb)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()


class HybridModel:
    def __init__(self, base_model, nn_model):
        self.base_model = base_model
        self.nn_model = nn_model

    def __call__(self, Ens, t, dt):
        base = self.base_model(Ens)
        correction = self.nn_model.predict(Ens)
        return base + correction


def build_dataset(trajectory, base_model, dt):
    X, Res = [], []

    for k in range(len(trajectory) - 1):
        x_k = trajectory[k]
        x_k1 = trajectory[k + 1]

        model_pred = base_model(x_k)

        X.append(x_k)
        Res.append(x_k1 - model_pred)

    return np.vstack(X), np.vstack(Res)


class NeuralEnKF(BaseAssimilationModel):
    def __init__(
        self, model_cfg: ModelConfig, data_cfg: DataCoreConfig, data: AssimDataBundle, state_dim=36, device="cpu"
    ):
        super().__init__(model_cfg, data_cfg, data)
        self.enkf = EnKF(model_cfg, data_cfg, data)

        self.nn_model = NeuralModel(state_dim, device=device)

        # Outer loop metrics
        self.training_metrics = []

        # Full history
        self.history = []

    def assimilate(self, n_iter=5):

        for it in range(n_iter):

            HybridModel(self.dyn.step, self.nn_model)

            # Run EnKF
            Ens, metrics, runtime = self.enkf.assimilate()

            # Extract trajectory
            trajectory = np.array(self.enkf.trajectory)

            # Build dataset
            X, Res = build_dataset(
                trajectory,
                self.dyn.step,
            )

            # Train NN
            self.nn_model.train_model(X, Res)

            # Store training metrics
            self.training_metrics.append(
                {
                    "iteration": it,
                    "rmse_final": metrics["rmse"][-1],
                    "spread_final": metrics["spread"][-1],
                    "dataset_size": len(X),
                    "runtime": runtime,
                }
            )

            # Store full time-series metrics
            self.history.append(metrics.copy())

        return self.training_metrics, self.history
