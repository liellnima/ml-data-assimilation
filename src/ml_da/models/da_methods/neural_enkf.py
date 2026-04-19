import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ml_da.data.dataclasses import AssimDataBundle
from ml_da.experiments.metrics import init_metrics
from ml_da.models.da_methods.base_model import BaseAssimilationModel
from ml_da.models.da_methods.enkf import EnKF
from ml_da.tools.config import DataCoreConfig, ModelConfig
from ml_da.tools.registry import da_method

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


class NeuralModel:
    def __init__(self, state_dim, lr=1e-3):

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
        x = torch.tensor(x_np, dtype=torch.float32, device=device)

        with torch.no_grad():
            y = self.forward(x)

        return y.cpu().numpy()

    def train_model(self, X_np, Y_np):
        self.conv.train()

        X = torch.tensor(X_np, dtype=torch.float32, device=device)
        Y = torch.tensor(Y_np, dtype=torch.float32, device=device)

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

    def step(self, state):
        base = self.base_model(state)
        correction = self.nn_model.predict(state)
        return base + correction


def build_dataset(trajectory, base_model):
    X, Res = [], []

    for k in range(len(trajectory) - 1):
        x_k = trajectory[k]
        x_k1 = trajectory[k + 1]

        model_pred = base_model(x_k)

        X.append(x_k)
        Res.append(x_k1 - model_pred)

    return np.vstack(X), np.vstack(Res)


@da_method
class NeuralEnKF(BaseAssimilationModel):
    def __init__(
        self,
        model_cfg: ModelConfig,
        data_cfg: DataCoreConfig,
        data: AssimDataBundle,
    ):
        super().__init__(model_cfg, data_cfg, data)
        self.nn_model = NeuralModel(self.system_dim)
        self.hybrid_model = HybridModel(self.dyn.step, self.nn_model)
        self.enkf = EnKF(model_cfg, data_cfg, data)

        self.metrics = init_metrics()

        self.runtime = None

    def assimilate(self, n_iter=5):

        start_time = time.time()
        for it in range(n_iter):
            hybrid_model = HybridModel(self.dyn.step, self.nn_model)

            # Inject updated model into EnKF
            self.enkf.dynamic_model = hybrid_model

            # Run EnKF
            Ens, metrics, runtime = self.enkf.assimilate()

            # time shared once
            if len(self.metrics["time"]) == 0:
                self.metrics["time"] = metrics["time"]

            # append full time-series per iteration
            self.metrics["rmse"].append(metrics["rmse"])
            self.metrics["obs_error"].append(metrics["obs_error"])
            self.metrics["mae"].append(metrics["mae"])
            self.metrics["bias"].append(metrics["bias"])
            self.metrics["spread"].append(metrics["spread"])
            self.metrics["crps"].append(metrics["crps"])
            self.metrics["trHK"].append(metrics["trHK"])

            # Extract trajectory
            trajectory = np.array(self.enkf.trajectory)

            # Build dataset
            X, Res = build_dataset(
                trajectory,
                self.dyn.step,
            )

            # Train NN
            self.nn_model.train_model(X, Res)

        self.runtime = time.time() - start_time

        return self.metrics, self.runtime
