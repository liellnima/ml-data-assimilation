import logging
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

logger = logging.getLogger(__name__)


class NeuralModel:
    def __init__(self, state_dim, lr=1e-3):

        self.archi = ((24, 5), (37, 5))
        self.use_bilin = True
        self.batchnorm = False
        self.epochs = 30
        self.batch_size = 36

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
        if self.use_bilin:
            self.bilin.eval()
        x = torch.tensor(np.array(x_np), dtype=torch.float32, device=device)

        with torch.no_grad():
            y = self.forward(x)

        return y.detach().cpu().numpy()

    def train_model(self, X_np, Y_np):
        self.conv.train()
        if self.use_bilin:
            self.bilin.train()

        X = torch.tensor(X_np, dtype=torch.float32, device=device)
        Y = torch.tensor(Y_np, dtype=torch.float32, device=device)

        dataset = torch.utils.data.TensorDataset(X, Y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.epochs):
            for xb, yb in loader:
                pred = self.forward(xb)
                loss = self.loss_fn(pred, yb)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        with torch.no_grad():
            test_corr = self.predict(X_np[:-1])
        logger.info(f"Mean |correction| after training: {np.mean(np.abs(test_corr))}")


class HybridModel:
    def __init__(self, base_model, nn_model):
        self.base_model = base_model
        self.nn_model = nn_model
        self.correction_scale = 0.1  # TODO tune this parameters

    def step(self, state):
        base = np.asarray(self.base_model(state))
        correction = self.nn_model.predict(state)
        return base + self.correction_scale * correction


def build_dataset(trajectory, base_model):
    X, Res = [], []

    for k in range(len(trajectory) - 1):
        x_k = np.asarray(trajectory[k])
        x_k1 = np.asarray(trajectory[k + 1])

        model_pred = np.array(base_model(x_k))

        X.append(x_k)
        Res.append(x_k1 - model_pred)

    X = np.concatenate(X, axis=0)  # ((T-1)*N, state_dim)
    Res = np.concatenate(Res, axis=0)  # ((T-1)*N, state_dim)

    return X, Res


def build_model_correction_dataset(forecast_trajectory, analysis_trajectory):
    X, Res = [], []

    for k in range(len(forecast_trajectory)):
        x_f = np.asarray(forecast_trajectory[k])  # (N, d)
        x_a = np.asarray(analysis_trajectory[k + 1])  # (N, d)

        X.append(x_f)
        Res.append(x_a - x_f)

    X = np.concatenate(X, axis=0)
    Res = np.concatenate(Res, axis=0)
    return X, Res


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

    # TODO log forecast RMSE vs analysis RMSE separatetly
    def assimilate(self, n_iter=5):

        metrics_per_iterations = {}

        start_time = time.time()

        for it in range(n_iter):
            logger.info(f"Assimilation iteration: {it}")
            hybrid_model = HybridModel(self.dyn.step, self.nn_model)
            # Inject updated model into EnKF
            self.enkf.dynamical_model = hybrid_model
            # Run EnKF
            metrics, _ = self.enkf.assimilate()
            self.enkf.metrics = init_metrics()

            metrics_per_iterations[it] = metrics

            # Extract trajectory
            trajectory = np.array(self.enkf.trajectory)

            # Build dataset
            X, Res = build_dataset(
                trajectory,
                self.dyn.step,
            )
            # X, Res = build_model_correction_dataset(
            #     self.enkf.forecast_trajectory,
            #     self.enkf.analysis_trajectory,
            # )

            # Train NN
            self.nn_model.train_model(X, Res)

        self.runtime = time.time() - start_time

        return metrics_per_iterations, self.runtime
