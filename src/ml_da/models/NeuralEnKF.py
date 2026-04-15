import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .EnKF import EnKF


class NeuralModel:
    def __init__(self, state_dim, lr=1e-3, device="cpu"):
        self.device = device

        # Default settings from Neural EnKF
        self.archi = ((24, 5), (37, 5))  # (channels, kernel)
        self.use_bilin = True
        self.batchnorm = True
        self.epochs = 1
        self.batch_size = 256

        # Build CNN
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

        # Bilinear term
        if self.use_bilin:
            self.bilin = nn.Bilinear(state_dim, state_dim, state_dim).to(device)

        # Optimizer
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
    """Combines the dynamic model and NN."""

    def __init__(self, base_model, nn_model):
        self.base_model = base_model
        self.nn_model = nn_model

    def __call__(self, Ens, t, dt):
        base = self.base_model(Ens, t, dt)
        correction = self.nn_model.predict(Ens)
        return base + correction


def build_dataset(trajectory, base_model, time_sequence, dt):
    """Converts the EnKF trajectory into a supervised learning dataset for the neural network."""
    X, Res = [], []

    for k in range(len(trajectory) - 1):
        x_k = trajectory[k]
        x_k1 = trajectory[k + 1]

        model_pred = base_model(x_k, time_sequence[k], dt)

        X.append(x_k)
        Res.append(x_k1 - model_pred)  # residual learning

    return np.vstack(X), np.vstack(Res)


class NeuralEnKF:
    def __init__(
        self,
        dynamic_model,
        observation_operator,
        CovX0,
        Covy,
        state_dim,
        dt,
        N_ens=20,
        device="cpu",
    ):
        # Create EnKF internally
        self.enkf = EnKF()
        self.enkf.N = N_ens

        self.dynamic_model = dynamic_model
        self.observation_operator = observation_operator
        self.CovX0 = CovX0
        self.Covy = Covy
        self.dt = dt

        # Neural model
        self.nn_model = NeuralModel(state_dim, device=device)

        self.metrics = []

    def run(self, obs, time_sequence, n_iter):

        for it in range(n_iter):

            hybrid_model = HybridModel(self.dynamic_model, self.nn_model)

            trajectory = []

            # Hook into EnKF
            def store(Ens, *args, **kwargs):
                trajectory.append(np.mean(Ens, axis=0))

            self.enkf.Compute_evaluation_metrics = (
                store  # Currently store trajectory, will fix it later to also keep track of rmse
            )

            # EnKF step
            self.enkf.step(
                ground_truth=None,
                obs=obs,
                CovX0=self.CovX0,
                Covy=self.Covy,
                time_sequence=time_sequence,
                dynamic_model=hybrid_model,
                observation_operator=self.observation_operator,
                dt=self.dt,
            )

            trajectory = np.array(trajectory)

            # Build dataset
            X, Res = build_dataset(
                trajectory,
                self.dynamic_model,
                time_sequence,
                self.dt,
            )

            # Train NN
            self.nn_model.train_model(X, Res)

            self.metrics.append(
                {
                    "iteration": it,
                    "dataset_size": len(X),
                }
            )
