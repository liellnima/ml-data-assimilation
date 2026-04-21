from __future__ import annotations

import logging
import time

import numpy as np

from ml_da.data.dataclasses import AssimDataBundle
from ml_da.evaluation.metrics import compute_metrics, init_metrics
from ml_da.models.da_methods.base_model import BaseAssimilationModel
from ml_da.tools.config import DataCoreConfig, ModelConfig
from ml_da.tools.registry import da_method

logger = logging.getLogger(__name__)


@da_method
class Persistence(BaseAssimilationModel):
    """Persistence baseline (no update, no DA)."""

    def __init__(self, model_cfg: ModelConfig, data_cfg: DataCoreConfig, data: AssimDataBundle):
        super().__init__(model_cfg, data_cfg, data)
        self.metrics = init_metrics()
        self.trajectory = []
        self.runtime = None

    def assimilate(
        self,
    ):
        start_time = time.time()
        x = self.dyn.initial_state

        for t in range(self.timesteps - 1):

            x = self.dyn.step()

            self.log(t + 1, x, self.ground_truth, self.obs_np)

        self.runtime = time.time() - start_time

        return self.metrics, self.runtime

    # Logging
    def log(self, t, x, ground_truth, obs):

        self.metrics["time"].append(t)

        compute_metrics(
            self.metrics,
            estimate=x,
            truth=ground_truth[t] if ground_truth is not None else None,
            observation=obs[t] if obs is not None else None,
        )

        self.metrics["trHK"].append(np.nan)


@da_method
class PersistenceEnsemble(BaseAssimilationModel):
    """
    Ensemble persistence (no analysis step).

    Equivalent to EnKF without update.
    """

    def __init__(self, model_cfg: ModelConfig, data_cfg: DataCoreConfig, data: AssimDataBundle):
        super().__init__(model_cfg, data_cfg, data)
        self.metrics = init_metrics()
        self.trajectory = []
        self.runtime = None

    def assimilate(
        self,
    ):
        start_time = time.time()
        Ens = self.dyn.initial_state

        self.log(0, Ens, self.ground_truth, self.obs_np)  # log once in the beginning

        for t in range(self.timesteps - 1):
            if t % 50 == 0:
                logger.info(f"Assimilated {t} steps")

            Ens = self.dyn.step()

            self.log(t + 1, Ens, self.ground_truth, self.obs_np)

        self.runtime = time.time() - start_time

        return self.metrics, self.runtime

    # Logging
    def log(self, t, Ens, ground_truth, obs):

        self.metrics["time"].append(t)

        compute_metrics(
            self.metrics,
            ensemble=Ens,
            truth=ground_truth[t] if ground_truth is not None else None,
            observation=obs[t] if obs is not None else None,
        )

        self.metrics["trHK"].append(np.nan)
