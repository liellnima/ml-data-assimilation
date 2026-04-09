from collections.abc import Callable

from ml_da.models.base import BaseAssimilationModel
from ml_da.models.persistence import PersistenceModel

MODEL_REGISTRY: dict[str, Callable[..., BaseAssimilationModel]] = {
    "persistence": PersistenceModel,
}


def build_model(name: str, params: dict) -> BaseAssimilationModel:
    try:
        cls = MODEL_REGISTRY[name]
    except KeyError as e:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unknown model '{name}'. Available: {available}") from e

    return cls(**params)
