SYSTEM_REGISTRY = {}
OBSERVER_REGISTRY = {}
DYNAMICAL_MODEL_REGISTRY = {}


def system(cls):
    SYSTEM_REGISTRY[cls.__name__] = cls
    return cls


def observer(cls):
    OBSERVER_REGISTRY[cls.__name__] = cls
    return cls


def dynamical_model(cls):
    DYNAMICAL_MODEL_REGISTRY[cls.__name__] = cls
    return cls
