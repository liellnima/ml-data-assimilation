GENERATOR_REGISTRY = {}


def data_generator(cls):
    GENERATOR_REGISTRY[cls.__name__] = cls
    return cls
