import importlib


def get_model(cfg, datamodule, phase="train"):
    """
    Inputs:
        cfg (Config): The configuration object containing model details.
        datamodule (DataModule): The data module object for data loading and processing.
        phase (str): The phase of model training. Default is "train".

    This function returns the model based on the model type specified in the configuration. If the model type is not supported, it raises a ValueError.

    Returns:
        Model (object): The model object.
    """
    modeltype = cfg.model.model_type
    if modeltype in ["mld", "temos"]:
        return get_module(cfg, datamodule)
    else:
        raise ValueError(f"Invalid model type {modeltype}.")


def get_module(cfg, datamodule):
    """
    Inputs:
        cfg (Config): The configuration object containing model details.
        datamodule (DataModule): The data module object for data loading and processing.

    This function imports the model module based on the model type specified in the configuration, gets the model class from the module, and returns an instance of the model class.

    Returns:
        Model (object): The model object.
    """
    modeltype = cfg.model.model_type
    model_module = importlib.import_module(
        f".modeltype.{cfg.model.model_type}", package="tma.models")
    Model = model_module.__getattribute__(f"{modeltype.upper()}")
    return Model(cfg=cfg, datamodule=datamodule)
