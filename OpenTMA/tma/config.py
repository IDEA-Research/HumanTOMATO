import importlib
from argparse import ArgumentParser
from omegaconf import OmegaConf
import os


def get_module_config(cfg_model, path="modules"):
    """
    This function loads configuration files from a specified directory and merges them into a given configuration model.

    Parameters:
    cfg_model: The configuration model to merge the loaded configurations into.
    path (str): The directory to load configuration files from. Default is "modules".

    Returns:
    The merged configuration model.
    """
    # List all files in the specified directory
    files = os.listdir(f'./configs/{path}/')
    for file in files:
        if file.endswith('.yaml'):
            with open(f'./configs/{path}/' + file, 'r') as f:
                # Load the configuration from the file and merge it into the configuration model
                cfg_model.merge_with(OmegaConf.load(f))
    # Return the merged configuration model
    return cfg_model


def get_obj_from_str(string, reload=False):
    """
    This function gets an object from a string that specifies the module and class name.

    Parameters:
    string (str): The string that specifies the module and class name.
    reload (bool): Whether to reload the module. Default is False.

    Returns:
    The object specified by the string.
    """
    # Split the string into the module and class name
    module, cls = string.rsplit(".", 1)
    if reload:
        # Import the module
        module_imp = importlib.import_module(module)
        # Reload the module
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    """
    This function instantiates an object from a configuration that specifies the target class and parameters.

    Parameters:
    config: The configuration that specifies the target class and parameters.

    Returns:
    cfg: The instantiated object.
    """
    # If the configuration does not specify a target class
    if not "target" in config:
        # If the configuration is '__is_first_stage__' or '__is_unconditional__', return None
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        # Otherwise, raise a KeyError
        raise KeyError("Expected key `target` to instantiate.")

    # Instantiate the object with the specified class and parameters
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def parse_args(phase="train"):
    """
    This function parses command-line arguments for different phases of a machine learning pipeline.

    Parameters:
    phase (str): The phase of the pipeline. Default is "train".

    Returns:
    The parsed command-line arguments.
    """
    # Create an argument parser
    parser = ArgumentParser()
    group = parser.add_argument_group("Training options")

    """
    The function checks if the phase is in the list ["train", "test", "demo"], and if so, it adds arguments related to the configuration file, asset paths, batch size, and device for training. These arguments are added to a group named "Training options" in the argument parser.
    If the phase is "demo", additional arguments related to rendering visualized figures, frame rate, input text, task, output directory, and output file type are added.
    If the phase is "render", arguments related to rendering, such as the configuration file, asset paths, npy motion files, render target, and joint type for the skeleton are added.
    """

    if phase in ["train", "test", "demo"]:
        group.add_argument(
            "--cfg",
            type=str,
            required=False,
            default="./configs/config.yaml",
            help="config file",
        )
        group.add_argument(
            "--cfg_assets",
            type=str,
            required=False,
            default="./configs/assets.yaml",
            help="config file for asset paths",
        )
        group.add_argument("--batch_size",
                           type=int,
                           required=False,
                           help="training batch size")
        group.add_argument("--device",
                           type=int,
                           nargs="+",
                           required=False,
                           help="training device")
        group.add_argument("--nodebug",
                           action="store_true",
                           required=False,
                           help="debug or not")
        group.add_argument("--dir",
                           type=str,
                           required=False,
                           help="evaluate existing npys")

    if phase == "demo":

        group.add_argument("--render",
                           action="store_true",
                           help="Render visulizaed figures")
        group.add_argument("--render_mode", type=str, help="video or sequence")
        group.add_argument(
            "--frame_rate",
            type=float,
            default=12.5,
            help="the frame rate for the input/output motion",
        )
        group.add_argument(
            "--replication",
            type=int,
            default=1,
            help="the frame rate for the input/output motion",
        )
        group.add_argument(
            "--example",
            type=str,
            required=False,
            help="input text and lengths with txt format",
        )
        group.add_argument(
            "--task",
            type=str,
            required=False,
            help="random_sampling, reconstrucion or text_motion",
        )
        group.add_argument(
            "--out_dir",
            type=str,
            required=False,
            help="output dir",
        )
        group.add_argument(
            "--allinone",
            action="store_true",
            required=False,
            help="output seperate or combined npy file",
        )

    if phase == "render":
        group.add_argument(
            "--cfg",
            type=str,
            required=False,
            default="./configs/render.yaml",
            help="config file",
        )
        group.add_argument(
            "--cfg_assets",
            type=str,
            required=False,
            default="./configs/assets.yaml",
            help="config file for asset paths",
        )

        group.add_argument("--npy",
                           type=str,
                           required=False,
                           default=None,
                           help="npy motion files")
        group.add_argument("--dir",
                           type=str,
                           required=False,
                           default=None,
                           help="npy motion folder")
        group.add_argument(
            "--mode",
            type=str,
            required=False,
            default="sequence",
            help="render target: video, sequence, frame",
        )
        group.add_argument(
            "--joint_type",
            type=str,
            required=False,
            default=None,
            help="mmm or vertices for skeleton",
        )

    # remove None params, and create a dictionnary
    params = parser.parse_args()

    # update config from files
    cfg_base = OmegaConf.load('./configs/base.yaml')
    cfg_exp = OmegaConf.merge(cfg_base, OmegaConf.load(params.cfg))

    cfg_model = get_module_config(cfg_exp.model, cfg_exp.model.target)
    cfg_assets = OmegaConf.load(params.cfg_assets)

    # Merge the experiment, model, and assets configurations.
    cfg = OmegaConf.merge(cfg_exp, cfg_model, cfg_assets)

    if phase in ["train", "test"]:
        # Update the batch size, device, and debug mode from the command-line arguments, if provided.
        cfg.TRAIN.BATCH_SIZE = (params.batch_size
                                if params.batch_size else cfg.TRAIN.BATCH_SIZE)
        cfg.DEVICE = params.device if params.device else cfg.DEVICE
        cfg.DEBUG = not params.nodebug if params.nodebug is not None else cfg.DEBUG

        # no debug in test
        cfg.DEBUG = False if phase == "test" else cfg.DEBUG
        if phase == "test":
            cfg.DEBUG = False
            cfg.DEVICE = [0]
            print("Force no debugging and one gpu when testing")
        cfg.TEST.TEST_DIR = params.dir if params.dir else cfg.TEST.TEST_DIR

    # If the phase is "demo"
    if phase == "demo":
        # Update the render, frame rate, example, task, folder, replication, and outall from the command-line arguments.
        cfg.DEMO.RENDER = params.render
        cfg.DEMO.FRAME_RATE = params.frame_rate
        cfg.DEMO.EXAMPLE = params.example
        cfg.DEMO.TASK = params.task
        cfg.TEST.FOLDER = params.out_dir if params.dir else cfg.TEST.FOLDER
        cfg.DEMO.REPLICATION = params.replication
        cfg.DEMO.OUTALL = params.allinone

    if phase == "render":
        # Update the npy, dir, joint type, and mode from the command-line arguments.
        if params.npy:
            cfg.RENDER.NPY = params.npy
            cfg.RENDER.INPUT_MODE = "npy"
        if params.dir:
            cfg.RENDER.DIR = params.dir
            cfg.RENDER.INPUT_MODE = "dir"
        cfg.RENDER.JOINT_TYPE = params.joint_type
        cfg.RENDER.MODE = params.mode

    # If debug mode is enabled, update the name and logger settings.
    if cfg.DEBUG:
        cfg.NAME = "debug--" + cfg.NAME
        cfg.LOGGER.WANDB.OFFLINE = True
        cfg.LOGGER.VAL_EVERY_STEPS = 1

    # Return the updated configuration.
    return cfg
