from os.path import join as pjoin

import numpy as np
from .humanml.utils.word_vectorizer import (
    WordVectorizer,
    WordVectorizer_only_text_token,
)
from .HumanML3D import HumanML3DDataModule
from .Kit import KitDataModule
from .Humanact12 import Humanact12DataModule
from .Uestc import UestcDataModule
from .UniMocap import UniMocapDataModule
from .utils import *
from .MotionX import Motion_XDataModule


def get_mean_std(phase, cfg, dataset_name):

    # todo: use different mean and val for phases
    name = "t2m" if dataset_name == "humanml3d" else dataset_name
    assert name in ["t2m", "kit", "motionx", "unimocap"]
    # if phase in ["train", "val", "test"]:
    if name in ["t2m", "kit"]:
        if phase in ["val"]:
            if name == "t2m":
                data_root = pjoin(cfg.model.t2m_path, name, "Comp_v6_KLD01", "meta")
            elif name == "kit":
                data_root = pjoin(cfg.model.t2m_path, name, "Comp_v6_KLD005", "meta")
            else:
                raise ValueError("Only support t2m and kit")
            mean = np.load(pjoin(data_root, "mean.npy"))
            std = np.load(pjoin(data_root, "std.npy"))
        else:
            data_root = eval(f"cfg.DATASET.{dataset_name.upper()}.ROOT")
            mean = np.load(pjoin(data_root, "Mean.npy"))
            std = np.load(pjoin(data_root, "Std.npy"))
    elif name in ["motionx"]:

        if phase in ["val"]:
            data_root = pjoin(
                cfg.model.t2m_path,
                name,
                cfg.DATASET.VERSION,
                cfg.DATASET.MOTION_TYPE,
                "Decomp_SP001_SM001_H512",
                "meta",
            )
            mean = np.load(pjoin(data_root, "mean.npy"))
            std = np.load(pjoin(data_root, "std.npy"))

        else:
            data_root = eval(f"cfg.DATASET.{dataset_name.upper()}.ROOT")
            mean = np.load(
                pjoin(
                    data_root,
                    "mean_std",
                    cfg.DATASET.VERSION,
                    cfg.DATASET.MOTION_TYPE,
                    "mean.npy",
                )
            )
            std = np.load(
                pjoin(
                    data_root,
                    "mean_std",
                    cfg.DATASET.VERSION,
                    cfg.DATASET.MOTION_TYPE,
                    "std.npy",
                )
            )

    elif name in ["unimocap"]:
        if phase in ["val"]:
            data_root = pjoin(cfg.model.t2m_path, name, "CLH_v0", "meta")
            print(f"Loading UniMocap dataset val mean std from {data_root}")
            mean = np.load(pjoin(data_root, "mean.npy"))
            std = np.load(pjoin(data_root, "std.npy"))
        else:
            data_root = eval(f"cfg.DATASET.{dataset_name.upper()}.ROOT")
            print(f"Loading UniMocap dataset train/test mean std from {data_root}")
            mean = np.load(pjoin(data_root, "Mean.npy"))
            std = np.load(pjoin(data_root, "Std.npy"))

    return mean, std


def get_njoints(dataset_name):
    if dataset_name == "humanml3d":
        njoints = 22
    elif dataset_name == "unimocap":
        njoints = 22
    elif dataset_name == "kit":
        njoints = 21
    else:
        raise NotImplementedError

    return njoints


def reget_mean_std(cfg, dataset_name, mean, std):

    njoints = get_njoints(dataset_name)
    # import pdb; pdb.set_trace()
    if cfg.DATASET.MOTION_TYPE == "root_position":
        mean = mean[..., : 4 + (njoints - 1) * 3]
    elif cfg.DATASET.MOTION_TYPE == "root_position_vel":
        mean = np.concatenate(
            (
                mean[..., : 4 + (njoints - 1) * 3],
                mean[..., 4 + (njoints - 1) * 9 : 4 + (njoints - 1) * 9 + njoints * 3],
            ),
            axis=0,
        )
    elif cfg.DATASET.MOTION_TYPE == "root_position_rot6d":
        mean = np.concatenate(
            (
                mean[..., : 4 + (njoints - 1) * 3],
                mean[..., 4 + (njoints - 1) * 3 : 4 + (njoints - 1) * 9],
            ),
            axis=0,
        )
    elif cfg.DATASET.MOTION_TYPE == "root_rot6d":
        mean = np.concatenate(
            (mean[..., :4], mean[..., 4 + (njoints - 1) * 3 : 4 + (njoints - 1) * 9]),
            axis=0,
        )
    elif cfg.DATASET.MOTION_TYPE == "vector_263":
        pass
    else:
        raise NotImplementedError

    if cfg.DATASET.MOTION_TYPE == "root_position":
        std = std[..., : 4 + (njoints - 1) * 3]
    elif cfg.DATASET.MOTION_TYPE == "root_position_vel":
        std = np.concatenate(
            (
                std[..., : 4 + (njoints - 1) * 3],
                std[..., 4 + (njoints - 1) * 9 : 4 + (njoints - 1) * 9 + njoints * 3],
            ),
            axis=0,
        )
    elif cfg.DATASET.MOTION_TYPE == "root_position_rot6d":
        std = np.concatenate(
            (
                std[..., : 4 + (njoints - 1) * 3],
                std[..., 4 + (njoints - 1) * 3 : 4 + (njoints - 1) * 9],
            ),
            axis=0,
        )
    elif cfg.DATASET.MOTION_TYPE == "root_rot6d":
        std = np.concatenate(
            (std[..., :4], std[..., 4 + (njoints - 1) * 3 : 4 + (njoints - 1) * 9]),
            axis=0,
        )
    elif cfg.DATASET.MOTION_TYPE == "vector_263":
        pass
    else:
        raise NotImplementedError

    return mean, std


def get_WordVectorizer(cfg, phase, dataset_name):
    # import pdb; pdb.set_trace()
    if phase not in ["text_only"]:
        if dataset_name.lower() in ["unimocap", "motionx", "humanml3d", "kit"]:
            if cfg.model.eval_text_source == "token":
                return WordVectorizer(
                    cfg.DATASET.WORD_VERTILIZER_PATH,
                    "our_vab",
                    cfg.model.eval_text_encode_way,
                )
            else:
                return WordVectorizer_only_text_token(
                    cfg.DATASET.WORD_VERTILIZER_PATH,
                    "our_vab",
                    cfg.model.eval_text_encode_way,
                )
        else:
            raise ValueError("Only support WordVectorizer for HumanML3D")
    else:
        return None


def get_collate_fn(name, cfg, phase="train"):
    if name.lower() in ["humanml3d", "kit", "motionx", "unimocap"]:
        if cfg.model.condition in [
            "text_all",
            "text_face",
            "text_body",
            "text_hand",
            "text_face_body",
            "text_seperate",
            "only_pose_concat",
            "only_pose_fusion",
        ]:
            return tma_collate_text_all
        else:
            return tma_collate
    elif name.lower() in ["humanact12", "uestc"]:
        return a2m_collate


# map config name to module&path
dataset_module_map = {
    "humanml3d": HumanML3DDataModule,
    "kit": KitDataModule,
    "humanact12": Humanact12DataModule,
    "uestc": UestcDataModule,
    "motionx": Motion_XDataModule,
    "unimocap": UniMocapDataModule,
}
motion_subdir = {
    "unimocap": "new_joint_vecs",
    "humanml3d": "new_joint_vecs",
    "kit": "new_joint_vecs",
    "motionx": "motion_data",
}


def get_datasets(cfg, logger=None, phase="train"):
    # get dataset names form cfg

    dataset_names = eval(f"cfg.{phase.upper()}.DATASETS")
    datasets = []
    for dataset_name in dataset_names:
        if dataset_name.lower() in ["unimocap", "humanml3d", "kit"]:
            data_root = eval(f"cfg.DATASET.{dataset_name.upper()}.ROOT")
            # get mean and std corresponding to dataset
            mean, std = get_mean_std(phase, cfg, dataset_name)
            mean_eval, std_eval = get_mean_std("val", cfg, dataset_name)

            mean, std = reget_mean_std(cfg, dataset_name, mean, std)
            mean_eval, std_eval = reget_mean_std(cfg, dataset_name, mean_eval, std_eval)

            # get WordVectorizer
            wordVectorizer = get_WordVectorizer(cfg, phase, dataset_name)

            # get collect_fn
            collate_fn = get_collate_fn(dataset_name, cfg, phase)

            # get dataset module
            if dataset_name.lower() == "unimocap":
                dataset = dataset_module_map[dataset_name.lower()](
                    cfg=cfg,
                    batch_size=cfg.TRAIN.BATCH_SIZE,
                    num_workers=cfg.TRAIN.NUM_WORKERS,
                    debug=cfg.DEBUG,
                    collate_fn=collate_fn,
                    mean=mean,
                    std=std,
                    mean_eval=mean_eval,
                    std_eval=std_eval,
                    w_vectorizer=wordVectorizer,
                    input_format=cfg.DATASET.MOTION_TYPE,
                    text_dir=pjoin(data_root, "texts"),
                    motion_dir=pjoin(data_root, motion_subdir[dataset_name]),
                    max_motion_length=cfg.DATASET.SAMPLER.MAX_LEN,
                    min_motion_length=cfg.DATASET.SAMPLER.MIN_LEN,
                    max_text_len=cfg.DATASET.SAMPLER.MAX_TEXT_LEN,
                    text_source=cfg.DATASET.TEXT_SOURCE,
                    unit_length=eval(f"cfg.DATASET.{dataset_name.upper()}.UNIT_LEN"),
                )
            else:
                dataset = dataset_module_map[dataset_name.lower()](
                    cfg=cfg,
                    batch_size=cfg.TRAIN.BATCH_SIZE,
                    num_workers=cfg.TRAIN.NUM_WORKERS,
                    debug=cfg.DEBUG,
                    collate_fn=collate_fn,
                    mean=mean,
                    std=std,
                    mean_eval=mean_eval,
                    std_eval=std_eval,
                    w_vectorizer=wordVectorizer,
                    input_format=cfg.DATASET.MOTION_TYPE,
                    text_dir=pjoin(data_root, "texts"),
                    motion_dir=pjoin(data_root, motion_subdir[dataset_name]),
                    max_motion_length=cfg.DATASET.SAMPLER.MAX_LEN,
                    min_motion_length=cfg.DATASET.SAMPLER.MIN_LEN,
                    max_text_len=cfg.DATASET.SAMPLER.MAX_TEXT_LEN,
                    unit_length=eval(f"cfg.DATASET.{dataset_name.upper()}.UNIT_LEN"),
                )
            datasets.append(dataset)
        elif dataset_name.lower() in ["humanact12", "uestc"]:
            # get collect_fn
            collate_fn = get_collate_fn(dataset_name, phase)
            # get dataset module
            dataset = dataset_module_map[dataset_name.lower()](
                datapath=eval(f"cfg.DATASET.{dataset_name.upper()}.ROOT"),
                cfg=cfg,
                batch_size=cfg.TRAIN.BATCH_SIZE,
                num_workers=cfg.TRAIN.NUM_WORKERS,
                debug=cfg.DEBUG,
                collate_fn=collate_fn,
                num_frames=cfg.DATASET.HUMANACT12.NUM_FRAMES,
                sampling=cfg.DATASET.SAMPLER.SAMPLING,
                sampling_step=cfg.DATASET.SAMPLER.SAMPLING_STEP,
                pose_rep=cfg.DATASET.HUMANACT12.POSE_REP,
                max_len=cfg.DATASET.SAMPLER.MAX_LEN,
                min_len=cfg.DATASET.SAMPLER.MIN_LEN,
                num_seq_max=cfg.DATASET.SAMPLER.MAX_SQE if not cfg.DEBUG else 100,
                glob=cfg.DATASET.HUMANACT12.GLOB,
                text_source=cfg.DATASET.TEXT_SOURCE,
                translation=cfg.DATASET.HUMANACT12.TRANSLATION,
            )
            cfg.DATASET.NCLASSES = dataset.nclasses
            datasets.append(dataset)
        elif dataset_name.lower() in ["amass"]:
            # todo: add amass dataset
            raise NotImplementedError

        elif dataset_name.lower() in ["motionx"]:
            data_root = eval(f"cfg.DATASET.{dataset_name.upper()}.ROOT")
            # get mean and std corresponding to dataset
            mean, std = get_mean_std(phase, cfg, dataset_name)
            mean_eval, std_eval = get_mean_std("val", cfg, dataset_name)
            # get WordVectorizer
            wordVectorizer = get_WordVectorizer(cfg, phase, dataset_name)
            # get collect_fn
            collate_fn = get_collate_fn(dataset_name, cfg, phase)
            # get dataset module
            dataset = dataset_module_map[dataset_name.lower()](
                cfg=cfg,
                batch_size=cfg.TRAIN.BATCH_SIZE,
                num_workers=cfg.TRAIN.NUM_WORKERS,
                debug=cfg.DEBUG,
                collate_fn=collate_fn,
                mean=mean,
                std=std,
                mean_eval=mean_eval,
                std_eval=std_eval,
                w_vectorizer=wordVectorizer,
                semantic_text_dir=cfg.DATASET.MOTIONX.SEMANTIC_TEXT_ROOT,
                face_text_dir=cfg.DATASET.MOTIONX.FACE_TEXT_ROOT,
                condition=cfg.model.condition,
                motion_dir=pjoin(data_root, motion_subdir[dataset_name]),
                dataset_name=dataset_name,
                eval_text_encode_way=cfg.model.eval_text_encode_way,
                text_source=cfg.DATASET.TEXT_SOURCE,
                motion_type=cfg.DATASET.MOTION_TYPE,
                max_motion_length=cfg.DATASET.SAMPLER.MAX_LEN,
                min_motion_length=cfg.DATASET.SAMPLER.MIN_LEN,
                max_text_len=cfg.DATASET.SAMPLER.MAX_TEXT_LEN,
                unit_length=eval(f"cfg.DATASET.{dataset_name.upper()}.UNIT_LEN"),
            )
            datasets.append(dataset)

        else:
            raise NotImplementedError

    cfg.DATASET.NFEATS = datasets[0].nfeats
    cfg.DATASET.NJOINTS = datasets[0].njoints
    return datasets
