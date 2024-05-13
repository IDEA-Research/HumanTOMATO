from human_body_prior.body_model.body_model import BodyModel
from pytorch_lightning import LightningModule
import numpy as np
import torch
import time
from torch import nn


class smplx_layer(LightningModule):
    def __init__(self):
        super(smplx_layer, self).__init__()
        self.smplx_model = BodyModel(
            bm_fname="/comp_robot/lushunlin/HumanML3D-1/body_models/smplx/neutral/model.npz",
            num_betas=10,
            model_type="smplx",
        )


if __name__ == "__main__":
    pose = (
        torch.tensor(
            np.load(
                "/comp_robot/lushunlin/visualization/visualization/test_case/motionx_humanml_smplx_322.npy"
            )
        )
        .float()
        .cuda()
    )
    smplx = smplx_layer().cuda()
    output = smplx.smplx_model(
        pose_body=pose[:, 3:66],
        pose_hand=pose[:, 66:156],
        root_orient=pose[:, :3],
        pose_jaw=pose[:, 156:159],
    ).Jtr
