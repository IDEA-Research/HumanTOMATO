import os
from pathlib import Path
import numpy as np
import torch
from pytorch_lightning import LightningModule
from tma.models.metrics import (
    ComputeMetrics,
    TM2TMetrics,
    MMMetrics,
    UncondMetrics,
)
from os.path import join as pjoin
from collections import OrderedDict


class BaseModel(LightningModule):
    """
    This class is a subclass of LightningModule.
    It serves as the base model for all other models.
    """

    def __init__(self, *args, **kwargs):
        """
        Inputs:
            *args, **kwargs: Variable length argument list and keyword arguments.

        This function is the constructor of the BaseModel class. It initializes the BaseModel with the given arguments and keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self.times = []

    def __post_init__(self):
        """
        This function is called after the BaseModel is initialized.
        It calculates the number of trainable and non-trainable parameters and stores them in the hparams.
        """
        trainable, nontrainable = 0, 0
        for p in self.parameters():
            if p.requires_grad:
                trainable += np.prod(p.size())
            else:
                nontrainable += np.prod(p.size())

        self.hparams.n_params_trainable = trainable
        self.hparams.n_params_nontrainable = nontrainable

    def training_step(self, batch, batch_idx):
        """
        Inputs:
            batch: The batch of data for training.
            batch_idx: The index of the batch.

        This function performs a training step and returns the result.

        Returns:
            The result of the training step.
        """
        return self.allsplit_step("train", batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        """
        Inputs:
            batch: The batch of data for validation.
            batch_idx: The index of the batch.

        This function performs a validation step and returns the result.

        Returns:
            The result of the validation step.
        """
        return self.allsplit_step("val", batch, batch_idx)

    def test_step(self, batch, batch_idx):
        """
        Inputs:
            batch: The batch of data for testing.
            batch_idx: The index of the batch.

        This function performs a test step and returns the result.
        It also prints the average time per sample if certain conditions are met.

        Returns:
            The result of the test step.
        """
        if (
            len(self.times) * self.cfg.TEST.BATCH_SIZE % (100) > 0
            and len(self.times) > 0
        ):
            print(
                f"Average time per sample ({self.cfg.TEST.BATCH_SIZE*len(self.times)}): ",
                np.mean(self.times) / self.cfg.TEST.BATCH_SIZE,
            )
        return self.allsplit_step("test", batch, batch_idx)

    def predict_step(self, batch, batch_idx):
        """
        Inputs:
            batch: The batch of data for prediction.
            batch_idx: The index of the batch.

        This function performs a prediction step and returns the result.

        Returns:
            The result of the prediction step.
        """
        return self.forward(batch)

    def allsplit_epoch_end(self, split: str, outputs):
        """
        Inputs:
            split (str): The split of the data ("train", "val", or "test").
            outputs: The outputs of the epoch.

        This function is called at the end of an epoch. It computes the losses and metrics, resets them, and logs them.

        Returns:
            None.
        """
        dico = {}

        # If the split is "train" or "val", compute the losses, reset them, and add them to the dictionary.
        if split in ["train", "val"]:
            losses = self.losses[split]
            loss_dict = losses.compute(split)
            losses.reset()
            dico.update(
                {
                    losses.loss2logname(loss, split): value.item()
                    for loss, value in loss_dict.items()
                    if not torch.isnan(value)
                }
            )

        # If the split is "val" or "test", compute the metrics, reset them, and add them to the dictionary.
        if split in ["val", "test"]:

            if self.trainer.datamodule.is_mm and "TM2TMetrics" in self.metrics_dict:
                metrics_dicts = ["MMMetrics"]
            else:
                metrics_dicts = self.metrics_dict
            for metric in metrics_dicts:
                metrics_dict = getattr(self, metric).compute(
                    sanity_flag=self.trainer.sanity_checking
                )
                # reset metrics
                getattr(self, metric).reset()
                dico.update(
                    {
                        f"Metrics/{metric}": value.item()
                        for metric, value in metrics_dict.items()
                    }
                )

        # If the split is not "test", add the current epoch and step to the dictionary.
        if split != "test":
            dico.update(
                {
                    "epoch": float(self.trainer.current_epoch),
                    "step": float(self.trainer.current_epoch),
                }
            )
        # don't write sanity check into log
        if not self.trainer.sanity_checking:
            self.log_dict(dico, sync_dist=True, rank_zero_only=True)

    def training_epoch_end(self, outputs):
        """
        Inputs:
            outputs: The outputs of the training epoch.

        This function is called at the end of a training epoch.
        It calls allsplit_epoch_end with "train" as the split.

        Returns:
            The result of allsplit_epoch_end.
        """
        return self.allsplit_epoch_end("train", outputs)

    def validation_epoch_end(self, outputs):
        """
        Inputs:
            outputs: The outputs of the validation epoch.

        This function is called at the end of a validation epoch.
        It calls allsplit_epoch_end with "val" as the split.

        Returns:
            The result of allsplit_epoch_end.
        """
        return self.allsplit_epoch_end("val", outputs)

    def test_epoch_end(self, outputs):
        """
        Inputs:
            outputs: The outputs of the test epoch.

        This function is called at the end of a test epoch.
        It saves the outputs, increments the test repetition index,
        and calls allsplit_epoch_end with "test" as the split.

        Returns:
            The result of allsplit_epoch_end.
        """
        self.save_npy(outputs)
        self.cfg.TEST.REP_I = self.cfg.TEST.REP_I + 1

        return self.allsplit_epoch_end("test", outputs)

    def on_save_checkpoint(self, checkpoint):
        """
        Inputs:
            checkpoint: The checkpoint to be saved.

        This function is called when a checkpoint is saved. It removes the 'text_encoder' state from the checkpoint.

        Returns:
            None.
        """
        state_dict = checkpoint["state_dict"]
        clip_k = []
        for k, v in state_dict.items():
            if "text_encoder" in k:
                clip_k.append(k)
        for k in clip_k:
            del checkpoint["state_dict"][k]

    def on_load_checkpoint(self, checkpoint):
        """
        Inputs:
            checkpoint: The checkpoint to be loaded.

        This function is called when a checkpoint is loaded. It restores the 'text_encoder' state to the checkpoint.

        Returns:
            None.
        """
        clip_state_dict = self.text_encoder.state_dict()
        new_state_dict = OrderedDict()
        for k, v in clip_state_dict.items():
            new_state_dict["text_encoder." + k] = v
        for k, v in checkpoint["state_dict"].items():
            if "text_encoder" not in k:
                new_state_dict[k] = v
        checkpoint["state_dict"] = new_state_dict

    def load_state_dict(self, state_dict, strict=True):
        """
        Inputs:
            state_dict: The state dictionary to be loaded.
            strict (bool): Whether to strictly enforce that the keys in state_dict match the keys returned by this module's state_dict() function. Default: True.

        This function loads a state dictionary into the module. If the module has a 'text_encoder', it also loads the 'text_encoder' state dictionary into the module.

        Returns:
            None.
        """
        if hasattr(self, "text_encoder"):
            clip_state_dict = self.text_encoder.state_dict()
            # Initialize an empty ordered dictionary to store the new state dictionary.
            new_state_dict = OrderedDict()
            for k, v in clip_state_dict.items():
                # add the item to the new state dictionary with 'text_encoder.' as the prefix of the key.
                new_state_dict["text_encoder." + k] = v
            for k, v in state_dict.items():
                if "text_encoder" not in k:
                    new_state_dict[k] = v
        else:
            new_state_dict = state_dict

        # Load the new state dictionary into the module.
        super().load_state_dict(new_state_dict, strict)

    def configure_optimizers(self):
        """
        This function is called to configure the optimizers.

        Returns:
            dict: A dictionary containing the optimizer.
        """
        return {"optimizer": self.optimizer}

    def configure_metrics(self):
        """
        This function is called to configure the metrics.

        For each metric in the metrics dictionary, it checks the type of the metric and initializes the corresponding metric object with the appropriate parameters.

        Returns:
            None.
        """
        for metric in self.metrics_dict:
            if metric == "TemosMetric":
                self.TemosMetric = ComputeMetrics(
                    njoints=self.njoints,
                    jointstype=self.cfg.DATASET.JOINT_TYPE,
                    dist_sync_on_step=self.cfg.METRIC.DIST_SYNC_ON_STEP,
                )
            elif metric == "TM2TMetrics":
                self.TM2TMetrics = TM2TMetrics(
                    diversity_times=30 if self.debug else self.cfg.TEST.DIVERSITY_TIMES,
                    dist_sync_on_step=self.cfg.METRIC.DIST_SYNC_ON_STEP,
                    use_TMR=self.cfg.LOSS.TRAIN_TMR,
                )
            elif metric == "UncondMetrics":
                self.UncondMetrics = UncondMetrics(
                    diversity_times=30 if self.debug else self.cfg.TEST.DIVERSITY_TIMES,
                    dist_sync_on_step=self.cfg.METRIC.DIST_SYNC_ON_STEP,
                )
            else:
                # else if the metric is not supported,
                # Raise a NotImplementedError.
                raise NotImplementedError(f"Do not support Metric Type {metric}")

        # If the metrics dictionary contains "TM2TMetrics" or "UncondMetrics",
        if "TM2TMetrics" in self.metrics_dict or "UncondMetrics" in self.metrics_dict:
            self.MMMetrics = MMMetrics(
                mm_num_times=self.cfg.TEST.MM_NUM_TIMES,
                dist_sync_on_step=self.cfg.METRIC.DIST_SYNC_ON_STEP,
            )

    def save_npy(self, outputs):
        """
        This function is used to save the output predictions as .npy files.

        Args:
            outputs (list): A list of output predictions.

        Returns:
            None.
        """

        cfg = self.cfg
        output_dir = Path(
            os.path.join(
                cfg.FOLDER,
                str(cfg.model.model_type),
                str(cfg.NAME),
                "samples",
            )
        )
        if cfg.TEST.SAVE_PREDICTIONS:

            if cfg.DATASET.MOTION_TYPE == "vector_263":
                lengths = [i[1] for i in outputs]
                texts = [i[2] for i in outputs]
                outputs = [i[0] for i in outputs]
            elif cfg.DATASET.MOTION_TYPE == "smplx_212":
                if cfg.TRAIN.use_joints:
                    lengths = [i[1] for i in outputs]
                    gen_motions = [
                        self.datamodule.renormt2m_back(i[0]) for i in outputs
                    ]
                    ref_motions = [
                        self.datamodule.renormt2m_back(i[2]) for i in outputs
                    ]
                else:
                    return
            else:
                raise NotImplementedError

            if cfg.TEST.DATASETS[0].lower() in ["humanml3d", "kit"]:
                # Get the list of keyids from the test dataset.
                keyids = self.trainer.datamodule.test_dataset.name_list
                for i in range(len(outputs)):
                    for bid in range(min(cfg.TEST.BATCH_SIZE, outputs[i].shape[0])):

                        # Get the keyid for the current batch.
                        keyid = keyids[i * cfg.TEST.BATCH_SIZE + bid]

                        # Get the generated joints and text for the current batch.
                        gen_joints = outputs[i][bid].cpu().numpy()
                        text = texts[i][bid]

                        # If the configuration specifies to replicate times,
                        if cfg.TEST.REPLICATION_TIMES > 1:
                            name = f"{keyid}_{cfg.TEST.REP_I}"
                        else:
                            name = f"{keyid}.npy"
                        # save predictions results
                        npypath = output_dir / name
                        np.save(npypath, gen_joints)

                        textpath = output_dir / "text" / (name + ".txt")
                        os.makedirs(os.path.split(textpath)[0], exist_ok=True)
                        with open(textpath, "w") as f:
                            f.write(text)
                        # import pdb; pdb.set_trace()
            elif cfg.TEST.DATASETS[0].lower() in ["humanact12", "uestc"]:
                keyids = range(len(self.trainer.datamodule.test_dataset))
                for i in range(len(outputs)):
                    for bid in range(min(cfg.TEST.BATCH_SIZE, outputs[i].shape[0])):
                        keyid = keyids[i * cfg.TEST.BATCH_SIZE + bid]
                        gen_joints = outputs[i][bid].cpu()
                        gen_joints = gen_joints.permute(2, 0, 1)[
                            : lengths[i][bid], ...
                        ].numpy()
                        if cfg.TEST.REPLICATION_TIMES > 1:
                            name = f"{keyid}_{cfg.TEST.REP_I}"
                        else:
                            name = f"{keyid}.npy"

                        # save predictions results
                        npypath = output_dir / name
                        np.save(npypath, gen_joints)
            elif cfg.TEST.DATASETS[0].lower() in ["motionx"]:
                keyids = self.trainer.datamodule.test_dataset.name_list

                for i in range(len(gen_motions)):
                    for bid in range(min(cfg.TEST.BATCH_SIZE, gen_motions[i].shape[0])):
                        keyid = keyids[i * cfg.TEST.BATCH_SIZE + bid]
                        gen_joints = gen_motions[i][bid].cpu().numpy()
                        ref_joints = ref_motions[i][bid].cpu().numpy()
                        if cfg.TEST.REPLICATION_TIMES > 1:
                            gen_name = f"{keyid}_{cfg.TEST.REP_I}"
                        else:
                            gen_name = f"{keyid}.npy"
                            ref_name = f"{keyid}_gt.npy"

                        # save predictions results
                        npypath = output_dir / gen_name
                        os.makedirs(os.path.split(npypath)[0], exist_ok=True)

                        np.save(npypath, gen_joints)
                        np.save(output_dir / ref_name, ref_joints)
