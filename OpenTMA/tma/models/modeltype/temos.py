from typing import List, Optional

import torch
from torch import Tensor
from omegaconf import DictConfig
from tma.models.tools.tools import remove_padding

from tma.models.metrics import ComputeMetrics
from torchmetrics import MetricCollection
from tma.models.modeltype.base import BaseModel
from torch.distributions.distribution import Distribution
from tma.config import instantiate_from_config

from tma.models.losses.temos import TemosLosses
from torch.optim import AdamW
from sentence_transformers import SentenceTransformer

from tma.models.architectures import t2m_textenc, t2m_motionenc
import os

import time

import numpy as np
import torch.nn.functional as f
from pathlib import Path

from tma.models.architectures.temos.textencoder.distillbert_actor import DistilbertActorAgnosticEncoder
from tma.models.architectures.temos.motionencoder.actor import ActorAgnosticEncoder
from collections import OrderedDict
from transformers import AutoTokenizer, AutoModel


class TEMOS(BaseModel):
    def __init__(self, cfg, datamodule, **kwargs):
        """
        This class is used to define the TEMOS model.

        Args:
            cfg (Config): The configuration object.
            datamodule (DataModule): The data module object.
            **kwargs: Additional keyword arguments.
        """
        super().__init__()

        # Initialize the model parameters from the configuration.
        self.is_vae = cfg.model.vae
        self.cfg = cfg
        self.condition = cfg.model.condition
        self.stage = cfg.TRAIN.STAGE
        self.datamodule = datamodule
        self.njoints = cfg.DATASET.NJOINTS
        self.debug = cfg.DEBUG
        self.motion_type = cfg.DATASET.MOTION_TYPE

        # Instantiate the text encoder, motion encoder, and motion decoder from the configuration.
        self.textencoder = instantiate_from_config(cfg.textencoder)
        self.motionencoder = instantiate_from_config(cfg.motionencoder)
        self.motiondecoder = instantiate_from_config(cfg.motiondecoder)

        if self.condition in [
            "text",
            "text_uncond",
            "text_all",
            "text_face",
            "text_body",
            "text_hand",
            "text_face_body",
            "text_seperate",
            "only_pose_concat",
            "only_pose_fusion",
        ]:
            self._get_t2m_evaluator(cfg)
            self.feats2joints = datamodule.feats2joints

        if cfg.TRAIN.OPTIM.TYPE.lower() == "adamw":
            self.optimizer = AdamW(lr=cfg.TRAIN.OPTIM.LR, params=self.parameters())
        else:
            raise NotImplementedError("Do not support other optimizer for now.")

        # Initialize the losses for training, testing, and validation.
        self._losses = MetricCollection(
            {
                split: TemosLosses(vae=self.is_vae, mode="xyz", cfg=cfg)
                for split in ["losses_train", "losses_test", "losses_val"]
            }
        )

        self.losses = {
            key: self._losses["losses_" + key] for key in ["train", "test", "val"]
        }

        # Configure the metrics.
        self.metrics_dict = cfg.METRIC.TYPE
        self.configure_metrics()

        # If we want to overide it at testing time
        self.sample_mean = False
        self.fact = None

        # If the configuration specifies to use the InfoNCE filter, initialize the filter model.
        if self.cfg.LOSS.USE_INFONCE_FILTER:
            self.filter_model = SentenceTransformer(
                "sentence-transformers/paraphrase-MiniLM-L6-v2"
            )

        # Initialize the retrieval text embedding, motion embedding, and SBERT embedding lists.
        self.retrieval_text_embedding = []
        self.retrieval_motion_embedding = []
        self.retrieval_sbert_embedding = []

        # Initialize the retrieval correspondence name.
        self.retrieval_corres_name = []

        self.gt_idx = 0

        self.__post_init__()

    # Forward: text => motion
    def forward(self, batch: dict):
        """
        This function is used to perform the forward pass of the model.

        Args:
            batch (dict): A dictionary containing the input batch.

        Returns:
            List[Tensor]: A list of tensors representing the output of the forward pass.
        """

        # Perform the text-to-motion forward pass.
        datastruct_from_text = self.text_to_motion_forward(
            batch["text"], batch["length"]
        )

        return remove_padding(datastruct_from_text.joints, batch["length"])

    def _get_t2m_evaluator(self, cfg):
        """
        This function is used to load the Text-to-Motion (T2M) text encoder and motion encoder for evaluation.

        Args:
            cfg (Config): The configuration object.
        """

        # Initialize the T2M text encoder and motion encoder based on the configuration.
        if cfg.model.eval_text_source == "token":
            # If the evaluation text source is 'token', initialize the T2M text encoder with the specified parameters.
            self.t2m_textencoder = t2m_textenc.TextEncoderBiGRUCo(
                word_size=cfg.model.t2m_textencoder.dim_word,
                pos_size=cfg.model.t2m_textencoder.dim_pos_ohot,
                hidden_size=cfg.model.t2m_textencoder.dim_text_hidden,
                output_size=cfg.model.t2m_textencoder.dim_coemb_hidden,
            )
        # If the evaluation text source is 'only_text_token', initialize the T2M text encoder with the specified parameters.
        elif cfg.model.eval_text_source == "only_text_token":
            if "unimocap" in cfg.EVAL.DATASETS:
                self.t2m_textencoder = t2m_textenc.TextEncoderBiGRUCoV2(
                    word_size=cfg.model.t2m_textencoder.dim_word,
                    pos_size=cfg.model.t2m_textencoder.dim_pos_ohot,  # added
                    hidden_size=cfg.model.t2m_textencoder.dim_text_hidden,
                    output_size=cfg.model.t2m_textencoder.dim_coemb_hidden,
                    dataset="unimocap",
                )
            else:
                self.t2m_textencoder = t2m_textenc.TextEncoderBiGRUCoV2(
                    word_size=cfg.model.t2m_textencoder.dim_word,
                    pos_size=cfg.model.t2m_textencoder.dim_pos_ohot,  # added
                    hidden_size=cfg.model.t2m_textencoder.dim_text_hidden,
                    output_size=cfg.model.t2m_textencoder.dim_coemb_hidden,
                )

        elif cfg.model.eval_text_source in ["caption"]:
            if cfg.model.eval_text_encode_way == "clip":
                self.t2m_textencoder, clip_preprocess = clip.load(
                    "ViT-B/32", device=opt.device, jit=False
                )  # Must set jit=False for training
                # Actually this line is unnecessary since clip by default already on float16
                clip.model.convert_weights(text_enc)
                self.t2m_textencoder.eval()
                for p in text_enc.parameters():
                    p.requires_grad = False

            elif cfg.model.eval_text_encode_way == "t5":
                os.environ["TOKENIZERS_PARALLELISM"] = "false"
                self.t2m_textencoder = SentenceTransformer(
                    "sentence-transformers/sentence-t5-xl"
                ).to(opt.device)
                self.t2m_textencoder.eval()
                for p in self.t2m_textencoder.parameters():
                    p.requires_grad = False

            elif "GRU" in cfg.model.eval_text_encode_way:
                if "unimocap" in cfg.EVAL.DATASETS:
                    self.t2m_textencoder = t2m_textenc.TextEncoderBiGRUCoV2(
                        word_size=cfg.model.t2m_textencoder.dim_word,
                        pos_size=cfg.model.t2m_textencoder.dim_pos_ohot,  # added
                        hidden_size=cfg.model.t2m_textencoder.dim_text_hidden,
                        output_size=cfg.model.t2m_textencoder.dim_coemb_hidden,
                        dataset="unimocap",
                    )
                else:
                    self.t2m_textencoder = t2m_textenc.TextEncoderBiGRUCoV2(
                        word_size=cfg.model.t2m_textencoder.dim_word,
                        pos_size=cfg.model.t2m_textencoder.dim_pos_ohot,  # added
                        hidden_size=cfg.model.t2m_textencoder.dim_text_hidden,
                        output_size=cfg.model.t2m_textencoder.dim_coemb_hidden,
                    )
            else:
                raise NotImplementedError

        # Initialize the T2M movement encoder and motion encoder with the specified parameters.
        self.t2m_moveencoder = t2m_motionenc.MovementConvEncoder(
            input_size=cfg.DATASET.NFEATS - 4,
            hidden_size=cfg.model.t2m_motionencoder.dim_move_hidden,
            output_size=cfg.model.t2m_motionencoder.dim_move_latent,
        )

        self.t2m_motionencoder = t2m_motionenc.MotionEncoderBiGRUCo(
            input_size=cfg.model.t2m_motionencoder.dim_move_latent,
            hidden_size=cfg.model.t2m_motionencoder.dim_motion_hidden,
            output_size=cfg.model.t2m_motionencoder.dim_motion_latent,
        )

        # Load the pretrained T2M model.
        dataname = cfg.TEST.DATASETS[0]
        dataname = "t2m" if dataname == "humanml3d" else dataname

        if dataname == "motionx":
            t2m_checkpoint = torch.load(
                os.path.join(
                    cfg.model.t2m_path,
                    dataname,
                    cfg.DATASET.VERSION,
                    cfg.DATASET.MOTION_TYPE,
                    "text_mot_match/model/finest.tar",
                ),
                map_location=torch.device("cpu"),
            )
        else:
            t2m_checkpoint = torch.load(
                os.path.join(
                    cfg.model.t2m_path, dataname, "text_mot_match/model/finest.tar"
                ),
                map_location=torch.device("cpu"),
            )

        self.t2m_textencoder.load_state_dict(t2m_checkpoint["text_encoder"])

        self.t2m_moveencoder.load_state_dict(t2m_checkpoint["movement_encoder"])

        self.t2m_motionencoder.load_state_dict(t2m_checkpoint["motion_encoder"])

        # Set the T2M text encoder, movement encoder, and motion encoder to evaluation mode and freeze their parameters.
        self.t2m_textencoder.eval()
        self.t2m_moveencoder.eval()
        self.t2m_motionencoder.eval()
        for p in self.t2m_textencoder.parameters():
            p.requires_grad = False
        for p in self.t2m_moveencoder.parameters():
            p.requires_grad = False
        for p in self.t2m_motionencoder.parameters():
            p.requires_grad = False

    def sample_from_distribution(
        self,
        distribution: Distribution,
        *,
        fact: Optional[bool] = None,
        sample_mean: Optional[bool] = False,
    ):
        """
        This function samples from a given distribution. If `sample_mean` is True, it returns the mean of the distribution.
        If `fact` is provided, it rescales the sample using the reparameterization trick.

        Args:
            distribution (Distribution): The distribution to sample from.
            fact (Optional[bool]): A factor to rescale the sample. Default is None.
            sample_mean (Optional[bool]): Whether to return the mean of the distribution. Default is False.

        Returns:
            Tensor: The sampled tensor.
        """
        fact = fact if fact is not None else self.fact
        sample_mean = sample_mean if sample_mean is not None else self.sample_mean

        if sample_mean:
            return distribution.loc

        # Reparameterization trick
        if fact is None:
            return distribution.rsample()

        # Resclale the eps
        eps = distribution.rsample() - distribution.loc
        latent_vector = distribution.loc + fact * eps
        return latent_vector

    def text_to_motion_forward(
        self,
        text_sentences: List[str],
        lengths: List[int],
        *,
        return_latent: bool = False,
    ):
        """
        This function encodes the given text sentences into a latent space and then decodes them into a motion.
        If `return_latent` is True, it also returns the latent vector and the distribution.

        Args:
            text_sentences (List[str]): The text sentences to encode.
            lengths (List[int]): The lengths of the text sentences.
            return_latent (bool): Whether to return the latent vector and the distribution. Default is False.

        Returns:
            Tensor: The decoded motion.
            Tensor: The latent vector. Only returned if `return_latent` is True.
            Distribution: The distribution. Only returned if `return_latent` is True.
        """
        # Encode the text to the latent space
        if self.is_vae:
            distribution = self.textencoder(text_sentences)
            latent_vector = self.sample_from_distribution(distribution)
        else:
            distribution = None
            latent_vector = self.textencoder(text_sentences)

        # Decode the latent vector to a motion
        features = self.motiondecoder(latent_vector, lengths)

        if not return_latent:
            return features
        return features, latent_vector, distribution

    def motion_to_motion_forward(
        self,
        features,
        lengths: Optional[List[int]] = None,
        return_latent: bool = False,
        mask_ratio=0,
    ):
        """
        This function encodes the given motion features into a latent space and then decodes them into a motion.
        If `return_latent` is True, it also returns the latent vector and the distribution.
        If `mask_ratio` is greater than 0, it masks a portion of the features before encoding.

        Args:
            features (Tensor): The motion features to encode.
            lengths (Optional[List[int]]): The lengths of the motion features. Default is None.
            return_latent (bool): Whether to return the latent vector and the distribution. Default is False.
            mask_ratio (float): The ratio of features to mask. Default is 0.

        Returns:
            features (Tensor): The decoded motion.
            latent_vector (Tensor): The latent vector. Only returned if `return_latent` is True.
            Distribution: The distribution. Only returned if `return_latent` is True.
        """

        # Encode the motion to the latent space
        # Behaves differently based on whether the class is set to use a VAE or not,
        # and whether a mask ratio is provided.
        if self.is_vae:
            if mask_ratio < 0.001:
                distribution = self.motionencoder(features, lengths)
                latent_vector = self.sample_from_distribution(distribution)
            else:
                num_mask = int(features.shape[1] * mask_ratio)
                mask = np.hstack(
                    [
                        np.zeros(num_mask, dtype=bool),
                        np.ones(features.shape[1] - num_mask),
                    ]
                )
                np.random.shuffle(mask)
                mask_shuffle = torch.tensor(mask, dtype=torch.bool)
                masked_features = features[:, mask_shuffle, :]

                lengths_new = [
                    int(mask_shuffle[: lengths[i]].sum()) for i in range(len(lengths))
                ]

                distribution = self.motionencoder(masked_features, lengths_new)
                latent_vector = self.sample_from_distribution(distribution)
        else:
            if mask_ratio < 0.001:
                distribution = None
                latent_vector: Tensor = self.motionencoder(features, lengths)
            else:
                num_mask = int(features.shape[1] * mask_ratio)
                mask = np.hstack(
                    [np.zeros(num_mask), np.ones(features.shape[1] - num_mask)]
                )
                np.random.shuffle(mask)
                mask_shuffle = torch.tensor(mask, dtype=torch.bool)
                masked_features = features[:, mask_shuffle, :]
                distribution = None
                lengths_new = [
                    int(mask_shuffle[: lengths[i]].sum()) for i in range(len(lengths))
                ]
                latent_vector: Tensor = self.motionencoder(masked_features, lengths_new)

        # Decode the latent vector to a motion
        features = self.motiondecoder(latent_vector, lengths)

        if not return_latent:
            return features
        return features, latent_vector, distribution

    def save_embeddings(self, batch):
        """
        This function saves the embeddings of the text and motion data in the batch.
        It also saves the embeddings of the text data filtered through the Sentence-BERT model if the USE_INFONCE_FILTER flag is set.

        Args:
            batch (dict): The batch of data. It should contain the following keys:
                - "text": The text data.
                - "motion": The motion data.
                - "length": The lengths of the motion data.
                - "word_embs": The word embeddings of the text data.
                - "pos_ohot": The one-hot encoded positions of the words in the text data.
                - "text_len": The lengths of the text data.
                - "retrieval_name": The names of the retrieval data.

        Returns:
            None. The embeddings are saved in the instance variables `retrieval_sbert_embedding`, `retrieval_text_embedding`, `retrieval_motion_embedding`, and `retrieval_corres_name`.
        """

        with torch.no_grad():
            # Initialize the variables to store the embeddings
            motion_all, text_all = None, None
            sbert_embedding_all = None

            # Extract the data from the batch
            texts = batch["text"]
            motions = batch["motion"].detach().clone()
            lengths = batch["length"]
            word_embs = batch["word_embs"].detach().clone()
            pos_ohot = batch["pos_ohot"].detach().clone()
            text_lengths = batch["text_len"].detach().clone()
            retrieval_name = batch["retrieval_name"]

            # Compute the text embeddings
            text_embedding = self.textencoder(texts).loc  # (bs, 256)

            # Compute the motion embeddings, with optional masking
            if self.mr < 0.001:
                motion_embedding = self.motionencoder(motions, lengths).loc  # (bs, 256)
            else:
                # Compute the mask
                num_mask = int(motions.shape[1] * self.mr)
                mask = np.hstack(
                    [
                        np.zeros(num_mask, dtype=bool),
                        np.ones(motions.shape[1] - num_mask),
                    ]
                )
                np.random.shuffle(mask)
                mask_shuffle = torch.tensor(mask, dtype=torch.bool)
                masked_features = motions[:, mask_shuffle, :]

                # Compute the new lengths
                lengths_new = [
                    int(mask_shuffle[: lengths[i]].sum()) for i in range(len(lengths))
                ]
                motion_embedding = self.motionencoder(masked_features, lengths_new).loc

            # Normalize the embeddings
            Emb_text = f.normalize(text_embedding, dim=1)
            Emb_motion = f.normalize(motion_embedding, dim=1)

            # Concatenate the embeddings
            if text_all == None:
                text_all = Emb_text
            else:
                text_all = torch.cat((text_all, Emb_text), 0)

            if motion_all == None:
                motion_all = Emb_motion
            else:
                motion_all = torch.cat((motion_all, Emb_motion), 0)

            # Compute and concatenate the Sentence-BERT embeddings if the USE_INFONCE_FILTER flag is set
            if self.cfg.LOSS.USE_INFONCE_FILTER:
                sbert_embedding = torch.tensor(
                    self.filter_model.encode(texts)
                )  # (bs, 384)
                sbert_embedding = f.normalize(sbert_embedding, dim=1)

                if sbert_embedding_all == None:
                    sbert_embedding_all = sbert_embedding
                else:
                    sbert_embedding_all = torch.cat(
                        (sbert_embedding_all, sbert_embedding), 0
                    )

                self.retrieval_sbert_embedding.append(
                    sbert_embedding_all.detach().cpu().numpy()
                )

            # Save the embeddings with merging into the list
            self.retrieval_text_embedding.append(text_all.detach().cpu().numpy())
            self.retrieval_motion_embedding.append(motion_all.detach().cpu().numpy())
            self.retrieval_corres_name.append(retrieval_name)

    def t2m_eval(self, batch):
        """
        This function evaluates the text-to-motion (t2m) model on a batch of data.

        Args:
            batch (dict): The batch of data. It should contain the following keys:
                - "retrieval_name": The names of the retrieval data.
                - "text": The text data.
                - "motion": The motion data.
                - "length": The lengths of the motion data.
                - "word_embs": The word embeddings of the text data.
                - "pos_ohot": The one-hot encoded positions of the words in the text data.
                - "text_len": The lengths of the text data.

        Returns:
            rs_set (dict): A dictionary containing the following keys:
                - "m_ref": The reference motion data.
                - "m_rst": The reconstructed motion data.
                - "lat_t": The text embeddings.
                - "lat_m": The motion embeddings.
                - "lat_rm": The reconstructed motion embeddings.
                - "joints_ref": The reference joint data.
                - "joints_rst": The reconstructed joint data.
                - "TMR_motion_embedding": The motion embeddings from the TMR model.
                - "TMR_GT_motion_embedding": The ground truth motion embeddings from the TMR model.
                - "TMR_text_embedding": The text embeddings from the TMR model.
        """

        # Extract the data from the batch
        retrieval_name = batch["retrieval_name"]
        texts = batch["text"]
        motions = batch["motion"].detach().clone()
        lengths = batch["length"]
        word_embs = batch["word_embs"].detach().clone()
        pos_ohot = batch["pos_ohot"].detach().clone()
        text_lengths = batch["text_len"].detach().clone()

        # start
        start = time.time()

        # If the data module is multimodal, repeat the data
        if self.trainer.datamodule.is_mm:
            texts = texts * self.cfg.TEST.MM_NUM_REPEATS
            motions = motions.repeat_interleave(self.cfg.TEST.MM_NUM_REPEATS, dim=0)
            lengths = lengths * self.cfg.TEST.MM_NUM_REPEATS
            word_embs = word_embs.repeat_interleave(self.cfg.TEST.MM_NUM_REPEATS, dim=0)
            pos_ohot = pos_ohot.repeat_interleave(self.cfg.TEST.MM_NUM_REPEATS, dim=0)
            text_lengths = text_lengths.repeat_interleave(
                self.cfg.TEST.MM_NUM_REPEATS, dim=0
            )

        # Ensure that the stage is 'temos'
        assert self.stage in ["temos"]

        # Encode the text/decode to a motion
        gt_motion = motions.clone()
        gt_lengths = lengths[:]
        gt_texts = texts[:]

        with torch.no_grad():
            rett = self.text_to_motion_forward(texts, lengths, return_latent=True)
            feat_from_text, latent_from_text, distribution_from_text = rett

            # Encode the motion/decode to a motion
            retm = self.motion_to_motion_forward(motions, lengths, return_latent=True)
            feat_from_motion, latent_from_motion, distribution_from_motion = retm
        clone_feat_from_text = feat_from_text.clone()

        # end time
        end = time.time()
        self.times.append(end - start)

        # joints recover
        joints_rst = self.feats2joints(feat_from_text, self.cfg.DATASET.MOTION_TYPE)
        joints_ref = self.feats2joints(motions, self.cfg.DATASET.MOTION_TYPE)

        # renorm for t2m evaluators
        feats_rst = self.datamodule.renorm4t2m(feat_from_text)
        motions = self.datamodule.renorm4t2m(motions)

        # t2m motion encoder
        m_lens = lengths.copy()
        m_lens = torch.tensor(m_lens, device=motions.device)
        align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
        motions = motions[align_idx]
        feats_rst = feats_rst[align_idx]
        m_lens = m_lens[align_idx]
        m_lens = torch.div(
            m_lens, self.cfg.DATASET.HUMANML3D.UNIT_LEN, rounding_mode="floor"
        )

        TMR_motion_embedding = None
        TMR_GT_motion_embedding = None
        TMR_text_embedding = None

        # Compute the reconstructed motion embeddings
        recons_mov = self.t2m_moveencoder(feats_rst[..., :-4]).detach()
        recons_emb = self.t2m_motionencoder(recons_mov, m_lens)
        motion_mov = self.t2m_moveencoder(motions[..., :-4]).detach()
        motion_emb = self.t2m_motionencoder(motion_mov, m_lens)

        # t2m text encoder
        if self.cfg.model.eval_text_source == "token":
            text_emb = self.t2m_textencoder(word_embs, pos_ohot, text_lengths)[
                align_idx
            ]
        elif self.cfg.model.eval_text_source == "only_text_token":
            text_emb = self.t2m_textencoder(word_embs, text_lengths)[align_idx]
        elif self.cfg.model.eval_text_source in ["caption"]:
            if self.cfg.model.eval_text_encode_way == "clip":
                raise NotImplementedError

            elif self.cfg.model.eval_text_encode_way == "t5":
                raise NotImplementedError

            elif "GRU" in self.cfg.model.eval_text_encode_way:
                text_emb = self.t2m_textencoder(word_embs, text_lengths)[align_idx]
            else:
                raise NotImplementedError

        rs_set = {
            "m_ref": motions,
            "m_rst": feats_rst,
            "lat_t": text_emb,
            "lat_m": motion_emb,
            "lat_rm": recons_emb,
            "joints_ref": joints_ref,
            "joints_rst": joints_rst,
            "TMR_motion_embedding": TMR_motion_embedding,
            "TMR_GT_motion_embedding": TMR_GT_motion_embedding,
            "TMR_text_embedding": TMR_text_embedding,
        }
        return rs_set

    def allsplit_step(self, split: str, batch, batch_idx):

        emb_dist = None
        # If the configuration specifies to use InfoNCE loss and filter, it
        # calculate the embedding distance.
        if self.cfg.LOSS.USE_INFONCE and self.cfg.LOSS.USE_INFONCE_FILTER:
            with torch.no_grad():
                text_embedding = self.filter_model.encode(batch["text"])
                text_embedding = torch.tensor(text_embedding).to(batch["motion"][0])
                normalized = f.normalize(text_embedding, p=2, dim=1)
                emb_dist = normalized.matmul(normalized.T)

        # Encode the text/decode to a motion
        ret = self.text_to_motion_forward(
            batch["text"], batch["length"], return_latent=True
        )
        feat_from_text, latent_from_text, distribution_from_text = ret

        # Encode the motion/decode to a motion
        self.mr = 0
        mr = self.mr
        if split in ["train"]:
            ret = self.motion_to_motion_forward(
                batch["motion"], batch["length"], return_latent=True, mask_ratio=mr
            )
        else:
            ret = self.motion_to_motion_forward(
                batch["motion"], batch["length"], return_latent=True, mask_ratio=mr
            )

        # Unpack the returned values
        feat_from_motion, latent_from_motion, distribution_from_motion = ret

        sync = self.cfg.LOSS.SYNC
        TMR_motion_embedding = None
        TMR_text_embedding = None

        # If the configuration specifies to synchronize the embeddings,
        # calculate the TMR embeddings.
        if sync:
            TMR_motion_embedding = self.t2m_TMR_motionencoder(
                feat_from_text, batch["length"]
            ).loc
            TMR_text_embedding = self.t2m_TMR_textencoder(batch["text"]).loc

        # Compare to a Normal distribution
        if self.is_vae:
            # Create a centred normal distribution to compare with
            mu_ref = torch.zeros_like(distribution_from_text.loc)
            scale_ref = torch.ones_like(distribution_from_text.scale)
            distribution_ref = torch.distributions.Normal(mu_ref, scale_ref)
        else:
            distribution_ref = None

        # Compute the losses
        loss = self.losses[split].update(
            f_text=feat_from_text,
            f_motion=feat_from_motion,
            f_ref=batch["motion"],
            lat_text=latent_from_text,
            lat_motion=latent_from_motion,
            dis_text=distribution_from_text,
            dis_motion=distribution_from_motion,
            dis_ref=distribution_ref,
            emb_dist=emb_dist,
            TMR_text_embedding=TMR_text_embedding,
            TMR_motion_embedding=TMR_motion_embedding,
        )

        if loss is None:
            raise ValueError("Loss is None, this happend with torchmetrics > 0.7")

        # Compute the metrics - currently evaluate results from text to motion
        # The metrics are computed differently depending on the split and the condition
        if split in ["val", "test"]:
            self.save_embeddings(batch)

            if self.condition in [
                "text",
                "text_uncond",
                "text_all",
                "text_face",
                "text_body",
                "text_hand",
                "text_face_body",
                "text_seperate",
                "only_pose_concat",
                "only_pose_fusion",
            ]:
                # use t2m evaluators
                rs_set = self.t2m_eval(batch)
            elif self.condition == "action":
                # use a2m evaluators
                rs_set = self.a2m_eval(batch)
            else:
                raise NotImplementedError

            # MultiModality evaluation sperately
            if self.trainer.datamodule.is_mm:
                metrics_dicts = ["MMMetrics"]
            else:
                metrics_dicts = self.metrics_dict

            for metric in metrics_dicts:
                if metric == "TemosMetric":
                    phase = split if split != "val" else "eval"
                    if eval(f"self.cfg.{phase.upper()}.DATASETS")[0].lower() not in [
                        "humanml3d",
                        "kit",
                        "motionx",
                        "unimocap",
                    ]:
                        raise TypeError(
                            "APE and AVE metrics only support humanml3d and kit datasets now"
                        )

                    getattr(self, metric).update(
                        rs_set["joints_rst"], rs_set["joints_ref"], batch["length"]
                    )
                elif metric == "TM2TMetrics":
                    getattr(self, metric).update(
                        rs_set["lat_t"],
                        rs_set["lat_rm"],
                        rs_set["lat_m"],
                        batch["length"],
                        rs_set["TMR_motion_embedding"],
                        rs_set["TMR_GT_motion_embedding"],
                        rs_set["TMR_text_embedding"],
                    )
                elif metric == "UncondMetrics":
                    getattr(self, metric).update(
                        recmotion_embeddings=rs_set["lat_rm"],
                        gtmotion_embeddings=rs_set["lat_m"],
                        lengths=batch["length"],
                    )
                elif metric == "MRMetrics":
                    getattr(self, metric).update(
                        rs_set["joints_rst"], rs_set["joints_ref"], batch["length"]
                    )
                elif metric == "MMMetrics":
                    getattr(self, metric).update(
                        rs_set["lat_rm"].unsqueeze(0), batch["length"]
                    )
                elif metric == "HUMANACTMetrics":
                    getattr(self, metric).update(
                        rs_set["m_action"],
                        rs_set["joints_eval_rst"],
                        rs_set["joints_eval_ref"],
                        rs_set["m_lens"],
                    )

                else:
                    raise TypeError(f"Not support this metric {metric}")

        # If the split is "test", return the results depending on the motion type
        if split in ["test"]:
            if self.motion_type == "vector_263":
                return rs_set["joints_rst"], batch["length"], batch["text"]
            elif self.motion_type == "smplx_212":
                if self.cfg.TRAIN.use_joints:
                    return rs_set["m_rst"], batch["length"], rs_set["m_ref"]
                else:
                    return batch["length"]

        return loss

    def allsplit_epoch_end(self, split: str, outputs):

        # Initialize an empty dictionary to store the results
        dico = {}

        # If the split is "val" or "test", save the embeddings every 100 epochs
        if split in ["val", "test"]:
            if (self.trainer.current_epoch + 1) % 100 == 0:
                # Define the directory where the embeddings will be saved
                output_dir = Path(
                    os.path.join(
                        self.cfg.FOLDER,
                        str(self.cfg.model.model_type),
                        str(self.cfg.NAME),
                        "embeddings",
                        split,
                        "epoch_" + str(self.trainer.current_epoch),
                    )
                )

                os.makedirs(output_dir, exist_ok=True)

                # Concatenate all the text and motion embeddings across all devices
                self.retrieval_text_embedding = torch.cat(
                    [
                        i.view(-1, i.shape[-1])
                        for i in self.all_gather(self.retrieval_text_embedding)
                    ],
                    dim=0,
                )
                self.retrieval_motion_embedding = torch.cat(
                    [
                        i.view(-1, i.shape[-1])
                        for i in self.all_gather(self.retrieval_motion_embedding)
                    ],
                    dim=0,
                )
                
                # convert the string to tensor via ASCII codes
                # Mainly because the all_gather function does not support string, but it supports tensor
                namelist = [i for sublist in self.retrieval_corres_name for i in sublist]
                max_length = 300
                self.tensor_list = []
                for string in namelist:
                    tensor = torch.zeros(max_length, dtype=torch.int)
                    # print(string)
                    for i, char in enumerate(string):
                        tensor[i] = ord(char)
                    self.tensor_list.append(tensor)

                # all_gather the tensor list
                gathered_data = self.all_gather(self.tensor_list)

                # convert the gathered data to a list of strings
                self._retrieval_corres_name = []
                for gathered_batch in gathered_data:
                    gathered_strings = [''.join([chr(int(char)) for char in tensor if int(char) > 0]) for tensor in gathered_batch]
                    self._retrieval_corres_name.extend(gathered_strings)
                
                
                # Save the corresponding names to a text file
                with open(output_dir / "test_name_list.txt", "w") as test_name_file:
                    for i in self._retrieval_corres_name:
                        test_name_file.write(i + "\n")

                # If the InfoNCE filter is used, concatenate all the sbert embeddings across all devices and save them
                if self.cfg.LOSS.USE_INFONCE_FILTER:
                    self.retrieval_sbert_embedding = torch.cat(
                        [
                            i.view(-1, i.shape[-1])
                            for i in self.all_gather(self.retrieval_sbert_embedding)
                        ],
                        dim=0,
                    )
                    np.save(
                        output_dir / "sbert_embedding.npy",
                        self.retrieval_sbert_embedding.detach().cpu().numpy(),
                    )

                # Save the text and motion embeddings
                np.save(
                    output_dir / "text_embedding.npy",
                    self.retrieval_text_embedding.detach().cpu().numpy(),
                )  # (2324, 256)
                np.save(
                    output_dir / "motion_embedding.npy",
                    self.retrieval_motion_embedding.detach().cpu().numpy(),
                )

                print(
                    "save embedding in {} at {}".format(
                        output_dir, self.trainer.current_epoch
                    )
                )

            # Reset the embeddings and the corresponding names
            self.retrieval_text_embedding = []
            self.retrieval_motion_embedding = []
            self.retrieval_sbert_embedding = []
            self.retrieval_corres_name = []

        # If the split is "train" or "val", compute and log the losses
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

        # If the split is "val" or "test", compute and log the metrics
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

        # If the split is not "test", log the current epoch and step
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

    # This function is called at the end of each training epoch
    # It simply calls the allsplit_epoch_end function with the split set to "train"
    def training_epoch_end(self, outputs):
        return self.allsplit_epoch_end("train", outputs)
