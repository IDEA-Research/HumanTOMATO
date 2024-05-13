from typing import List

import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.functional import pairwise_euclidean_distance
import torch.nn.functional as F

from .utils import *


class TM2TMetrics(Metric):
    full_state_update = True

    def __init__(
        self,
        top_k=3,
        R_size=32,
        #  R_size=256,
        diversity_times=300,
        dist_sync_on_step=True,
        use_TMR=None,
        **kwargs,
    ):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.name = "matching, fid, and diversity scores"
        self.use_TMR = use_TMR
        self.top_k = top_k
        self.R_size = R_size
        self.diversity_times = diversity_times

        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("count_seq", default=torch.tensor(0), dist_reduce_fx="sum")

        self.metrics = []
        # Matching scores
        self.add_state(
            "Matching_score", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state(
            "gt_Matching_score", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.Matching_metrics = ["Matching_score", "gt_Matching_score"]
        if self.use_TMR:
            self.add_state(
                "TMR_Matching_score", default=torch.tensor(0.0), dist_reduce_fx="sum"
            )
            self.add_state(
                "TMR_gt_Matching_score", default=torch.tensor(0.0), dist_reduce_fx="sum"
            )
            self.Matching_metrics = [
                "Matching_score",
                "gt_Matching_score",
                "TMR_Matching_score",
                "TMR_gt_Matching_score",
            ]
        for k in range(1, top_k + 1):
            self.add_state(
                f"R_precision_top_{str(k)}",
                default=torch.tensor(0.0),
                dist_reduce_fx="sum",
            )
            self.Matching_metrics.append(f"R_precision_top_{str(k)}")
        for k in range(1, top_k + 1):
            self.add_state(
                f"gt_R_precision_top_{str(k)}",
                default=torch.tensor(0.0),
                dist_reduce_fx="sum",
            )
            self.Matching_metrics.append(f"gt_R_precision_top_{str(k)}")
        # import pdb; pdb.set_trace()
        if self.use_TMR:
            # New feature extractor
            for k in range(1, top_k + 1):
                self.add_state(
                    f"TMR_R_precision_top_{str(k)}",
                    default=torch.tensor(0.0),
                    dist_reduce_fx="sum",
                )
                self.Matching_metrics.append(f"TMR_R_precision_top_{str(k)}")
            for k in range(1, top_k + 1):
                self.add_state(
                    f"TMR_gt_R_precision_top_{str(k)}",
                    default=torch.tensor(0.0),
                    dist_reduce_fx="sum",
                )
                self.Matching_metrics.append(f"TMR_gt_R_precision_top_{str(k)}")

        self.metrics.extend(self.Matching_metrics)

        # Diversity
        self.add_state("Diversity", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("gt_Diversity", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.metrics.extend(["Diversity", "gt_Diversity"])

        # chached batches
        self.add_state("text_embeddings", default=[], dist_reduce_fx=None)
        self.add_state("recmotion_embeddings", default=[], dist_reduce_fx=None)
        self.add_state("gtmotion_embeddings", default=[], dist_reduce_fx=None)

        if self.use_TMR:
            # chached batches
            self.add_state("TMR_text_embeddings", default=[], dist_reduce_fx=None)
            self.add_state("TMR_recmotion_embeddings", default=[], dist_reduce_fx=None)
            self.add_state("TMR_gtmotion_embeddings", default=[], dist_reduce_fx=None)

    def compute(self, sanity_flag):
        count = self.count.item()
        count_seq = self.count_seq.item()

        # init metrics
        metrics = {metric: getattr(self, metric) for metric in self.metrics}

        # if in sanity check stage then jump
        if sanity_flag:
            return metrics
        # import pdb; pdb.set_trace()
        # cat all embeddings
        shuffle_idx = torch.randperm(count_seq)
        all_texts = torch.cat(self.text_embeddings, axis=0).cpu()[shuffle_idx, :]
        all_genmotions = torch.cat(self.recmotion_embeddings, axis=0).cpu()[
            shuffle_idx, :
        ]
        all_gtmotions = torch.cat(self.gtmotion_embeddings, axis=0).cpu()[
            shuffle_idx, :
        ]
        if self.use_TMR:
            TMR_all_texts = torch.cat(self.TMR_text_embeddings, axis=0).cpu()[
                shuffle_idx, :
            ]
            TMR_all_genmotions = torch.cat(self.TMR_recmotion_embeddings, axis=0).cpu()[
                shuffle_idx, :
            ]
            TMR_all_gtmotions = torch.cat(self.TMR_gtmotion_embeddings, axis=0).cpu()[
                shuffle_idx, :
            ]

        # Compute r-precision
        assert count_seq > self.R_size

        top_k_mat = torch.zeros((self.top_k,))
        TMR_top_k_mat = torch.zeros((self.top_k,))
        for i in range(count_seq // self.R_size):
            # [bs=32, 1*256]
            group_texts = all_texts[i * self.R_size : (i + 1) * self.R_size]
            # [bs=32, 1*256]
            group_motions = all_genmotions[i * self.R_size : (i + 1) * self.R_size]
            if self.use_TMR:
                # [bs=32, 1*256]
                TMR_group_texts = F.normalize(
                    TMR_all_texts[i * self.R_size : (i + 1) * self.R_size]
                )
                # [bs=32, 1*256]
                TMR_group_motions = F.normalize(
                    TMR_all_genmotions[i * self.R_size : (i + 1) * self.R_size]
                )

            dist_mat = euclidean_distance_matrix(
                group_texts, group_motions
            ).nan_to_num()
            if self.use_TMR:
                TMR_dist_mat = euclidean_distance_matrix(
                    TMR_group_texts, TMR_group_motions
                ).nan_to_num()
            # print(dist_mat[:5])
            self.Matching_score += dist_mat.trace()
            argsmax = torch.argsort(dist_mat, dim=1)
            top_k_mat += calculate_top_k(argsmax, top_k=self.top_k).sum(axis=0)

            if self.use_TMR:
                self.TMR_Matching_score += TMR_dist_mat.trace()
                TMR_argsmax = torch.argsort(TMR_dist_mat, dim=1)
                TMR_top_k_mat += calculate_top_k(TMR_argsmax, top_k=self.top_k).sum(
                    axis=0
                )

        R_count = count_seq // self.R_size * self.R_size
        metrics["Matching_score"] = self.Matching_score / R_count
        if self.use_TMR:
            metrics["TMR_Matching_score"] = self.TMR_Matching_score / R_count
        for k in range(self.top_k):
            metrics[f"R_precision_top_{str(k+1)}"] = top_k_mat[k] / R_count
            if self.use_TMR:
                metrics[f"TMR_R_precision_top_{str(k+1)}"] = TMR_top_k_mat[k] / R_count

        # Compute r-precision with gt
        assert count_seq > self.R_size
        top_k_mat = torch.zeros((self.top_k,))
        TMR_top_k_mat = torch.zeros((self.top_k,))
        for i in range(count_seq // self.R_size):
            # [bs=32, 1*256]
            group_texts = all_texts[i * self.R_size : (i + 1) * self.R_size]
            # [bs=32, 1*256]
            group_motions = all_gtmotions[i * self.R_size : (i + 1) * self.R_size]
            if self.use_TMR:
                # [bs=32, 1*256]
                TMR_group_texts = F.normalize(
                    TMR_all_texts[i * self.R_size : (i + 1) * self.R_size]
                )
                # [bs=32, 1*256]
                TMR_group_motions = F.normalize(
                    TMR_all_gtmotions[i * self.R_size : (i + 1) * self.R_size]
                )
            # [bs=32, 32]
            dist_mat = euclidean_distance_matrix(
                group_texts, group_motions
            ).nan_to_num()
            if self.use_TMR:
                # [bs=32, 32]
                TMR_dist_mat = euclidean_distance_matrix(
                    TMR_group_texts, TMR_group_motions
                ).nan_to_num()
            # match score
            self.gt_Matching_score += dist_mat.trace()
            argsmax = torch.argsort(dist_mat, dim=1)
            top_k_mat += calculate_top_k(argsmax, top_k=self.top_k).sum(axis=0)

            if self.use_TMR:
                self.TMR_gt_Matching_score += TMR_dist_mat.trace()
                TMR_argsmax = torch.argsort(TMR_dist_mat, dim=1)
                TMR_top_k_mat += calculate_top_k(TMR_argsmax, top_k=self.top_k).sum(
                    axis=0
                )
        metrics["gt_Matching_score"] = self.gt_Matching_score / R_count
        if self.use_TMR:
            metrics["TMR_gt_Matching_score"] = self.TMR_gt_Matching_score / R_count

        for k in range(self.top_k):
            metrics[f"gt_R_precision_top_{str(k+1)}"] = top_k_mat[k] / R_count
            if self.use_TMR:
                metrics[f"TMR_gt_R_precision_top_{str(k+1)}"] = (
                    TMR_top_k_mat[k] / R_count
                )

        # tensor -> numpy for FID
        all_genmotions = all_genmotions.numpy()
        all_gtmotions = all_gtmotions.numpy()
        if self.use_TMR:
            TMR_all_genmotions = TMR_all_genmotions.numpy()
            TMR_all_gtmotions = TMR_all_gtmotions.numpy()

        # Compute fid
        mu, cov = calculate_activation_statistics_np(all_genmotions)
        gt_mu, gt_cov = calculate_activation_statistics_np(all_gtmotions)

        if self.use_TMR:
            TMR_mu, TMR_cov = calculate_activation_statistics_np(TMR_all_genmotions)
            TMR_gt_mu, TMR_gt_cov = calculate_activation_statistics_np(
                TMR_all_gtmotions
            )

        # Compute diversity
        assert count_seq > self.diversity_times
        metrics["Diversity"] = calculate_diversity_np(
            all_genmotions, self.diversity_times
        )
        metrics["gt_Diversity"] = calculate_diversity_np(
            all_gtmotions, self.diversity_times
        )

        return {**metrics}

    def update(
        self,
        text_embeddings: Tensor,
        recmotion_embeddings: Tensor,
        gtmotion_embeddings: Tensor,
        lengths: List[int],
        TMR_motion_embedding=None,
        TMR_GT_motion_embedding=None,
        TMR_text_embedding=None,
    ):

        self.count += sum(lengths)
        self.count_seq += len(lengths)

        # [bs, nlatent*ndim] <= [bs, nlatent, ndim]
        text_embeddings = torch.flatten(text_embeddings, start_dim=1).detach()
        recmotion_embeddings = torch.flatten(recmotion_embeddings, start_dim=1).detach()
        gtmotion_embeddings = torch.flatten(gtmotion_embeddings, start_dim=1).detach()

        # store all texts and motions
        self.text_embeddings.append(text_embeddings)
        self.recmotion_embeddings.append(recmotion_embeddings)
        self.gtmotion_embeddings.append(gtmotion_embeddings)

        if self.use_TMR:
            self.TMR_text_embeddings.append(TMR_text_embedding)
            self.TMR_recmotion_embeddings.append(TMR_motion_embedding)
            self.TMR_gtmotion_embeddings.append(TMR_GT_motion_embedding)
