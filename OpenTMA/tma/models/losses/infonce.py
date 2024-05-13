import torch
import torch.nn.functional as F
import numpy as np


class InfoNCE:
    """
    This class implements the InfoNCE loss function.

    Attributes:
    - t: a temperature parameter for the softmax function in the loss calculation.

    Methods:
    - __call__: computes the InfoNCE loss given the motion and text features, and an optional distance matrix.
    """

    def __init__(self, t):
        """
        Initializes the InfoNCE object with a given temperature parameter.

        Inputs:
        - t: a temperature parameter for the softmax function in the loss calculation.
        """
        self.t = t

    def __call__(self, f, dist):
        """
        Computes the InfoNCE loss given the motion and text features, and an optional distance matrix.

        Inputs:
        - f: a tuple containing the motion and text features. Each feature is a 2D tensor of shape (N, d).
        - dist: an optional distance matrix. If provided, it is used to mask the logits.

        Outputs:
        - loss_m: the InfoNCE loss computed using the motion features.
        - loss_t: the InfoNCE loss computed using the text features.
        """
        t = self.t
        f_motion, f_text = f[0], f[1]

        N, d = f_motion.shape[0], f_motion.shape[1]

        # Normalize the motion and text features
        Emb_motion = F.normalize(f_motion, dim=1)
        Emb_text = F.normalize(f_text, dim=1)

        # Compute the logits as the dot product of the normalized features
        t = torch.tensor(t).to(f_motion.device)
        logits = torch.mm(Emb_motion, Emb_text.T)

        # If a distance matrix is provided, use it to mask the logits
        if dist is not None:
            text_logits = dist.detach()
            mask = torch.where(
                torch.logical_and(text_logits > 0.85, text_logits < 1.0 - 1e-100),
                torch.tensor(float("-inf")).to(f_motion.device),
                torch.tensor(1.0e100).to(f_motion.device),
            )
            mask.diagonal().fill_(float("inf"))
            logits = torch.min(mask, logits)

        N = f_motion.shape[0]

        # Compute the labels as the indices of the features
        labels = torch.arange(N).to(f_motion.device)

        # Compute the InfoNCE loss for the motion and text features
        loss_m = F.cross_entropy(logits / t, labels)
        loss_t = F.cross_entropy(logits.T / t, labels)

        loss = (loss_m + loss_t) / 2

        return loss

    def __repr__(self):
        return "InfoNCE()"
