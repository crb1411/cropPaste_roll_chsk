import torch
import torch.nn as nn
import logging

logger = logging.getLogger("dinov3")


def entropy(p: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    p: [B, K], assumed to be probability distribution (after softmax)
    return: [B], entropy for each sample
    """
    assert p is not None, "sharpen head value must be provided"
    
    return -(p * (p + eps).log()).sum(dim=-1)


def sharpness_loss_from_entropy(H: torch.Tensor, H_target: float) -> torch.Tensor:
    """
    H: [B], entropy
    Enforce: H <= H_target
    """
    return torch.relu(H - H_target).mean()


class SharpnessLoss(nn.Module):
    """
    Sharpness regularization for prototype distributions.

    Goal:
        - Encourage per-sample distribution to be sharp (low entropy)
        - While DT-CHSK enforces global uniform usage

    Loss:
        L = E_b [ max( H(p_b) - H_target , 0 ) ]

    Interpretation:
        - If entropy is already smaller than target → no penalty
        - Only penalize overly flat distributions
    """

    def __init__(
        self,
        H_target: float = 1.0,
        eps: float = 1e-8,
        weight: float = 1.0,
    ):
        super().__init__()
        self.H_target = float(H_target)
        self.eps = eps
        self.weight = weight

    def forward(
        self,
        sharpen_data,
        *,
        iteration: int = 0,
        logger_freq: int = 0,
        logger_loss: str | None = None,
        H_target: float | None = None,
    ):
        """
        sharpen_data:
            - Tensor: [B, K]
            - or dict[str, Tensor]: each Tensor is [B, K]

        Returns:
            total_loss: scalar tensor
            loss_dict: dict[str, scalar tensor]
        """

        # allow runtime override (for curriculum)
        H_target = self.H_target if H_target is None else float(H_target)

        loss_dict = {}

        if isinstance(sharpen_data, dict):
            for key, p in sharpen_data.items():
                H = entropy(p, eps=self.eps)
                loss = sharpness_loss_from_entropy(H, H_target)
                loss_dict[key] = loss
        else:
            H = entropy(sharpen_data, eps=self.eps)
            loss_dict["sharpness"] = sharpness_loss_from_entropy(H, H_target)

        # aggregate
        total_loss = sum(loss_dict.values()) * self.weight

        # ===============================
        # logging
        # ===============================
        if logger is not None and logger_freq > 0 and iteration % logger_freq == 0:
            tag = f"[{logger_loss}] " if logger_loss else ""
            msg = f"{tag}[SharpnessLoss] H_target={H_target:.3f} | "
            msg += " | ".join(
                [f"{k}: {v.item():.4e}" for k, v in loss_dict.items()]
            )
            msg += f" | total: {total_loss.item():.4e}"
            logger.info(msg)

        return total_loss, loss_dict
