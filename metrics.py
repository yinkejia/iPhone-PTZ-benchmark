from typing import Literal, List

import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics.functional.image.lpips import _NoTrainLpips
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.metric import Metric
from torchmetrics.utilities import dim_zero_cat
from torchmetrics.utilities.imports import _TORCHVISION_AVAILABLE

from torchvision.models.inception import inception_v3
from torchvision import transforms
from scipy import linalg
from PIL import Image
from torch import nn

def compute_psnr(
    preds: torch.Tensor,
    targets: torch.Tensor,
    masks: torch.Tensor | None = None,
) -> float:
    """
    Args:
        preds (torch.Tensor): (..., 3) predicted images in [0, 1].
        targets (torch.Tensor): (..., 3) target images in [0, 1].
        masks (torch.Tensor | None): (...,) optional binary masks where the
            1-regions will be taken into account.

    Returns:
        psnr (float): Peak signal-to-noise ratio.
    """
    if masks is None:
        masks = torch.ones_like(preds[..., 0])
    return (
        -10.0
        * torch.log(
            F.mse_loss(
                preds * masks[..., None],
                targets * masks[..., None],
                reduction="sum",
            )
            / masks.sum().clamp(min=1.0)
            / 3.0
        )
        / np.log(10.0)
    ).item()

class mPSNR(PeakSignalNoiseRatio):
    sum_squared_error: list[torch.Tensor]
    total: list[torch.Tensor]

    def __init__(self, **kwargs) -> None:
        super().__init__(
            data_range=1.0,
            base=10.0,
            dim=None,
            reduction="elementwise_mean",
            **kwargs,
        )
        self.add_state("sum_squared_error", default=[], dist_reduce_fx="cat")
        self.add_state("total", default=[], dist_reduce_fx="cat")

    def __len__(self) -> int:
        return len(self.total)

    def update(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        masks: torch.Tensor | None = None,
    ):
        """Update state with predictions and targets.

        Args:
            preds (torch.Tensor): (..., 3) float32 predicted images.
            targets (torch.Tensor): (..., 3) float32 target images.
            masks (torch.Tensor | None): (...,) optional binary masks where the
                1-regions will be taken into account.
        """
        if masks is None:
            masks = torch.ones_like(preds[..., 0])
        self.sum_squared_error.append(
            torch.sum(torch.pow((preds - targets) * masks[..., None], 2))
        )
        self.total.append(masks.sum().to(torch.int64) * 3)

    def compute(self) -> torch.Tensor:
        """Compute peak signal-to-noise ratio over state."""
        sum_squared_error = dim_zero_cat(self.sum_squared_error)
        total = dim_zero_cat(self.total)
        return -10.0 * torch.log(sum_squared_error / total).mean() / np.log(10.0)

class mLPIPS(Metric):
    sum_scores: list[torch.Tensor]
    total: list[torch.Tensor]

    def __init__(
        self,
        net_type: Literal["vgg", "alex", "squeeze"] = "alex",
        **kwargs,
    ):
        super().__init__(**kwargs)

        if not _TORCHVISION_AVAILABLE:
            raise ModuleNotFoundError(
                "LPIPS metric requires that torchvision is installed."
                " Either install as `pip install torchmetrics[image]` or `pip install torchvision`."
            )

        valid_net_type = ("vgg", "alex", "squeeze")
        if net_type not in valid_net_type:
            raise ValueError(
                f"Argument `net_type` must be one of {valid_net_type}, but got {net_type}."
            )
        self.net = _NoTrainLpips(net=net_type, spatial=True)

        self.add_state("sum_scores", [], dist_reduce_fx="cat")
        self.add_state("total", [], dist_reduce_fx="cat")

    def __len__(self) -> int:
        return len(self.total)

    def update(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        masks: torch.Tensor | None = None,
    ):
        """Update internal states with lpips scores.

        Args:
            preds (torch.Tensor): (B, H, W, 3) float32 predicted images.
            targets (torch.Tensor): (B, H, W, 3) float32 target images.
            masks (torch.Tensor | None): (B, H, W) optional float32 binary
                masks where the 1-regions will be taken into account.
        """
        if masks is None:
            masks = torch.ones_like(preds[..., 0])
        scores = self.net(
            (preds * masks[..., None]).permute(0, 3, 1, 2),
            (targets * masks[..., None]).permute(0, 3, 1, 2),
            normalize=True,
        )
        self.sum_scores.append((scores * masks[:, None]).sum())
        self.total.append(masks.sum().to(torch.int64))

    def compute(self) -> torch.Tensor:
        """Compute final perceptual similarity metric."""
        return (
            torch.tensor(self.sum_scores, device=self.device)
            / torch.tensor(self.total, device=self.device)
        ).mean()


class FIDCalculator:
    """
    Compute Fréchet Inception Distance (FID) between two sets of images.

    Input:
        - real_images: list of PIL.Image (ground truth)
        - fake_images: list of PIL.Image (generated)

    Output:
        - fid_score: float
    """

    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = inception_v3(pretrained=True, transform_input=False).to(self.device)
        self.model.fc = torch.nn.Identity()  # remove classification head
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])

    @torch.no_grad()
    def _extract_features(self, images: List[Image.Image], batch_size: int = 32):
        feats = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            imgs = torch.stack([self.transform(img) for img in batch]).to(self.device)
            with torch.cuda.amp.autocast(enabled=self.device == "cuda"):
                feat = self.model(imgs).detach().cpu().numpy()
            feats.append(feat)
        return np.concatenate(feats, axis=0)

    @staticmethod
    def _compute_fid(mu1, sigma1, mu2, sigma2):
        diff = mu1 - mu2
        covmean = linalg.sqrtm(sigma1 @ sigma2)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        fid = np.sum(diff**2) + np.trace(sigma1 + sigma2 - 2 * covmean)
        return float(fid)

    def compute_fid(self, real_images: List[Image.Image], fake_images: List[Image.Image]):
        assert len(real_images) > 0 and len(fake_images) > 0, "Empty image lists!"
        print(f"Extracting Inception features from {len(real_images)} real and {len(fake_images)} fake images...")

        real_feats = self._extract_features(real_images)
        fake_feats = self._extract_features(fake_images)

        mu1, sigma1 = real_feats.mean(axis=0), np.cov(real_feats, rowvar=False)
        mu2, sigma2 = fake_feats.mean(axis=0), np.cov(fake_feats, rowvar=False)

        fid = self._compute_fid(mu1, sigma1, mu2, sigma2)
        return fid