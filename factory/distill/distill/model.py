"""The edge student: MobileNetV2 backbone + two linear heads, three arch variants.

- arch="tv" (default): torchvision width-0.75, FROM SCRATCH. ~1.36M params
  (measured 1,364,391). The "~2.6M" that circulated is the FULL width-0.75 net
  incl. its discarded 1000-class ImageNet classifier; this student is
  features-only + two tiny heads. torchvision ships no pretrained 0.75 weights.
- arch="timm_mnv2_100": timm mobilenetv2_100 (width 1.0), ImageNet-pretrainable.
- arch="timm_mnv2_050": timm mobilenetv2_050 (width 0.5), ImageNet-pretrainable.

The pretrained timm variants exist because 11k labeled crops is too little for
from-scratch grip learning (kd-v1 lost the lookup strawman on grip). timm has no
width-0.75 ImageNet checkpoint, so we ablate 1.0 vs 0.5 and pick empirically.
timm fixes the final conv at 1280 features for every width, so both heads are
identical across archs.
"""
import torch
from torch import nn
from torchvision.models import mobilenet_v2

from distill.schema import FORCES, GRIPS

_TIMM_MODELS = {"timm_mnv2_100": "mobilenetv2_100", "timm_mnv2_050": "mobilenetv2_050"}
_FEAT_DIM = 1280


class DistillStudent(nn.Module):
    def __init__(self, arch: str = "tv", load_pretrained: bool = False) -> None:
        super().__init__()
        self.arch = arch
        if arch == "tv":
            base = mobilenet_v2(weights=None, width_mult=0.75)
            self.features = base.features
            self.pool = nn.AdaptiveAvgPool2d(1)
            feat_dim = base.last_channel                  # 1280 for width<=1.0 (floor at 1280)
        elif arch in _TIMM_MODELS:
            import timm  # lazy: arch="tv" must work without timm importable
            self.backbone = timm.create_model(
                _TIMM_MODELS[arch], pretrained=load_pretrained, num_classes=0, global_pool="avg")
            assert self.backbone.num_features == _FEAT_DIM, self.backbone.num_features
            feat_dim = self.backbone.num_features
        else:
            raise ValueError(f"unknown arch {arch!r} (expected 'tv', {', '.join(map(repr, _TIMM_MODELS))})")
        self.dropout = nn.Dropout(0.2)
        self.grip_head = nn.Linear(feat_dim, len(GRIPS))
        self.force_head = nn.Linear(feat_dim, len(FORCES))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.arch == "tv":
            z = self.dropout(torch.flatten(self.pool(self.features(x)), 1))
        else:
            z = self.dropout(self.backbone(x))            # timm avg-pools to [B, 1280]
        return self.grip_head(z), self.force_head(z)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
