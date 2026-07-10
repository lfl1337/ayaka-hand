import pytest
import torch

from distill.model import DistillStudent, count_params


def test_forward_shapes():
    m = DistillStudent()
    g, f = m(torch.randn(2, 3, 160, 160))
    assert g.shape == (2, 5) and f.shape == (2, 2)


def test_param_count_is_the_communicated_number():
    # Measured truth: ~1.36M params, NOT the "~2.6M" that leaked into the deck.
    # 2.6M is the FULL mobilenet_v2(width_mult=0.75) incl. its 1000-class ImageNet
    # classifier (1.28M params) — which this student discards (features-only + two
    # tiny heads). README/deck must state 1.36M. Band guards against drift.
    n = count_params(DistillStudent())
    assert 1_200_000 < n < 1_600_000, n


def test_heads_share_backbone():
    m = DistillStudent()
    x = torch.randn(1, 3, 160, 160)
    g1, _ = m(x)
    assert g1.requires_grad


# Per-arch param bands. timm nets are constructed with load_pretrained=False so
# the tests never hit the network; the ImageNet weights only matter at train time.
@pytest.mark.parametrize("arch,lo,hi", [
    ("timm_mnv2_100", 2_000_000, 2_600_000),
    ("timm_mnv2_050", 600_000, 1_200_000),
])
def test_timm_arch_shapes_and_params(arch, lo, hi):
    m = DistillStudent(arch=arch, load_pretrained=False)
    g, f = m(torch.randn(2, 3, 160, 160))
    assert g.shape == (2, 5) and f.shape == (2, 2)
    n = count_params(m)
    assert lo < n < hi, n


def test_invalid_arch_raises():
    with pytest.raises(ValueError):
        DistillStudent(arch="resnet")
