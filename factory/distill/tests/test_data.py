import torch

from distill.build_dataset import build_index
from distill.data import IMG_SIZE, CropsDataset, make_loader


def test_dataset_shapes_and_labels(crops_root):
    idx = build_index(crops_root, "labels-test.jsonl")
    ds = CropsDataset(crops_root, idx["records"], train=False)
    x, g, f = ds[0]
    assert x.shape == (3, IMG_SIZE, IMG_SIZE) and x.dtype == torch.float32
    assert isinstance(g, int) and 0 <= g < 5 and f in (0, 1)
    # val transform is deterministic
    assert torch.equal(ds[0][0], ds[0][0])


def test_train_transform_is_stochastic(crops_root):
    idx = build_index(crops_root, "labels-test.jsonl")
    ds = CropsDataset(crops_root, idx["records"], train=True)
    torch.manual_seed(1)
    a = ds[0][0]
    torch.manual_seed(2)
    b = ds[0][0]
    assert not torch.equal(a, b)


def test_loader_batches(crops_root):
    idx = build_index(crops_root, "labels-test.jsonl")
    dl = make_loader(crops_root, idx["records"], train=True, batch_size=4, workers=0)
    x, g, f = next(iter(dl))
    assert x.shape == (4, 3, IMG_SIZE, IMG_SIZE) and g.shape == (4,) and f.shape == (4,)
