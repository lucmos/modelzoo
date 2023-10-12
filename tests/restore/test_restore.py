from pathlib import Path

import hydra
import pytest
from omegaconf import OmegaConf

from modelzoo.utils.io_model import load_local_ckpt

CHECKPOINTS_TO_TEST = list(Path(__file__).parent.rglob("*.zip"))


@pytest.mark.parametrize("ckpt_path", CHECKPOINTS_TO_TEST)
def test_restore(ckpt_path: Path):
    # Load model
    model, ckpt = load_local_ckpt(ckpt_path, strict=False)
    assert model is not None
    assert ckpt is not None

    # Load config
    cfg = OmegaConf.create(ckpt["cfg"])
    assert cfg is not None

    # Instantiate datamodule
    datamodule = hydra.utils.instantiate(OmegaConf.to_container(cfg.nn.data), _recursive_=False)
    datamodule.setup(stage="fit")
    val_loader = datamodule.val_dataloader()[0]
    assert val_loader is not None

    # Get a batch
    batch = next(iter(val_loader))
    assert batch is not None

    # Forward pass
    reconstruction = model(batch["x"])
    assert reconstruction is not None
