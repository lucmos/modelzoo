import importlib
from pathlib import Path
from typing import Any, Dict, Tuple

import pytorch_lightning as pl

from nn_core.common import PROJECT_ROOT
from nn_core.serialization import NNCheckpointIO, load_model

from modelzoo.utils.wandb_utils import checkpoint_selection


def load_wandb_ckpt(
    entity: str,
    project: str,
    run_id: str,
) -> Tuple[pl.LightningModule, Dict[str, Any]]:
    # Ensure the wandb directory exists
    (PROJECT_ROOT / "wandb").mkdir(exist_ok=True)

    # Download or detect the checkpoint
    ckpt_path = checkpoint_selection(entity, project, run_id)

    return load_local_ckpt(ckpt_path)


def load_local_ckpt(ckpt_path: Path, strict: bool) -> Tuple[pl.LightningModule, Dict[str, Any]]:
    ckpt = NNCheckpointIO.load(ckpt_path)
    clspath = ckpt["cfg"]["nn"]["module"]["_target_"]
    modulename, classname = clspath.rsplit(".", 1)

    clazz = getattr(importlib.import_module(modulename), classname)
    return load_model(
        module_class=clazz,
        checkpoint_path=ckpt_path,
        strict=strict,
        map_location="cpu",
    )
