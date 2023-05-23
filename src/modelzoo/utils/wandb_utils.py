import datetime
from pathlib import Path
from typing import List

import wandb
from omegaconf import OmegaConf
from tqdm import tqdm

from nn_core.common import PROJECT_ROOT
from nn_core.serialization import NNCheckpointIO, load_model

WANDB_DIR: Path = PROJECT_ROOT / "wandb"


def get_run_dir(entity: str, project: str, run_id: str) -> Path:
    """Get run directory.

    :param run_path: "entity/project/run_id"
    :return:
    """
    api = wandb.Api()
    run = api.run(path=f"{entity}/{project}/{run_id}")
    created_at: datetime = datetime.datetime.strptime(run.created_at, "%Y-%m-%dT%H:%M:%S")

    timestamp: str = created_at.strftime("%Y%m%d_%H%M%S")

    matching_runs: List[Path] = [item for item in WANDB_DIR.iterdir() if item.is_dir() and item.name.endswith(run_id)]

    if len(matching_runs) > 1:
        raise RuntimeError(f"More than one run matching unique id {run_id}! Are you sure about that?")

    if len(matching_runs) == 1:
        return matching_runs[0]

    run_dir: Path = WANDB_DIR / f"restored-{timestamp}-{run.id}" / "files"
    files = [file for file in run.files() if "checkpoint" in file.name]
    for file in tqdm(files, desc="Downloading files..."):
        file.download(root=run_dir)
    return run_dir


def local_checkpoint_selection(run_dir: Path, idx: int) -> Path:
    checkpoint_paths: List[Path] = list(run_dir.rglob("checkpoints/*"))
    if len(checkpoint_paths) == 0:
        raise FileNotFoundError(f"There's no checkpoint under {run_dir}! Are you sure the restore was successful?")
    return checkpoint_paths[idx]


def checkpoint_selection(entity: str, project: str, run_id: str):
    rud_dir = get_run_dir(entity=entity, project=project, run_id=run_id)
    ckpt_path = local_checkpoint_selection(rud_dir, 0)
    return ckpt_path


def load_model_cfg(module_class, ckpt_path, map_location="cpu"):
    model = load_model(module_class=module_class, checkpoint_path=ckpt_path, map_location=map_location)
    cfg = NNCheckpointIO.load(path=ckpt_path, map_location=map_location)["cfg"]

    return model, OmegaConf.create(cfg)
