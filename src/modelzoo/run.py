import logging
import shutil
from pathlib import Path
from typing import List, Optional

import git
import hydra
import omegaconf
import pandas as pd
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, ListConfig
from pytorch_lightning import Callback

from nn_core.callbacks import NNTemplateCore
from nn_core.common import PROJECT_ROOT
from nn_core.common.utils import enforce_tags, seed_index_everything
from nn_core.model_logging import NNLogger
from nn_core.serialization import NNCheckpointIO

# Force the execution of __init__.py if this file is executed directly.
import modelzoo  # noqa
from modelzoo.data.vision.datamodule import MetaData

torch.set_float32_matmul_precision("high")

pylogger = logging.getLogger(__name__)

MODELZOO_ROOT: Path = Path("models")


def build_callbacks(cfg: ListConfig, *args: Callback) -> List[Callback]:
    """Instantiate the callbacks given their configuration.

    Args:
        cfg: a list of callbacks instantiable configuration
        *args: a list of extra callbacks already instantiated

    Returns:
        the complete list of callbacks to use
    """
    callbacks: List[Callback] = list(args)

    for callback in cfg:
        pylogger.info(f"Adding callback <{callback['_target_'].split('.')[-1]}>")
        callbacks.append(hydra.utils.instantiate(callback, _recursive_=False))

    return callbacks


def run(cfg: DictConfig) -> str:
    """Generic train loop.

    Args:
        cfg: run configuration, defined by Hydra in /conf

    Returns:
        the run directory inside the storage_dir used by the current experiment
    """
    seed_index_everything(cfg.train)

    fast_dev_run: bool = cfg.train.trainer.fast_dev_run
    if fast_dev_run:
        pylogger.info(f"Debug mode <{cfg.train.trainer.fast_dev_run=}>. Forcing debugger friendly configuration!")
        # Debuggers don't like GPUs nor multiprocessing
        cfg.train.trainer.accelerator = "cpu"
        cfg.nn.data.num_workers.train = 0
        cfg.nn.data.num_workers.val = 0
        cfg.nn.data.num_workers.test = 0

    cfg.core.tags = enforce_tags(cfg.core.get("tags", None))

    # Instantiate datamodule
    pylogger.info(f"Instantiating <{cfg.nn.data['_target_']}>")
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(cfg.nn.data, _recursive_=False)

    metadata: Optional[MetaData] = getattr(datamodule, "metadata", None)
    if metadata is None:
        pylogger.warning(f"No 'metadata' attribute found in datamodule <{datamodule.__class__.__name__}>")

    # Instantiate model
    pylogger.info(f"Instantiating <{cfg.nn.module['_target_']}>")
    model: pl.LightningModule = hydra.utils.instantiate(cfg.nn.module, _recursive_=False, metadata=metadata)

    # Instantiate the callbacks
    template_core: NNTemplateCore = NNTemplateCore(
        restore_cfg=cfg.train.get("restore", None),
    )
    callbacks: List[Callback] = build_callbacks(cfg.train.callbacks, template_core)

    storage_dir: str = cfg.core.storage_dir

    logger: NNLogger = NNLogger(logging_cfg=cfg.train.logging, cfg=cfg, resume_id=template_core.resume_id)

    pylogger.info("Instantiating the <Trainer>")
    trainer = pl.Trainer(
        default_root_dir=storage_dir,
        plugins=[NNCheckpointIO(jailing_dir=logger.run_dir)],
        logger=logger,
        callbacks=callbacks,
        **cfg.train.trainer,
    )

    pylogger.info("Starting training!")
    trainer.fit(
        model=model,
        datamodule=datamodule,
        ckpt_path=template_core.trainer_ckpt_path,
    )

    if fast_dev_run:
        pylogger.info("Skipping testing in 'fast_dev_run' mode!")
    else:
        if "test" in cfg.nn.data.datasets and trainer.checkpoint_callback.best_model_path is not None:
            pylogger.info("Starting testing!")
            trainer.test(datamodule=datamodule)

    # Store model locally
    if trainer.checkpoint_callback.best_model_path is not None:
        wandb_id = f"{trainer.logger.version}"

        ckpt_relpath = MODELZOO_ROOT / "checkpoints" / f"{wandb_id}.ckpt.zip"
        pylogger.info(f"Storing the best model into modelzoo storage: {ckpt_relpath}")

        filepath = PROJECT_ROOT / MODELZOO_ROOT / "index.csv"
        ckptpath = PROJECT_ROOT / ckpt_relpath
        ckptpath.parent.mkdir(exist_ok=True, parents=True)

        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha

        modelrow = {
            "timestamp": pd.Timestamp.utcnow(),
            "seed_index": cfg.train.seed_index,
            "entity": f"{trainer.logger.experiment.entity}",
            "project_name": f"{trainer.logger.experiment.project_name()}",
            "wandb_id": wandb_id,
            "name": cfg.core.name,
            "dataset": cfg.nn.data.datasets.hf.name,  # TODO: generalize modelzoo to any datasets not only hf
            "score": trainer.checkpoint_callback.best_model_score.detach().cpu().item(),
            "module": cfg.nn.module._target_,
            "model": cfg.nn.module.model._target_,
            "parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "path": str(ckpt_relpath),
            "version": cfg.core.version,
            "commit": sha,
            "tags": ",".join(cfg.core.tags),
        }
        df = pd.DataFrame(modelrow, index=[0])

        df.to_csv(filepath, mode="a", index=False, sep="\t", header=not filepath.exists())

        # TODO: grab from NNIOCheckpoint the correct ckpt name
        shutil.copyfile(f"{trainer.checkpoint_callback.best_model_path}.zip", ckptpath)

    # Logger closing to release resources/avoid multi-run conflicts
    if logger is not None:
        logger.experiment.finish()

    return logger.run_dir


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default", version_base="1.1")
def main(cfg: omegaconf.DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
