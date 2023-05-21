import hashlib
import json
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf


def compute_cfg_hash(cfg: DictConfig) -> str:
    """Compute a hash of the config to use as a key for caching"""
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg_hash = hashlib.md5(json.dumps(cfg_dict, sort_keys=True).encode("utf-8")).hexdigest()
    return cfg_hash


def detach_tensors(x: Any) -> Any:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    else:
        return x
