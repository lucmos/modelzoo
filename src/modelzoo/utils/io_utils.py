from collections import namedtuple
from pathlib import Path
from typing import Callable, Dict

import numpy as np
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk

from modelzoo import PACKAGE_ROOT
from modelzoo.data.datasetdict import MyDatasetDict
from modelzoo.utils.utils import compute_cfg_hash

DatasetParams = namedtuple("DatasetParams", ["name", "fine_grained", "train_split", "test_split", "hf_key"])


def convert_to_rgb(image):
    if image.mode != "RGB":
        return image.convert("RGB")
    return np.asarray(image)


def preprocess_dataset(
    dataset: Dataset,
    cfg: Dict,
) -> Dataset:
    dataset = dataset.rename_column(cfg.label_key, cfg.standard_y_key)
    dataset = dataset.rename_column(cfg.image_key, cfg.standard_x_key)

    return dataset


def add_ids_to_dataset(dataset):
    N = len(dataset["train"])
    M = len(dataset["test"])
    indices = {"train": list(range(N)), "test": list(range(N, N + M))}

    for mode in ["train", "test"]:
        dataset[mode] = dataset[mode].map(
            lambda _, ind: {"id": indices[mode][ind]},
            with_indices=True,
            keep_in_memory=True,
            load_from_cache_file=False,
        )

    return dataset


def transform_dataset(cfg: Dict, dataset: Dataset, transform_fn: Callable):
    dataset_params: DatasetParams = DatasetParams(
        cfg.ref,
        None,
        cfg.train_split,
        cfg.test_split,
        (cfg.ref,),
    )

    transforms_hash = compute_cfg_hash(cfg.transforms)

    DATASET_KEY = "_".join(
        map(
            str,
            [v for k, v in dataset_params._asdict().items() if k != "hf_key" and v is not None],
        )
    )
    DATASET_DIR: Path = PACKAGE_ROOT / "data" / "cache" / f"{DATASET_KEY}_transforms_{transforms_hash}"

    if not DATASET_DIR.exists():
        dataset = dataset.map(
            function=transform_fn,
            desc="Applying transform",
            num_proc=1,
            writer_batch_size=100,
            # load_from_cache_file=False,
            # keep_in_memory=True,
        )
        save_dataset_to_disk(dataset, DATASET_DIR)
    else:
        dataset: Dataset = load_from_disk(dataset_path=str(DATASET_DIR))
    return dataset


def load_data(cfg):
    dataset_params: DatasetParams = DatasetParams(
        cfg.ref,
        None,
        cfg.train_split,
        cfg.test_split,
        (cfg.ref,),
    )
    DATASET_KEY = "_".join(
        map(
            str,
            [v for k, v in dataset_params._asdict().items() if k != "hf_key" and v is not None],
        )
    )
    DATASET_DIR: Path = PACKAGE_ROOT / "data" / "datasets" / DATASET_KEY

    if not DATASET_DIR.exists():
        train_dataset = load_dataset(
            dataset_params.name,
            split=dataset_params.train_split,
            use_auth_token=True,
        )
        test_dataset = load_dataset(dataset_params.name, split=dataset_params.test_split)
        dataset: DatasetDict = MyDatasetDict(train=train_dataset, test=test_dataset)

        dataset = preprocess_dataset(dataset, cfg)
        dataset = add_ids_to_dataset(dataset)

        save_dataset_to_disk(dataset, DATASET_DIR)
    else:
        dataset: Dataset = load_from_disk(dataset_path=str(DATASET_DIR))

    return dataset


def save_dataset_to_disk(dataset: MyDatasetDict, output_path: Path):
    if not isinstance(output_path, Path):
        output_path = Path(output_path)

    if not output_path.exists():
        output_path.mkdir(parents=True)

    dataset.save_to_disk(output_path)
