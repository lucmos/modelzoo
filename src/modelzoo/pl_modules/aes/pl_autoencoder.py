import logging
from typing import Any, Mapping, Optional, Set

import hydra
import omegaconf
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from nn_core.common import PROJECT_ROOT
from nn_core.model_logging import NNLogger

from modelzoo.data.vision.datamodule import MetaData
from modelzoo.modules.aes.enumerations import Output, Stage, SupportedViz
from modelzoo.pl_modules.aes.pl_visualizations import (
    add_2D_latents,
    aggregate,
    on_fit_end_viz,
    on_fit_start_viz,
    validation_epoch_end_viz,
)
from modelzoo.pl_modules.pl_abstractmodule import AbstractLightningModule
from modelzoo.utils.utils import detach_tensors

pylogger = logging.getLogger(__name__)


class LightningAutoencoder(AbstractLightningModule):
    logger: NNLogger

    def __init__(self, metadata: Optional[MetaData] = None, *args, **kwargs) -> None:
        super().__init__(metadata, *args, **kwargs)

        self.autoencoder = hydra.utils.instantiate(
            kwargs["model"] if "model" in kwargs else kwargs["autoencoder"], metadata=metadata, _recursive_=False
        )

        self.reconstruction_quality_metrics = {
            "mse": F.mse_loss,
            "l1": F.l1_loss,
        }

        self.register_buffer("anchors_images", self.metadata.anchors_images)
        self.register_buffer("anchors_latents", self.metadata.anchors_latents)
        self.register_buffer("fixed_images", self.metadata.fixed_images)

        self.loss = self.autoencoder.loss_function

        self.supported_viz = self.supported_viz()
        pylogger.info(f"Enabled visualizations: {str(sorted(x.value for x in self.supported_viz))}")

        self.validation_step_outputs = []

    def supported_viz(self) -> Set[SupportedViz]:
        supported_viz = set()

        if self.fixed_images is not None:
            supported_viz.add(SupportedViz.VALIDATION_IMAGES_SOURCE)

        if self.anchors_images is not None:
            supported_viz.add(SupportedViz.ANCHORS_SOURCE)

        # if isinstance(self.autoencoder, RAE):
        #     supported_viz.add(SupportedViz.INVARIANT_LATENT_DISTRIBUTION)

        # supported_viz.add(SupportedViz.LATENT_EVOLUTION_PLOTLY_ANIMATION)
        supported_viz.add(SupportedViz.ANCHORS_RECONSTRUCTED)
        supported_viz.add(SupportedViz.VALIDATION_IMAGES_RECONSTRUCTED)
        # supported_viz.add(SupportedViz.ANCHORS_SELF_INNER_PRODUCT)
        # supported_viz.add(SupportedViz.ANCHORS_VALIDATION_IMAGES_INNER_PRODUCT)
        # supported_viz.add(SupportedViz.ANCHORS_SELF_INNER_PRODUCT_NORMALIZED)
        # supported_viz.add(SupportedViz.ANCHORS_VALIDATION_IMAGES_INNER_PRODUCT_NORMALIZED)
        # supported_viz.add(SupportedViz.LATENT_SPACE)
        # supported_viz.add(SupportedViz.LATENT_SPACE_NORMALIZED)
        # supported_viz.add(SupportedViz.LATENT_SPACE_PCA)
        return supported_viz

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Method for the forward pass.

        'training_step', 'validation_step' and 'test_step' should call
        this method in order to compute the output predictions and the loss.

        Returns:
            output_dict: forward output containing the predictions (output logits ecc...) and the loss if any.
        """
        encoding = self.encode(x)
        return self.decode(**encoding)

    def encode(self, *args, **kwargs):
        return self.autoencoder.encode(*args, **kwargs)

    # @property
    # def encode_output(self) -> Set[str]:
    #     raise NotImplementedError
    #
    # @property
    # def decode_input(self) -> Set[str]:
    #     raise NotImplementedError

    def decode(self, *args, **kwargs):
        return self.autoencoder.decode(*args, **kwargs)

    def step(self, batch, batch_index: int, stage: Stage, return_all: bool = False) -> Mapping[str, Any]:
        image_batch = batch["x"]
        out = self(image_batch)

        for metric_name, metric in self.reconstruction_quality_metrics.items():
            metric_value = metric(image_batch, out[Output.RECONSTRUCTION])
            self.log(
                f"{stage}/{metric_name}", metric_value, on_step=False, on_epoch=True, batch_size=image_batch.shape[0]
            )

        loss_out = self.loss(model_out=out, batch=batch)

        self.log_dict(
            {f"{loss_name}/{stage}": value.cpu().detach() for loss_name, value in loss_out.items()},
            on_step=stage == Stage.TRAIN_STAGE,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch["x"].shape[0],
        )

        if return_all:
            return {
                Output.LOSS: loss_out["loss"],
                Output.BATCH: {key: detach_tensors(value) for key, value in batch.items()},
                **{key: detach_tensors(value) for key, value in out.items()},
            }
        else:
            return {
                Output.LOSS: loss_out["loss"],
            }

    def on_fit_start(self) -> None:
        on_fit_start_viz(lightning_module=self, fixed_images=self.fixed_images, anchors_images=self.anchors_images)

    def on_fit_end(self) -> None:
        on_fit_end_viz(lightning_module=self, validation_stats_df=None)

    def training_step(self, batch: Any, batch_idx: int) -> Mapping[str, Any]:
        return self.step(batch, batch_idx, stage=Stage.TRAIN_STAGE, return_all=False)

    def validation_step(self, batch: Any, batch_idx: int) -> Mapping[str, Any]:
        out = self.step(batch, batch_idx, stage=Stage.VAL_STAGE, return_all=True)
        self.validation_step_outputs.append(out)
        return out

    def on_validation_epoch_end(self) -> None:
        outputs = self.validation_step_outputs
        if self.trainer.sanity_checking:
            return

        validation_aggregation = {}
        for output in outputs:
            aggregate(
                validation_aggregation,
                image_index=output["batch"]["id"].tolist(),
                class_name=[self.metadata.idx_to_class[x] for x in output["batch"]["y"]],
                target=output["batch"]["y"],
                latents=output[Output.DEFAULT_LATENT],
                epoch=[self.current_epoch] * len(output["batch"]["id"]),
                is_anchor=[False] * len(output["batch"]["id"]),
                anchor_index=[None] * len(output["batch"]["id"]),
            )

        if self.anchors_images is not None:
            anchors_num = self.anchors_images.shape[0]
            anchors_out = self(self.anchors_images)
            if Output.ANCHORS_LATENT in anchors_out:
                anchors_latents = anchors_out[Output.ANCHORS_LATENT]
            else:
                anchors_latents = anchors_out[Output.DEFAULT_LATENT]
            anchors_reconstructed = anchors_out.get(Output.RECONSTRUCTION, None)

        else:
            pass
            # TODO: Decommenta
            # assert self.anchors_latents is not None
            # anchors_num = self.anchors_latents.shape[0]
            # # TODO: these latents should be normalized if the normalization is enabled..
            # anchors_latents = self.anchors_latents
            # if isinstance(self.autoencoder.decoder, RaeDecoder):
            #     anchors_reconstructed, _ = self.autoencoder.decoder(
            #         self.anchors_latents, anchors_latents=self.anchors_latents
            #     )
            # else:
            #     anchors_reconstructed = self.autoencoder.decoder(self.anchors_latents)

        non_elements = ["none"] * anchors_num
        aggregate(
            validation_aggregation,
            image_index=self.metadata.anchors_idxs
            if self.metadata.anchors_idxs is not None
            else list(range(anchors_num)),
            class_name=[self.metadata.idx_to_class[x] for x in self.metadata.anchors_targets]
            if self.metadata.anchors_targets is not None
            else non_elements,
            target=self.metadata.anchors_targets.cpu() if self.metadata.anchors_targets is not None else non_elements,
            latents=anchors_latents.cpu(),
            epoch=[self.current_epoch] * anchors_num,
            is_anchor=[True] * anchors_num,
            anchor_index=list(range(anchors_num)),
        )

        if (
            SupportedViz.LATENT_SPACE in self.supported_viz
            or SupportedViz.LATENT_SPACE_NORMALIZED in self.supported_viz
            or SupportedViz.LATENT_SPACE_PCA in self.supported_viz
        ):
            latents = validation_aggregation["latents"]
            self.fit_pca(latents)
            add_2D_latents(validation_aggregation, latents=latents, pca=self.validation_pca)
        del validation_aggregation["latents"]

        validation_epoch_end_viz(
            lightning_module=self,
            outputs=outputs,
            validation_stats_df=pd.DataFrame(validation_aggregation),
            anchors_reconstructed=anchors_reconstructed,
            anchors_latents=anchors_latents,
            fixed_images_out=self(self.fixed_images),
        )
        self.validation_step_outputs.clear()  # free memory


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default", version_base="1.1")
def main(cfg: omegaconf.DictConfig) -> None:
    """Debug main to quickly develop the Lightning Module.

    Args:
        cfg: the hydra configuration
    """
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(cfg.nn.data, _recursive_=False)

    _: pl.LightningModule = hydra.utils.instantiate(
        cfg.nn.module,
        metadata=datamodule.metadata,
        _recursive_=False,
    )


if __name__ == "__main__":
    main()
