from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Sequence, Union

import plotly
import torch
import torch.nn.functional as F
import wandb
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

from modelzoo.modules.aes.enumerations import Output, SupportedViz
from modelzoo.utils.plotting import plot_images, plot_latent_evolution, plot_latent_space, plot_matrix, plot_violin

if TYPE_CHECKING:
    from modelzoo.pl_modules.pl_abstractmodule import AbstractLightningModule


def check_all_equal_size(elements: Iterable[Any]) -> bool:
    """Check if all elements have the same size.

    Args:
        elements: the elements to check

    Returns:
        True if all the elements have the same size, False otherwise
    """
    first_element_size = len(next(iter(elements)))
    return all(len(x) == first_element_size for x in elements)


def aggregate(
    aggregation: Dict[str, Union[torch.Tensor, Sequence[Any], Any]],
    dim: int = 0,
    device: str = "cpu",
    **kwargs: Union[torch.Tensor, Sequence[Union[str, int, bool, Any]]],
) -> Dict[str, Union[torch.Tensor, Sequence[Any]]]:
    """Extend the elements in the aggregation dictionary with the kwargs.

    Args:
        aggregation: the aggregation dictionary, can contain tensors or sequences
        dim: the dimension over which to aggregate
        device: the device in which we should perform the aggregation
        **kwargs: named-arguments with matching keys in the aggregation dictionary
                  to extend the corresponding values
    Returns:
        the updated aggregation dictionary
    """
    assert check_all_equal_size(kwargs.values())

    for key, value in kwargs.items():
        if isinstance(value, torch.Tensor):
            value = value.to(device)
            if key not in aggregation:
                aggregation[key] = torch.empty(0)
            aggregation[key] = torch.cat((aggregation[key], value), dim=dim)
        elif isinstance(value, Sequence):
            if key not in aggregation:
                aggregation[key] = []
            aggregation[key].extend(value)

    assert check_all_equal_size(aggregation.values())
    return aggregation


def add_2D_latents(
    aggregation: Dict[str, Union[torch.Tensor, Sequence[Any], Any]],
    latents: torch.Tensor,
    pca: PCA,
) -> Dict[str, Union[torch.Tensor, Sequence[Any], Any]]:
    latents_normalized = F.normalize(latents, p=2, dim=-1)
    latents_pca = pca.transform(latents)

    aggregation["latent_0"] = latents[:, 0]
    aggregation["latent_1"] = latents[:, 1]
    aggregation["latent_0_normalized"] = latents_normalized[:, 0]
    aggregation["latent_1_normalized"] = latents_normalized[:, 1]
    aggregation["latent_0_pca"] = latents_pca[:, 0]
    aggregation["latent_1_pca"] = latents_pca[:, 1]

    return aggregation


def on_fit_start_viz(
    lightning_module: AbstractLightningModule,
    fixed_images: Optional[torch.Tensor] = None,
    anchors_images: Optional[torch.Tensor] = None,
):
    to_log = {}
    to_close = set()
    if SupportedViz.VALIDATION_IMAGES_SOURCE in lightning_module.supported_viz:
        to_log["images/source"] = plot_images(fixed_images, "Source images")
        to_close.add(to_log["images/source"])

    if SupportedViz.ANCHORS_SOURCE in lightning_module.supported_viz:
        to_log["anchors/source"] = plot_images(anchors_images, "Anchors images")
        to_close.add(to_log["anchors/source"])

    if to_log:
        lightning_module.logger.experiment.log(to_log, step=lightning_module.global_step)

    for fig in to_close:
        plt.close(fig)


def on_fit_end_viz(
    lightning_module: AbstractLightningModule,
    validation_stats_df,
):
    if SupportedViz.LATENT_EVOLUTION_PLOTLY_ANIMATION in lightning_module.supported_viz:
        latent_plot = plot_latent_evolution(
            metadata=lightning_module.metadata,
            validation_stats_df=validation_stats_df,
            n_samples=lightning_module.hparams.plot_n_val_samples,
        )
        # Convert to HTML as a workaround to https://github.com/wandb/client/issues/2191
        lightning_module.logger.experiment.log({"latent": wandb.Html(plotly.io.to_html(latent_plot), inject=True)})


def validation_epoch_end_viz(
    lightning_module: AbstractLightningModule,
    outputs: List[Dict[str, Any]],
    validation_stats_df,
    anchors_reconstructed: Optional[torch.Tensor] = None,
    anchors_latents: Optional[torch.Tensor] = None,
    fixed_images_out: Optional[torch.Tensor] = None,
):
    to_log = {}
    to_close = set()
    if SupportedViz.ANCHORS_RECONSTRUCTED in lightning_module.supported_viz:
        to_log["anchors/reconstructed"] = plot_images(anchors_reconstructed, "Anchors reconstructed")
        to_close.add(to_log["anchors/reconstructed"])

    if SupportedViz.VALIDATION_IMAGES_RECONSTRUCTED in lightning_module.supported_viz:
        to_log["images/reconstructed"] = plot_images(fixed_images_out[Output.RECONSTRUCTION], "Reconstructed images")
        to_close.add(to_log["images/reconstructed"])

    if SupportedViz.ANCHORS_SELF_INNER_PRODUCT in lightning_module.supported_viz:
        anchors_self_inner_product = anchors_latents @ anchors_latents.T
        to_log["anchors-vs-anchors/inner"] = plot_matrix(
            anchors_self_inner_product,
            title="Anchors vs Anchors inner products",
            labels={"x": "anchors", "y": "anchors"},
        )

    if SupportedViz.ANCHORS_VALIDATION_IMAGES_INNER_PRODUCT in lightning_module.supported_viz:
        batch_latent = fixed_images_out[Output.BATCH_LATENT]
        anchors_batch_latents_inner_product = anchors_latents @ batch_latent.T
        to_log["anchors-vs-samples/inner"] = plot_matrix(
            anchors_batch_latents_inner_product,
            title="Anchors vs Samples images inner products",
            labels={"x": "samples", "y": "anchors"},
        )

    if SupportedViz.ANCHORS_SELF_INNER_PRODUCT_NORMALIZED in lightning_module.supported_viz:
        anchors_latents_normalized = F.normalize(anchors_latents, p=2, dim=-1)
        anchors_self_inner_product_normalized = anchors_latents_normalized @ anchors_latents_normalized.T
        to_log["anchors-vs-anchors/inner-normalized"] = plot_matrix(
            anchors_self_inner_product_normalized,
            title="Anchors vs Anchors inner products",
            labels={"x": "anchors", "y": "anchors"},
        )

    if SupportedViz.ANCHORS_VALIDATION_IMAGES_INNER_PRODUCT_NORMALIZED in lightning_module.supported_viz:
        batch_latent = fixed_images_out[Output.BATCH_LATENT]
        anchors_latents_normalized = F.normalize(anchors_latents, p=2, dim=-1)
        batch_latent_normalized = F.normalize(batch_latent, p=2, dim=-1)
        anchors_batch_latents_inner_product_normalized = anchors_latents_normalized @ batch_latent_normalized.T
        to_log["anchors-vs-samples/inner-normalized"] = plot_matrix(
            anchors_batch_latents_inner_product_normalized,
            title="Anchors vs Samples images inner products",
            labels={"x": "samples", "y": "anchors"},
        )

    if SupportedViz.INVARIANT_LATENT_DISTRIBUTION in lightning_module.supported_viz:
        fig = plot_violin(
            torch.cat([output[Output.INV_LATENTS] for output in outputs], dim=0),
            title="Relative Latent Space distribution",
            y_label="validation distribution",
            x_label="anchors",
        )
        to_log["distributions/invariant-latent-space"] = wandb.Image(fig)
        to_close.add(fig)

    if SupportedViz.LATENT_SPACE in lightning_module.supported_viz:
        to_log["latent/space"] = plot_latent_space(
            metadata=lightning_module.metadata,
            validation_stats_df=validation_stats_df,
            epoch=lightning_module.current_epoch,
            x_data="latent_0",
            y_data="latent_1",
            n_samples=lightning_module.hparams.plot_n_val_samples,
        )

    if SupportedViz.LATENT_SPACE_NORMALIZED in lightning_module.supported_viz:
        to_log["latent/space-normalized"] = plot_latent_space(
            metadata=lightning_module.metadata,
            validation_stats_df=validation_stats_df,
            epoch=lightning_module.current_epoch,
            x_data="latent_0_normalized",
            y_data="latent_1_normalized",
            n_samples=lightning_module.hparams.plot_n_val_samples,
        )

    if SupportedViz.LATENT_SPACE_PCA in lightning_module.supported_viz:
        to_log["latent/space-pca"] = plot_latent_space(
            metadata=lightning_module.metadata,
            validation_stats_df=validation_stats_df,
            epoch=lightning_module.current_epoch,
            x_data="latent_0_pca",
            y_data="latent_1_pca",
            n_samples=lightning_module.hparams.plot_n_val_samples,
        )

    if to_log:
        lightning_module.logger.experiment.log(to_log, step=lightning_module.global_step)

    for fig in to_close:
        plt.close(fig)
