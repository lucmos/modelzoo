from typing import Any, Dict, List

import torch
from anypy.nn.blocks import build_dynamic_encoder_decoder
from torch import Tensor, nn
from torch.nn import functional as F

from modelzoo.modules.aes.enumerations import Output


class VanillaAE(nn.Module):
    def __init__(
        self,
        metadata,
        input_size,
        encoder_layers_config: List[Dict[str, Any]],
        decoder_layers_config: List[Dict[str, Any]],
        **kwargs,
    ) -> None:
        """https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py

        Args:
            in_channels:
            latent_dim:
            hidden_dims:
            **kwargs:
        """
        super().__init__()

        self.metadata = metadata
        self.input_size = input_size

        self.encoder, self.encoder_out_shape, self.decoder = build_dynamic_encoder_decoder(
            encoder_layers_config=encoder_layers_config,
            decoder_layers_config=decoder_layers_config,
            input_shape=[-1, metadata.n_channels, metadata.height, metadata.width],
        )

    def encode(self, input: Tensor) -> Dict[str, Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        return {
            Output.BATCH_LATENT: result,
        }

    def decode(self, batch_latent: Tensor) -> Dict[Output, Tensor]:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = batch_latent.view(-1, *self.encoder_out_shape[1:])
        result = self.decoder(result)
        return {
            Output.RECONSTRUCTION: result,
            Output.DEFAULT_LATENT: batch_latent,
            Output.BATCH_LATENT: batch_latent,
        }

    def loss_function(self, model_out, batch, *args, **kwargs) -> dict:
        predictions = model_out[Output.RECONSTRUCTION]
        targets = batch["x"]
        loss = F.mse_loss(predictions, targets, reduction="mean")
        return {
            "loss": loss,
            "reconstruction": F.mse_loss(predictions, targets, reduction="mean"),
        }

    def sample(self, num_samples: int, current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples, self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples
