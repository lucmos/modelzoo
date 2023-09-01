import math
from typing import Dict, List, Optional

import hydra.utils
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from modelzoo.modules.aes.blocks import build_dynamic_encoder_decoder
from modelzoo.modules.aes.enumerations import Output


class VanillaAE(nn.Module):
    def __init__(
        self,
        metadata,
        input_size,
        latent_dim: int,
        decoder_in_normalization: nn.Module,
        hidden_dims: List = None,
        latent_activation: Optional[str] = None,
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
        self.latent_dim = latent_dim

        self.encoder, self.encoder_out_shape, self.decoder = build_dynamic_encoder_decoder(
            width=metadata.width, height=metadata.height, n_channels=metadata.n_channels, hidden_dims=hidden_dims
        )
        encoder_out_numel = math.prod(self.encoder_out_shape[1:])

        self.encoder_out = nn.Sequential(
            nn.Linear(encoder_out_numel, latent_dim),
            hydra.utils.instantiate({"_target_": latent_activation})
            if latent_activation is not None
            else nn.Identity(),
        )

        self.decoder_in = nn.Sequential(
            nn.Linear(
                self.latent_dim,
                encoder_out_numel,
            ),
            hydra.utils.instantiate({"_target_": latent_activation})
            if latent_activation is not None
            else nn.Identity(),
        )
        self.decoder_in_normalization = decoder_in_normalization

    def encode(self, input: Tensor) -> Dict[str, Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        result = self.encoder_out(result)
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
        result = self.decoder_in_normalization(batch_latent)
        result = self.decoder_in(result)
        result = result.view(-1, *self.encoder_out_shape[1:])
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
