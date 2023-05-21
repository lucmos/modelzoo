import torch
from torch import Tensor, nn
from torchvision import transforms


class ChannelAdapt(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels: int = in_channels
        self.out_channels: int = out_channels

        assert self.in_channels in {1, 3, self.out_channels}

        self.transform: nn.Module = (
            nn.Identity()
            if self.in_channels == self.out_channels
            else transforms.Grayscale(num_output_channels=self.out_channels)
        )

    def forward(self, tensor: Tensor) -> Tensor:
        """
        Args:
            tensor (Tensor): Tensor/PIL image to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        return self.transform(tensor)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(in_channels={self.in_channels}, "
            f"out_channels={self.out_channels}, transform={self.transform})"
        )


class ChannelOrder(torch.nn.Module):
    def forward(self, tensor: Tensor) -> Tensor:
        """
        Args:
            tensor (Tensor): Tensor/PIL image to be normalized in [C, H, W].

        Returns:
            Tensor: Normalized Tensor image.
        """
        if tensor.shape[-1] != 1 or tensor.shape[-1] != 3:
            tensor = tensor.unsqueeze(-1)

        if tensor.ndim == 3:
            return tensor.permute(-1, -2, -3)
        elif tensor.ndim == 4:
            return tensor.permute(0, -1, -3, -2)
        else:
            raise RuntimeError(f"Unsupported tensor shape {tensor.shape}")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class Normalize01(torch.nn.Module):
    def forward(self, tensor: Tensor) -> Tensor:
        """
        Args:
            tensor (Tensor): Tensor/PIL image to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        return tensor.type(torch.float32) / 255.0

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
