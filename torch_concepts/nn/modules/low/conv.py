"""Convolutional layers for image-valued variables.

A PGM keeps every variable on a single flat event axis, so an image variable of
shape ``(C, H, W)`` is ``C * H * W`` scalars to the graph. A dense head to that
many outputs is ruinous — CelebA at native resolution is 116,412 pixels, so the
last layer alone would carry ~60M parameters. :class:`ConvDecoder` spends those
parameters on a stack of transposed convolutions instead, and flattens only at
the very end so the CPD contract is unchanged.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn


class ConvDecoder(nn.Module):
    """Decode a flat feature vector into a flat image via transposed convolutions.

    Projects ``in_features`` to a small ``base_size x base_size`` spatial grid,
    then doubles the resolution once per entry of ``hidden_channels`` before a
    final convolution to the requested channel count. Input and output are both
    **flat**: the module is a drop-in parametrization for an image variable, and
    the parameter's domain activation is applied outside it, as
    :class:`~torch_concepts.nn.modules.mid.factors.cpd.ParametricCPD` requires
    (compose with :class:`~torch_concepts.nn.DefaultActivation`).

    Parameters
    ----------
    in_features : int
        Width of the incoming feature vector (a concept bottleneck, a latent).
    out_shape : tuple of int
        Image event shape ``(channels, height, width)``. ``height`` and
        ``width`` must be ``base_size * 2 ** len(hidden_channels)``.
    hidden_channels : sequence of int, default ``(128, 64, 32)``
        Channel count after each upsampling stage; its length is the number of
        stages, so each entry doubles the resolution.
    base_size : int, optional
        Side of the spatial grid the linear projection produces. Derived from
        ``out_shape`` when omitted — ``side / 2 ** len(hidden_channels)``, which
        is the only value that reaches the target — so a config only has to
        choose the depth. Defaults to ``None`` (derive).
    activation : type, default ``nn.LeakyReLU``
        Activation class used between stages.
    batch_norm : bool, default True
        Whether to insert ``BatchNorm2d`` after each transposed convolution.

    Raises
    ------
    ValueError
        If ``out_shape`` is not 3-dimensional, or its spatial size is not
        ``base_size * 2 ** len(hidden_channels)``.

    Examples
    --------
    >>> import torch
    >>> from torch_concepts.nn import ConvDecoder
    >>> decoder = ConvDecoder(in_features=48, out_shape=(3, 32, 32),
    ...                       hidden_channels=(64, 32, 16))
    >>> decoder(torch.randn(8, 48)).shape
    torch.Size([8, 3072])

    ``base_size`` follows from the depth, so MNIST works with the same config as
    long as the depth divides the side (``28 = 7 * 2 ** 2``):

    >>> ConvDecoder(in_features=48, out_shape=(3, 28, 28),
    ...             hidden_channels=(64, 32))(torch.randn(2, 48)).shape
    torch.Size([2, 2352])

    A depth the target cannot reach is rejected up front rather than silently
    resized:

    >>> ConvDecoder(in_features=48, out_shape=(3, 28, 28),
    ...             hidden_channels=(64, 32, 16))
    Traceback (most recent call last):
        ...
    ValueError: ConvDecoder: a 28x28 target is not reachable with 3 upsampling ...
    """

    def __init__(
        self,
        in_features: int,
        out_shape: Union[Tuple[int, ...], torch.Size],
        hidden_channels: Sequence[int] = (128, 64, 32),
        base_size: Optional[int] = None,
        activation: type = nn.LeakyReLU,
        batch_norm: bool = True,
    ) -> None:
        super().__init__()
        out_shape = tuple(int(s) for s in out_shape)
        if len(out_shape) != 3:
            raise ValueError(
                f"ConvDecoder: out_shape must be (channels, height, width), "
                f"got {out_shape}."
            )
        channels, height, width = out_shape
        hidden_channels = tuple(int(c) for c in hidden_channels)
        n_stages = len(hidden_channels)
        if height != width:
            raise ValueError(
                f"ConvDecoder: out_shape {out_shape} is not square "
                f"({height}x{width}). Resize the images to a square first."
            )
        if base_size is None:
            # Each stage doubles the resolution, so the starting grid is fixed
            # once the depth is chosen. Deriving it lets a config pick only the
            # depth and stay correct across datasets of different resolutions.
            if height % (2 ** n_stages):
                raise ValueError(
                    f"ConvDecoder: a {height}x{height} target is not reachable "
                    f"with {n_stages} upsampling stages ({height} is not "
                    f"divisible by {2 ** n_stages}). Use a different number of "
                    "hidden_channels, or set base_size explicitly."
                )
            base_size = height // (2 ** n_stages)
        reached = base_size * 2 ** n_stages
        if (height, width) != (reached, reached):
            raise ValueError(
                f"ConvDecoder: out_shape {out_shape} has spatial size "
                f"{height}x{width}, but base_size={base_size} and "
                f"{n_stages} upsampling stages reach "
                f"{reached}x{reached}. Choose a base_size and hidden_channels "
                "whose product matches, or resize the images."
            )

        self.in_features = in_features
        self.out_shape = out_shape
        self.base_size = int(base_size)
        self.out_features = channels * height * width

        first = hidden_channels[0]
        self.project = nn.Linear(in_features, first * base_size * base_size)
        self.unflatten = nn.Unflatten(-1, (first, base_size, base_size))

        widths = list(hidden_channels) + [channels]
        stages: list[nn.Module] = []
        for i, (in_c, out_c) in enumerate(zip(widths, widths[1:])):
            stages.append(
                nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2, padding=1)
            )
            # The last stage stays bare: it emits the raw parameter, which the
            # DefaultActivation composed downstream maps into its domain.
            if i < len(widths) - 2:
                if batch_norm:
                    stages.append(nn.BatchNorm2d(out_c))
                stages.append(activation())
        self.stages = nn.Sequential(*stages)
        self.flatten = nn.Flatten(start_dim=-3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.unflatten(self.project(x))
        # Conv layers take exactly one batch axis, so fold any extra leading
        # dimensions away and restore them afterwards.
        leading = h.shape[:-3]
        h = h.reshape(-1, *h.shape[-3:])
        h = self.stages(h)
        h = h.reshape(*leading, *h.shape[-3:])
        return self.flatten(h)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_shape={self.out_shape}, "
            f"base_size={self.base_size}"
        )
