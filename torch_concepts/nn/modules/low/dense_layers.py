"""Simple fully-connected neural network layers.

This module provides Dense, MLP, and ResidualMLP layers adapted from the 
torch-spatiotemporal library. These layers serve as building blocks for 
neural network architectures in concept-based models.

Reference: https://torch-spatiotemporal.readthedocs.io/en/latest/
"""

import torch
from torch import nn
import torch.nn.functional as F

_torch_activations_dict = {
    'elu': 'ELU',
    'leaky_relu': 'LeakyReLU',
    'prelu': 'PReLU',
    'relu': 'ReLU',
    'rrelu': 'RReLU',
    'selu': 'SELU',
    'celu': 'CELU',
    'gelu': 'GELU',
    'glu': 'GLU',
    'mish': 'Mish',
    'sigmoid': 'Sigmoid',
    'softplus': 'Softplus',
    'tanh': 'Tanh',
    'silu': 'SiLU',
    'swish': 'SiLU',
    'linear': 'Identity'
}

def get_layer_activation(activation):
    """Get PyTorch activation layer class from string name.
    
    Args:
        activation (str or None): Activation function name (case-insensitive).
            Supported: 'elu', 'leaky_relu', 'prelu', 'relu', 'rrelu', 'selu',
            'celu', 'gelu', 'glu', 'mish', 'sigmoid', 'softplus', 'tanh', 
            'silu', 'swish', 'linear'. None returns Identity.
    
    Returns:
        torch.nn.Module: Activation layer class (uninstantiated).
        
    Raises:
        ValueError: If activation name is not recognized.
        
    Example:
        >>> from torch_concepts.nn.modules.low.dense_layers import get_layer_activation
        >>> act_class = get_layer_activation('relu')
        >>> activation = act_class()  # ReLU()
        >>> act_class = get_layer_activation(None)
        >>> activation = act_class()  # Identity()
    """
    if activation is None:
        return nn.Identity
    activation = activation.lower()
    if activation in _torch_activations_dict:
        return getattr(nn, _torch_activations_dict[activation])
    raise ValueError(f"Activation '{activation}' not valid.")



class Dense(nn.Module):
    r"""A simple fully-connected layer implementing

    .. math::

        \mathbf{x}^{\prime} = \sigma\left(\boldsymbol{\Theta}\mathbf{x} +
        \mathbf{b}\right)

    where :math:`\mathbf{x} \in \mathbb{R}^{d_{in}}, \mathbf{x}^{\prime} \in
    \mathbb{R}^{d_{out}}` are the input and output features, respectively,
    :math:`\boldsymbol{\Theta} \in \mathbb{R}^{d_{out} \times d_{in}} \mathbf{b}
    \in \mathbb{R}^{d_{out}}` are trainable parameters, and :math:`\sigma` is
    an activation function.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        activation (str, optional): Activation function to be used.
            (default: :obj:`'relu'`)
        dropout (float, optional): The dropout rate.
            (default: :obj:`0`)
        bias (bool, optional): If :obj:`True`, then the bias vector is used.
            (default: :obj:`True`)
    """

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 activation: str = 'relu',
                 dropout: float = 0.,
                 bias: bool = True):
        super(Dense, self).__init__()
        self.affinity = nn.Linear(in_features, out_features, bias=bias)
        self.activation = get_layer_activation(activation)()
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

    def reset_parameters(self) -> None:
        """Reset layer parameters to initial random values."""
        self.affinity.reset_parameters()

    def forward(self, x):
        """Apply linear transformation, activation, and dropout.
        
        Args:
            x (torch.Tensor): Input tensor of shape (..., in_features).
            
        Returns:
            torch.Tensor: Output tensor of shape (..., out_features).
        """
        out = self.activation(self.affinity(x))
        return self.dropout(out)



class MLP(nn.Module):
    """Simple Multi-layer Perceptron encoder with optional linear readout.

    Args:
        input_size (int): Input size.
        hidden_size (int): Units in the hidden layers.
        output_size (int, optional): Size of the optional readout.
        n_layers (int, optional): Number of hidden layers. (default: 1)
        activation (str, optional): Activation function. (default: `relu`)
        dropout (float, optional): Dropout probability.
    """

    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size=None,
                 n_layers=1,
                 activation='relu',
                 dropout=0.):
        super(MLP, self).__init__()
        
        layers = [
            Dense(in_features=input_size if i == 0 else hidden_size,
                  out_features=hidden_size,
                  activation=activation,
                  dropout=dropout) for i in range(n_layers)
        ]
        self.mlp = nn.Sequential(*layers)

        if output_size is not None:
            self.readout = nn.Linear(hidden_size, output_size)
        else:
            self.register_parameter('readout', None)

    def reset_parameters(self) -> None:
        """Reset all layer parameters to initial random values."""
        for module in self.mlp._modules.values():
            module.reset_parameters()
        if self.readout is not None:
            self.readout.reset_parameters()

    def forward(self, x):
        """Forward pass through MLP layers with optional readout.
        
        Args:
            x (torch.Tensor): Input tensor of shape (..., input_size).
            
        Returns:
            torch.Tensor: Output tensor of shape (..., output_size) 
                if readout is defined, else (..., hidden_size).
        """
        out = self.mlp(x)
        if self.readout is not None:
            return self.readout(out)
        return out



class ResidualMLP(nn.Module):
    """Multi-layer Perceptron with residual connections.

    Args:
        input_size (int): Input size.
        hidden_size (int): Units in the hidden layers.
        output_size (int, optional): Size of the optional readout.
        n_layers (int, optional): Number of hidden layers. (default: 1)
        activation (str, optional): Activation function. (default: `relu`)
        dropout (float, optional): Dropout probability. (default: 0.)
        parametrized_skip (bool, optional): Whether to use parametrized skip
            connections for the residuals.
    """

    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size=None,
                 n_layers=1,
                 activation='relu',
                 dropout=0.,
                 parametrized_skip=False):
        super(ResidualMLP, self).__init__()

        self.layers = nn.ModuleList([
            nn.Sequential(
                Dense(in_features=input_size if i == 0 else hidden_size,
                      out_features=hidden_size,
                      activation=activation,
                      dropout=dropout), nn.Linear(hidden_size, hidden_size))
            for i in range(n_layers)
        ])

        self.skip_connections = nn.ModuleList()
        for i in range(n_layers):
            if i == 0 and input_size != output_size:
                self.skip_connections.append(nn.Linear(input_size,
                                                       hidden_size))
            elif parametrized_skip:
                self.skip_connections.append(
                    nn.Linear(hidden_size, hidden_size))
            else:
                self.skip_connections.append(nn.Identity())

        if output_size is not None:
            self.readout = nn.Linear(hidden_size, output_size)
        else:
            self.register_parameter('readout', None)

    def forward(self, x):
        """Forward pass with residual connections.
        
        Args:
            x (torch.Tensor): Input tensor of shape (..., input_size).
            
        Returns:
            torch.Tensor: Output tensor of shape (..., output_size) 
                if readout is defined, else (..., hidden_size).
                
        Note:
            Each layer applies: x = layer(x) + skip(x), where skip is either
            Identity, a projection layer, or a parametrized transformation.
        """
        for layer, skip in zip(self.layers, self.skip_connections):
            x = layer(x) + skip(x)
        if self.readout is not None:
            return self.readout(x)
        return x



class LinearEmbeddingEncoder(torch.nn.Module):
    """
    Linear encoder that transforms embeddings into a set of embeddings.

    Applies a single linear projection from ``in_features`` to
    ``n_embeddings * out_features``, then unflattens the last dimension to
    ``(n_embeddings, out_features)``.

    Attributes:
        out_shape (Tuple[int, int]): Target shape used by ``nn.Unflatten``.
        encoder (nn.Sequential): ``Linear -> Unflatten`` encoder.

    Args:
        in_features (int): Number of input features.
        out_features (int): Feature dimension of each output embedding.
        n_embeddings (int, optional): Number of output embeddings.
            Defaults to ``1``.

    Example:
        >>> import torch
        >>> from torch_concepts.nn import LinearEmbeddingEncoder
        >>>
        >>> encoder = LinearEmbeddingEncoder(
        ...     in_features=128,
        ...     out_features=16,
        ...     n_embeddings=5,
        ... )
        >>> embeddings = torch.randn(4, 128)
        >>> out = encoder(embeddings)
        >>> out.shape
        torch.Size([4, 5, 16])

    References:
        Espinosa Zarlenga et al. "Concept Embedding Models: Beyond the
        Accuracy-Explainability Trade-Off", NeurIPS 2022.
        https://arxiv.org/abs/2209.09056
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_embeddings: int = 1,
    ):
        """
        Initialize the linear embedding encoder.

        Args:
            in_features: Number of input features.
            out_features: Dimension of each output embedding.
            n_embeddings: Number of output embeddings.
        """
        super().__init__()

        self.out_shape = (n_embeddings, out_features)

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(
                in_features,
                n_embeddings * out_features
            ),
            torch.nn.Unflatten(-1, self.out_shape)
        )

    def forward(self, x: torch.Tensor):
        """
        Encode into a set of embeddings.

        Args:
            x: Input tensor of shape ``(..., in_features)``.

        Returns:
            torch.Tensor: Embeddings of shape ``(..., n_embeddings, out_features)``.
        """
        return self.encoder(x)



class SelectorEmbeddingEncoder(torch.nn.Module):
    """
    Memory-based selector for embeddings with attention mechanism.

    This module maintains a learnable memory bank of embeddings and uses an
    attention mechanism to select relevant embeddings based on input. It
    supports both soft (weighted) and hard (Gumbel-softmax) selection.

    Attributes:
        temperature (float): Temperature for softmax/Gumbel-softmax.
        memory_size (int): Number of memory slots.
        out_features (int): Feature dimension of each output embedding.
        memory (nn.Embedding): Learnable memory bank.
        selector (nn.Sequential): Attention network for memory selection.

    Args:
        in_features: Number of input features.
        out_features: Feature dimension of each output embedding.
        n_embeddings: Number of output embeddings. Defaults to ``1``.
        memory_size: Number of memory slots. Defaults to ``2``.
        temperature: Temperature parameter for selection. Defaults to ``1.0``.
        *args: Additional arguments for the linear layer.
        **kwargs: Additional keyword arguments for the linear layer.

    Example:
        >>> import torch
        >>> from torch_concepts.nn import SelectorEmbeddingEncoder
        >>>
        >>> # Create memory selector
        >>> selector = SelectorEmbeddingEncoder(
        ...     in_features=64,
        ...     out_features=32,
        ...     n_embeddings=5,
        ...     memory_size=10,
        ...     temperature=0.5
        ... )
        >>>
        >>> # Forward pass with soft selection
        >>> embeddings = torch.randn(4, 64)  # batch_size=4
        >>> selected = selector(embeddings, sampling=False)
        >>> print(selected.shape)
        torch.Size([4, 5, 32])
        >>>
        >>> # Forward pass with hard selection (Gumbel-softmax)
        >>> selected_hard = selector(embeddings, sampling=True)
        >>> print(selected_hard.shape)
        torch.Size([4, 5, 32])
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_embeddings: int = 1,
        memory_size: int = 2,
        temperature: float = 1.0,
        *args,
        **kwargs,
    ):
        """
        Initialize the memory selector.

        Args:
            in_features: Number of input features.
            out_features: Feature dimension of each output embedding.
            n_embeddings: Number of output embeddings. Defaults to ``1``.
            memory_size: Number of memory slots. Defaults to ``2``.
            temperature: Temperature for selection. Defaults to ``1.0``.
            *args: Additional arguments for the linear layer.
            **kwargs: Additional keyword arguments for the linear layer.
        """
        super().__init__()
        self.temperature = temperature
        self.memory_size = memory_size
        self.out_features = out_features
        self._selector_out_shape = (n_embeddings, memory_size)
        self._selector_out_dim = torch.tensor(self._selector_out_shape).prod().item()

        # init memory of embeddings [n_embeddings, memory_size * out_features]
        self.memory = torch.nn.Embedding(n_embeddings, memory_size * out_features)

        # init selector [B, n_embeddings]
        self.selector = torch.nn.Sequential(
            torch.nn.Linear(in_features, out_features),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(
                out_features,
                self._selector_out_dim,
                *args,
                **kwargs,
            ),
            torch.nn.Unflatten(-1, self._selector_out_shape),
        )

    def forward(
        self,
        x: torch.Tensor,
        sampling: bool = False,
    ) -> torch.Tensor:
        """
        Select memory embeddings based on input.

        Computes attention weights over memory slots and returns a weighted
        combination of memory embeddings. Can use soft attention or hard
        selection via Gumbel-softmax.

        Args:
            x: Input tensor of shape ``(..., in_features)``.
            sampling: If True, use Gumbel-softmax for hard selection;
                     if False, use soft attention (default: False).

        Returns:
            torch.Tensor: Selected embeddings of shape
                         ``(..., n_embeddings, out_features)``.
        """
        memory = self.memory.weight.view(-1, self.memory_size, self.out_features)
        mixing_coeff = self.selector(x)
        if sampling:
            mixing_probs = F.gumbel_softmax(mixing_coeff, dim=1, tau=self.temperature, hard=True)
        else:
            mixing_probs = torch.softmax(mixing_coeff / self.temperature, dim=1)

        embeddings = torch.einsum("btm,tme->bte", mixing_probs, memory) # [Batch x Task x Memory] x [Task x Memory x Emb] -> [Batch x Task x Emb]
        return embeddings
