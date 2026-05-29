"""Simple fully-connected neural network layers.

This module provides Dense, MLP, and ResidualMLP layers adapted from the 
torch-spatiotemporal library. These layers serve as building blocks for 
neural network architectures in concept-based models.

Reference: https://torch-spatiotemporal.readthedocs.io/en/latest/
"""

from torch import nn


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
        input_size (int): Number of input features.
        output_size (int): Number of output features.
        activation (str, optional): Activation function to be used.
            (default: :obj:`'relu'`)
        dropout (float, optional): The dropout rate.
            (default: :obj:`0`)
        bias (bool, optional): If :obj:`True`, then the bias vector is used.
            (default: :obj:`True`)
    """

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 activation: str = 'relu',
                 dropout: float = 0.,
                 bias: bool = True):
        super(Dense, self).__init__()
        self.affinity = nn.Linear(input_size, output_size, bias=bias)
        self.activation = get_layer_activation(activation)()
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

    def reset_parameters(self) -> None:
        """Reset layer parameters to initial random values."""
        self.affinity.reset_parameters()

    def forward(self, x):
        """Apply linear transformation, activation, and dropout.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size).
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
                 hidden_size=64,
                 output_size=None,
                 n_layers=1,
                 activation='relu',
                 dropout=0.):
        super(MLP, self).__init__()
        
        layers = [
            Dense(input_size=input_size if i == 0 else hidden_size,
                  output_size=hidden_size,
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
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size) 
                if readout is defined, else (batch_size, hidden_size).
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
                Dense(input_size=input_size if i == 0 else hidden_size,
                      output_size=hidden_size,
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
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size) 
                if readout is defined, else (batch_size, hidden_size).
                
        Note:
            Each layer applies: x = layer(x) + skip(x), where skip is either
            Identity, a projection layer, or a parametrized transformation.
        """
        for layer, skip in zip(self.layers, self.skip_connections):
            x = layer(x) + skip(x)
        if self.readout is not None:
            return self.readout(x)
        return x
