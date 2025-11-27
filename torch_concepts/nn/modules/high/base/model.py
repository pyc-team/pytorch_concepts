"""Base model class for concept-based neural networks.

This module defines the abstract BaseModel class that serves as the foundation
for all concept-based models in the library. It handles backbone integration,
encoder setup, and provides hooks for data preprocessing.

BaseModel supports two training modes:

1. **Standard PyTorch Training** (Manual Loop):
   - Initialize model without loss parameter
   - Manually define optimizer, loss function, training loop
   - Full control over forward pass, loss computation, optimization
   - Ideal for custom training procedures

2. **PyTorch Lightning Training** (Automatic):
   - Initialize model with loss, optim_class, optim_kwargs parameters
   - Use Lightning Trainer for automatic training/validation/testing
   - Inherits training logic from Learner classes (JointLearner, IndependentLearner)
   - Ideal for rapid experimentation with standard procedures

See Also
--------
torch_concepts.nn.modules.high.learners.JointLearner : Lightning training logic
torch_concepts.nn.modules.high.models.cbm.ConceptBottleneckModel : Concrete implementation
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Mapping, Dict
import torch
import torch.nn as nn

from .....annotations import Annotations
from ...low.dense_layers import MLP
from .....typing import BackboneType
from .....utils import add_distribution_to_annotations

class BaseModel(nn.Module, ABC):
    """Abstract base class for concept-based models.

    Provides common functionality for models that use backbones for feature extraction, 
    and encoders for latent representations. All concrete model implementations 
    should inherit from this class.

    BaseModel is flexible and supports two distinct training paradigms:

    **Mode 1: Standard PyTorch Training (Manual Loop)**
    
    Initialize model without loss/optimizer parameters for full manual control.
    You define the training loop, optimizer, and loss function externally.
    
    **Mode 2: PyTorch Lightning Training (Automatic)**
    
    Initialize model with loss, optim_class, and optim_kwargs for automatic training
    via PyTorch Lightning Trainer. The model inherits training logic from Learner classes.

    Parameters
    ----------
    input_size : int
        Dimensionality of input features after backbone processing. If no backbone
        is used (backbone=None), this should match raw input dimensionality.
    annotations : Annotations
        Concept annotations containing variable names, cardinalities, and optional
        distribution metadata. Distributions specify how the model represents each
        concept (e.g., Bernoulli for binary, Categorical for multi-class).
    variable_distributions : Mapping, optional
        Dictionary mapping concept names to torch.distributions classes (e.g.,
        ``{'c1': Bernoulli, 'c2': Categorical}``). Required if annotations lack
        'distribution' metadata. If provided, distributions are added to annotations
        internally. Can also be a GroupConfig object. Defaults to None.
    backbone : BackboneType, optional
        Feature extraction module (e.g., ResNet, ViT) applied before latent encoder.
        Can be nn.Module or callable. If None, assumes inputs are pre-computed features.
        Defaults to None.
    latent_encoder : nn.Module, optional
        Custom encoder mapping backbone outputs to latent space. If provided,
        latent_encoder_kwargs are passed to this constructor. If None and
        latent_encoder_kwargs provided, uses MLP. Defaults to None.
    latent_encoder_kwargs : Dict, optional
        Arguments for latent encoder construction. Common keys:
        - 'hidden_size' (int): Latent dimension
        - 'n_layers' (int): Number of hidden layers
        - 'activation' (str): Activation function name
        If None, uses nn.Identity (no encoding). Defaults to None.
    **kwargs
        Additional arguments passed to nn.Module superclass.

    Attributes
    ----------
    concept_annotations : AxisAnnotation
        Axis-1 annotations with distribution metadata for each concept.
    concept_names : List[str]
        List of concept variable names from annotations.
    backbone : BackboneType or None
        Feature extraction module (None if using pre-computed features).
    latent_encoder : nn.Module
        Encoder transforming backbone outputs to latent representations.
    latent_size : int
        Dimensionality of latent encoder output (input to concept encoders).

    Notes
    -----
    - **Concept Distributions**: The model needs to know which distribution to use
      for each concept (Bernoulli, Categorical, Normal, etc.). This can be provided
      in two ways:
      
      1. In annotations metadata: ``metadata={'c1': {'distribution': Bernoulli}}``
      2. Via variable_distributions parameter at initialization
      
      If distributions are in annotations, variable_distributions is not needed.
      If not, variable_distributions is required and will be added to annotations.
    - Subclasses must implement ``forward()``, ``filter_output_for_loss()``,
      and ``filter_output_for_metrics()`` methods.
    - For Lightning training, subclasses typically inherit from both BaseModel
      and a Learner class (e.g., JointLearner) via multiple inheritance.
    - The latent_size attribute is critical for downstream concept encoders
      to determine input dimensionality.

    Examples
    --------
    Distributions specify how the model represents concepts. Provide them either
    in annotations metadata OR via variable_distributions parameter:
    
    >>> import torch
    >>> import torch.nn as nn
    >>> from torch.distributions import Bernoulli
    >>> from torch_concepts.nn import ConceptBottleneckModel
    >>> from torch_concepts.annotations import AxisAnnotation, Annotations
    >>> 
    >>> # Option 1: Distributions in annotations metadata
    >>> ann = Annotations({
    ...     1: AxisAnnotation(
    ...         labels=['c1', 'c2', 'task'],
    ...         cardinalities=[1, 1, 1],
    ...         metadata={
    ...             'c1': {'type': 'binary', 'distribution': Bernoulli},
    ...             'c2': {'type': 'binary', 'distribution': Bernoulli},
    ...             'task': {'type': 'binary', 'distribution': Bernoulli}
    ...         }
    ...     )
    ... })
    >>> model = ConceptBottleneckModel(
    ...     input_size=10,
    ...     annotations=ann,  # Distributions already in metadata
    ...     task_names=['task']
    ... )
    >>> 
    >>> # Option 2: Distributions via variable_distributions parameter
    >>> ann_no_dist = Annotations({
    ...     1: AxisAnnotation(
    ...         labels=['c1', 'c2', 'task'],
    ...         cardinalities=[1, 1, 1]
    ...     )
    ... })
    >>> variable_distributions = {'c1': Bernoulli, 'c2': Bernoulli, 'task': Bernoulli}
    >>> model = ConceptBottleneckModel(
    ...     input_size=10,
    ...     annotations=ann_no_dist,
    ...     variable_distributions=variable_distributions,  # Added here
    ...     task_names=['task']
    ... )
    >>> 
    >>> # Manual training loop
    >>> optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    >>> loss_fn = nn.BCEWithLogitsLoss()
    >>> x = torch.randn(32, 10)
    >>> y = torch.randint(0, 2, (32, 3)).float()
    >>> 
    >>> for epoch in range(100):
    ...     optimizer.zero_grad()
    ...     out = model(x, query=['c1', 'c2', 'task'])
    ...     loss = loss_fn(out, y)
    ...     loss.backward()
    ...     optimizer.step()

    See Also
    --------
    torch_concepts.nn.modules.high.models.cbm.ConceptBottleneckModel : Concrete CBM implementation
    torch_concepts.nn.modules.high.learners.JointLearner : Lightning training logic for joint models
    torch_concepts.annotations.Annotations : Concept annotation container
    """

    def __init__(
        self,
        input_size: int,
        annotations: Annotations,
        variable_distributions: Optional[Mapping] = None,
        backbone: Optional[BackboneType] = None,
        latent_encoder: Optional[nn.Module] = None,
        latent_encoder_kwargs: Optional[Dict] = None,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)

        if annotations is not None:
            annotations = annotations.get_axis_annotation(1)

            # Add distribution information to annotations metadata
            if annotations.has_metadata('distribution'):
                self.concept_annotations = annotations
            else:
                assert variable_distributions is not None, (
                    "variable_distributions must be provided if annotations "
                    "lack 'distribution' metadata."
                )
                self.concept_annotations = add_distribution_to_annotations(
                    annotations, variable_distributions
                )
            self.concept_names = self.concept_annotations.labels

        self._backbone = backbone

        if latent_encoder is not None:
            self._latent_encoder = latent_encoder(input_size,
                                    **(latent_encoder_kwargs or {}))
        elif latent_encoder_kwargs is not None:
            # assume an MLP encoder if latent_encoder_kwargs provided but no latent_encoder
            self._latent_encoder = MLP(input_size=input_size,
                                **latent_encoder_kwargs)
        else:
            self._latent_encoder = nn.Identity()

        self.latent_size = latent_encoder_kwargs.get('hidden_size') if latent_encoder_kwargs else input_size

    def __repr__(self):
        backbone_name = self.backbone.__class__.__name__ if self.backbone is not None else "None"
        latent_encoder_name = self._latent_encoder.__class__.__name__ if self._latent_encoder is not None else "None"
        return f"{self.__class__.__name__}(backbone={backbone_name}, latent_encoder={latent_encoder_name})"

    @property
    def backbone(self) -> BackboneType:
        """The backbone feature extractor.

        Returns the backbone module used for feature extraction from raw inputs.
        If None, the model expects pre-computed features as inputs.

        Returns
        -------
        BackboneType or None
            Backbone module (e.g., ResNet, ViT) or None if using pre-computed features.
        """
        return self._backbone

    @property
    def latent_encoder(self) -> nn.Module:
        """The encoder mapping backbone output to latent space.

        Returns the latent encoder module that transforms backbone features
        (or raw inputs if no backbone) into latent representations used by
        concept encoders.

        Returns
        -------
        nn.Module
            Latent encoder network (MLP, custom module, or nn.Identity if no encoding).
        """
        return self._latent_encoder

    # TODO: add decoder?
    # @property
    # def encoder(self) -> nn.Module:
    #     """The decoder mapping back to the input space.

    #     Returns:
    #         nn.Module: Decoder network.
    #     """
    #     return self._encoder

    @abstractmethod
    def filter_output_for_loss(self, forward_out, target):
        """Filter model outputs before passing to loss function.

        Override this method in your model to customize what outputs are passed to the loss.
        Useful when your model returns auxiliary outputs that shouldn't be
        included in loss computation or need specific formatting.

        This method is called automatically during Lightning training in the
        ``shared_step()`` method of Learner classes. For manual PyTorch training,
        you typically don't need to call this method explicitly.

        Parameters
        ----------
        forward_out : Any
            Raw model output from forward pass (typically concept predictions,
            but can include auxiliary outputs like attention weights, embeddings).
        target : torch.Tensor
            Ground truth labels/targets.

        Returns
        -------
        dict
            Dictionary with keys expected by your loss function. Common format:
            ``{'input': predictions, 'target': ground_truth}`` for standard losses.

        Notes
        -----
        - For standard losses like nn.BCEWithLogitsLoss, return format should match
          the loss function's expected signature.
        - This method enables models to return rich outputs (embeddings, attentions)
          without interfering with loss computation.
        - Must be implemented by all concrete model subclasses.

        Examples
        --------
        Standard implementation passes predictions and targets directly to loss:
        
        >>> def filter_output_for_loss(self, forward_out, target):
        ...     return {'input': forward_out, 'target': target}

        See Also
        --------
        filter_output_for_metrics : Similar filtering for metrics computation
        torch_concepts.nn.modules.high.learners.JointLearner.shared_step : Where this is called
        """
        pass

    @abstractmethod
    def filter_output_for_metrics(self, forward_out, target):
        """Filter model outputs before passing to metric computation.

        Override this method in your model to customize what outputs are passed to the metrics.
        Useful when your model returns auxiliary outputs that shouldn't be
        included in metric computation or viceversa.

        Args:
            forward_out: Model output (typically concept predictions).
            target: Ground truth concepts.
        Returns:
            dict: Filtered outputs for metric computation.
        """
        pass

    # ------------------------------------------------------------------
    # Features extraction helpers
    # ------------------------------------------------------------------

    def maybe_apply_backbone(
        self,
        x: torch.Tensor,
        backbone_args: Optional[Mapping[str, Any]] = None,
    ) -> torch.Tensor:
        """Apply the backbone to ``x`` unless features are pre-computed.

        Args:
            x (torch.Tensor): Raw input tensor or already computed embeddings.
            backbone_kwargs (Any): Extra keyword arguments forwarded to the
                backbone callable when it is invoked.

        Returns:
            torch.Tensor: Feature embeddings.

        Raises:
            TypeError: If backbone is not None and not callable.
        """

        if self.backbone is None:
            return x

        if not callable(self.backbone):
            raise TypeError(
                "The provided backbone is not callable. Received "
                f"instance of type {type(self.backbone).__name__}."
            )

        return self.backbone(x, **backbone_args if backbone_args else {})


    # ------------------------------------------------------------------
    # Output helpers
    # ------------------------------------------------------------------
    
    def filter_output_for_loss(self, out_concepts):
        """Filter model outputs before passing to loss function.

        Override this method to customize what outputs are passed to the loss.
        Useful when your model returns auxiliary outputs that shouldn't be
        included in loss computation or viceversa.

        Args:
            out_concepts: Model output (typically concept predictions).

        Returns:
            Filtered output passed to loss function. By default, returns
            out_concepts unchanged.

        Example:
            >>> def filter_output_for_loss(self, out):
            ...     # Only use concept predictions, ignore attention weights
            ...     return out['concepts']
        """
        return out_concepts
    
    def filter_output_for_metrics(self, out_concepts):
        """Filter model outputs before passing to metrics.

        Override this method to customize what outputs are passed to metrics.
        Useful when your model returns auxiliary outputs that shouldn't be
        included in metric computation or viceversa.

        Args:
            out_concepts: Model output (typically concept predictions).

        Returns:
            Filtered output passed to metrics. By default, returns
            out_concepts unchanged.
        """
        return out_concepts
