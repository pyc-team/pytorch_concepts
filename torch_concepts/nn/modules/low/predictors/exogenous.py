import torch

from ..base.layer import BasePredictor
from ....functional import grouped_concept_exogenous_mixture, replace_expand_cols
from typing import List, Callable


class MixConceptExogegnousToConcept(BasePredictor):
    """
    Concept exogenous predictor with mixture of concept activations and exogenous features.

    This predictor implements the Concept Embedding Model (CEM) task predictor that
    combines concept activations with learned exogenous using a mixture operation.

    Main reference: "Concept Embedding Models: Beyond the Accuracy-Explainability
    Trade-Off" (Espinosa Zarlenga et al., NeurIPS 2022).

    Attributes:
        in_concepts (int): Number of input concepts.
        in_exogenous (int): Number of exogenous features.
        out_concepts (int): Number of output concepts.
        cardinalities (List[int]): Cardinalities for grouped concepts.
        predictor (nn.Module): Linear predictor module.

    Args:
        in_concepts: Number of input concepts.
        in_exogenous: Number of exogenous features (must be even).
        out_concepts: Number of output concepts.
        activation: Activation function for concept logits (default: sigmoid).
        cardinalities: List of concept group cardinalities (optional).

    Example:
        >>> import torch
        >>> from torch_concepts.nn import MixConceptExogegnousToConcept
        >>>
        >>> # Create predictor with 10 concepts, 20 exogenous dims, 3 output concepts
        >>> predictor = MixConceptExogegnousToConcept(
        ...     in_concepts=10,
        ...     in_exogenous=10,  # Must be half of exogenous latent size when no cardinalities are provided
        ...     out_concepts=3,
        ...     activation=torch.sigmoid
        ... )
        >>>
        >>> # Generate random inputs
        >>> concepts = torch.randn(4, 10)  # batch_size=4, n_concepts=10
        >>> exogenous = torch.randn(4, 10, 20)  # (batch, n_concepts, emb_size)
        >>>
        >>> # Forward pass
        >>> output = predictor(concepts=concepts, exogenous=exogenous)
        >>> print(output.shape)  # torch.Size([4, 3])
        >>>
        >>> # With concept groups (e.g., color has 3 values, shape has 4, etc.)
        >>> predictor_grouped = MixConceptExogegnousToConcept(
        ...     in_concepts=10,
        ...     in_exogenous=20, # Must be equal to exogenous latent size when cardinalities are provided
        ...     out_concepts=3,
        ...     cardinalities=[3, 4, 3]  # 3 groups summing to 10
        ... )
        >>>
        >>> # Forward pass with grouped concepts
        >>> output = predictor_grouped(concepts=concepts, exogenous=exogenous)
        >>> print(output.shape)  # torch.Size([4, 3])

    References:
        Espinosa Zarlenga et al. "Concept Embedding Models: Beyond the
        Accuracy-Explainability Trade-Off", NeurIPS 2022.
        https://arxiv.org/abs/2209.09056
    """
    def __init__(
        self,
        in_concepts: int,
        in_exogenous: int,
        out_concepts: int,
        cardinalities: List[int],
        activation: Callable = torch.sigmoid
    ):
        super().__init__(
            in_concepts=in_concepts,
            in_exogenous=in_exogenous,
            out_concepts=out_concepts,
            activation=activation,
        )
        if cardinalities is None:
            # assume all binary
            self.cardinalities = [1] * in_concepts
        else:
            self.cardinalities = cardinalities
            assert sum(self.cardinalities) == in_concepts

        # find positions of concepts with cardinality 1 for Bernoulli to Categorical splitting
        self.cardinalities_expanded = torch.tensor(cardinalities)
        cumsum = torch.cumsum(self.cardinalities_expanded, dim=0)
        start_positions = cumsum - self.cardinalities_expanded
        self.mask_cardinality_1 = start_positions[self.cardinalities_expanded == 1]
        self.cardinalities_expanded[self.cardinalities_expanded == 1] = 2

        self.bernoulli_to_categorical_exogenous_splitter = torch.nn.Sequential(
            torch.nn.Linear(in_exogenous, in_exogenous*2),
            torch.nn.LeakyReLU(),
            torch.nn.Unflatten(-1, (-1, in_exogenous)),
        )
        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(
                in_exogenous * len(self.cardinalities),
                out_concepts
            ),
            torch.nn.Unflatten(-1, (out_concepts,)),
        )

    def forward(
        self,
        concepts: torch.Tensor,
        exogenous: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through the predictor.

        Args:
            concepts: Concept logits of shape (batch_size, in_concepts).
            exogenous: Concept exogenous of shape (batch_size, in_concepts, exogenous_dim).

        Returns:
            torch.Tensor: Output concepts of shape (batch_size, out_concepts).
        """
        in_probs = self.activation(concepts)

        # For concepts with cardinality 1, split the Bernoulli probability into a categorical distribution
        if len(self.mask_cardinality_1) > 0:
            exogenous_split = self.bernoulli_to_categorical_exogenous_splitter(exogenous[:, self.mask_cardinality_1])
            in_probs_split = torch.cat([
                in_probs[:, self.mask_cardinality_1[:, None]],
                1-in_probs[:, self.mask_cardinality_1[:, None]],
            ], dim=-1)
            exogenous = replace_expand_cols(exogenous, self.mask_cardinality_1, exogenous_split)
            in_probs = replace_expand_cols(in_probs, self.mask_cardinality_1, in_probs_split)

        c_mix = grouped_concept_exogenous_mixture(
            exogenous,
            in_probs,
            groups=list(self.cardinalities_expanded),
        )
        return self.predictor(c_mix.flatten(start_dim=1))
    

class MixMemoryConceptExogenousToConcept(BasePredictor):
    """
    Memory-based concept-to-task predictor used in Concept-based Memory Reasoner.

    This predictor combines concept probabilities with rule selection probabilities
    and a learned task-specific rule memory. Each task has an embedding that is
    decoded into rule parameters with 3 coefficients per concept-rule pair.
    Input concepts are treated as Bernoulli random variables.

    Main reference: "Interpretable Concept-Based Memory Reasoning"
    (Debot et al., NeurIPS 2024).

    Attributes:
        in_concepts (int): Number of input concepts.
        in_exogenous (int): Number of rules per output concept.
        out_concepts (int): Number of output concepts.
        eps (float): Small scaling factor to avoid floating-point problems with the softmax.
        memory_decoder_hidden_layers (int): Number of hidden layers in
            ``memory_decoder``.
        memory_network_shape (tuple[int, int, int]): Decoded memory tensor shape
            ``(in_exogenous, in_concepts, 3)``.
        memory (nn.Embedding): Learnable task memory embeddings.
        memory_decoder (nn.Sequential): Decoder from memory embeddings to rule parameters.

    Args:
        in_concepts: Number of input concepts.
        in_exogenous: Number of rules per output concept.
        out_concepts: Number of output tasks.
        memory_latent_size: Size of the learned task memory embedding.
        memory_decoder_hidden_layers: Number of hidden layers in the memory
            decoder MLP. Must be >= 0.
        eps: Scaling factor applied after memory softmax to avoid floating-point error problems.

    Input Shapes:
        - ``concepts``: ``(batch_size, in_concepts)``
        - ``exogenous``: ``(batch_size, out_concepts, in_exogenous)``

    Output Shape:
        - ``(batch_size, out_concepts)`` task probabilities.

    References:
        Debot et al. "Interpretable Concept-Based Memory Reasoning", NeurIPS 2024. https://arxiv.org/abs/2407.15527
    """
    def __init__(
        self,
        in_concepts: int,   # concepts
        in_exogenous: int,  # rules
        out_concepts: int,  # tasks
        memory_latent_size: int = 100, # size of the learned rule memory latent space
        memory_decoder_hidden_layers: int = 1,
        eps: float = 0.001
    ):
        super().__init__(
            in_concepts=in_concepts,
            in_exogenous=in_exogenous,
            out_concepts=out_concepts,
        )

        if memory_decoder_hidden_layers < 0:
            raise ValueError("memory_decoder_hidden_layers must be >= 0")

        self.eps = eps
        self.memory_decoder_hidden_layers = memory_decoder_hidden_layers

        self.memory_network_shape = (in_exogenous, in_concepts, 3)  # nb_rules, nb_concepts, 3 (for 3 memory slots per concept)
        self.memory_network_latent = self.memory_network_shape[0] * self.memory_network_shape[1] * self.memory_network_shape[2]
        
        self.memory = torch.nn.Embedding(out_concepts, memory_latent_size)

        decoder_layers = [torch.nn.Linear(memory_latent_size, self.memory_network_latent)]
        for _ in range(memory_decoder_hidden_layers):
            decoder_layers.extend([
                torch.nn.LeakyReLU(),
                torch.nn.Linear(self.memory_network_latent, self.memory_network_latent),
            ])
        decoder_layers.append(torch.nn.Unflatten(-1, self.memory_network_shape))
        self.memory_decoder = torch.nn.Sequential(*decoder_layers)

    def forward(
        self,
        concepts: torch.Tensor,
        exogenous: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass through the predictor.

        Args:
            concepts: Concept logits of shape (batch_size, in_concepts) representing Bernoulli random variables.
            exogenous: Concept exogenous of shape (batch_size, out_concepts, exogenous_dim) representing rule selection probabilities.
            **kwargs: Optional controls:
                - include_rec (bool): If True, include a rule reconstruction-quality term.
                - rec_weight (float): Exponent applied to the reconstruction-quality term.
                - hard_roles (bool): If True, discretize decoded memory roles with argmax.

        Returns:
            torch.Tensor: Task probabilities of shape (batch_size, out_concepts).

        Note:
            Concept probabilities are detached from the graph in this method,
            so gradients do not flow from the task loss back into ``concepts``, avoiding task leakage.
        """
        include_rec = kwargs.get("include_rec", False)
        rec_weight = kwargs.get("rec_weight", 1.0)
        hard_roles = kwargs.get("hard_roles", False)

        c_probs = torch.sigmoid(concepts).detach()
        c_probs_expanded = c_probs.unsqueeze(1).unsqueeze(1).expand(-1, exogenous.shape[1], exogenous.shape[2], -1)  # (batch_size, out_concepts, in_exogenous, in_concepts)

        # Decode the memory
        memory_decoded = self.memory_decoder(self.memory.weight)  # (out_concepts, in_exogenous, in_concepts, 3) = (nb_tasks, nb_rules, nb_concepts, 3)
        memory_decoded = (1-self.eps) * torch.softmax(memory_decoded, dim=-1)
        memory_decoded_expanded = memory_decoded.unsqueeze(0).expand(concepts.shape[0], -1, -1, -1, -1)  # (batch_size, out_concepts, in_exogenous, in_concepts, 3)

        if hard_roles:
            role_indices = torch.argmax(memory_decoded_expanded, dim=-1)  # (batch_size, out_concepts, in_exogenous, in_concepts)
            memory_decoded_expanded = torch.nn.functional.one_hot(role_indices, num_classes=3).float()  # (batch_size, out_concepts, in_exogenous, in_concepts, 3)

        # Use CMR's inference equations to compute each task using the predicted concepts, the decoded memory (rules), and the exogenous (rule selection probabilities)
        y_per_rule = (c_probs_expanded * memory_decoded_expanded[..., 0] + (1-c_probs_expanded) * memory_decoded_expanded[..., 1] + memory_decoded_expanded[..., 2]).prod(dim=3)  # (batch_size, out_concepts, in_exogenous)
        if include_rec:
            y_rec_per_rule = (c_probs_expanded * memory_decoded_expanded[..., 0] + (1-c_probs_expanded) * memory_decoded_expanded[..., 1] + 0.5 * memory_decoded_expanded[..., 2]).prod(dim=3)  # (batch_size, out_concepts, in_exogenous)
            y_rec_per_rule = torch.pow(y_rec_per_rule + 1e-6, rec_weight)
        else:
            y_rec_per_rule = torch.ones_like(y_per_rule)  # dummy

        y_pred = (y_per_rule * y_rec_per_rule * exogenous).sum(dim=2)  # (batch_size, out_concepts)

        return y_pred
