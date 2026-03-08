"""
Stochastic Variational Inference for Probabilistic Graphical Models.

This module implements probabilistic inference and training using Pyro's
Stochastic Variational Inference (SVI), supporting both discrete and continuous
variables in concept-based models.
"""

from typing import Callable, Dict, List, Optional, Union

import torch
from torch.distributions import (
    Bernoulli, Categorical,
    RelaxedBernoulli, RelaxedOneHotCategorical,
    Normal, LogNormal, Beta, Gamma,
)

# Pyro imports
import pyro
from pyro.infer import SVI, Trace_ELBO, TraceEnum_ELBO, Predictive, config_enumerate
from pyro.infer.autoguide import AutoNormal
from pyro.optim import ClippedAdam

from .forward import ForwardInference, LazyForwardInference
from ..models.variable import Variable, ConceptVariable
from ..models.probabilistic_model import ProbabilisticModel
from ...low.base.graph import BaseGraphLearner
from .utils import (
    DISCRETE_DISTRIBUTIONS,
    CONTINUOUS_DISTRIBUTIONS,
    validate_pyro_distributions,
    get_pyro_distribution,
    build_obs_dict_from_evidence,
    build_pyro_model,
)


class SVIInference(ForwardInference):
    """
    Probabilistic inference and training using Pyro Stochastic Variational Inference.

    This inference engine uses Pyro's SVI to optimise the ELBO over the MLP
    parameters of a concept-based probabilistic model.  During training all
    concept variables are observed, so the posterior collapses to a point and
    SVI effectively performs maximum-likelihood estimation.  At test time, the
    trained guide is used with ``Predictive`` to draw posterior samples and
    estimate marginal probabilities for unobserved (query) variables.

    Key Features:
        - SVI-based training of concept model parameters
        - Marginal computation via ``Predictive`` at test time
        - Support for both discrete and continuous variables
        - Evidence conditioning via ``obs=`` parameter
        - Conditional probability queries
        - Integration with trained CBMs and PGMs

    Args:
        probabilistic_model: The probabilistic model to perform inference on.
        graph_learner: Optional graph learner for weighted adjacency structure.
        num_samples: Number of posterior samples for marginal estimation.
        lr: Learning rate for the SVI optimiser (ClippedAdam).
        enumerate_discrete: If ``True``, use ``TraceEnum_ELBO`` with
            ``config_enumerate`` to sum out unobserved discrete variables
            exactly.  No guide is needed in this mode.
        **kwargs: Additional arguments passed to ForwardInference.

    Example:
        >>> import torch
        >>> import pyro
        >>> from torch.distributions import Bernoulli
        >>> from torch_concepts import LatentVariable, ConceptVariable
        >>> from torch_concepts.nn import (
        ...     SVIInference, ParametricCPD, ProbabilisticModel
        ... )
        >>>
        >>> # Create a simple PGM: input -> [A, B] -> C
        >>> input_var = LatentVariable('input', parents=[], size=10)
        >>> var_A = ConceptVariable('A', parents=['input'], distribution=Bernoulli)
        >>> var_B = ConceptVariable('B', parents=['input'], distribution=Bernoulli)
        >>> var_C = ConceptVariable('C', parents=['A', 'B'], distribution=Bernoulli)
        >>>
        >>> # ... define CPDs and create model ...
        >>>
        >>> # Create SVI inference engine
        >>> inference = SVIInference(model, num_samples=1000, lr=0.005)
        >>>
        >>> # Build guide and SVI optimiser for the training loop
        >>> guide, svi = inference.build_svi()
        >>> evidence = {'input': x}
        >>> obs = {'A': c_train[:, 0], 'B': c_train[:, 1], 'C': c_train[:, 2]}
        >>> for step in range(2000):
        ...     loss = svi.step(evidence, obs_dict=obs)
        >>>
        >>> # Query marginal: p(A | x)
        >>> p_A = inference.marginal(['A'], evidence={'input': x})
        >>>
        >>> # Conditional query: p(C | A=1, x)
        >>> p_C_given_A = inference.marginal(
        ...     ['C'],
        ...     evidence={'input': x, 'A': torch.ones(batch_size, 1)}
        ... )
    """

    def __init__(
        self,
        probabilistic_model: ProbabilisticModel,
        graph_learner: BaseGraphLearner = None,
        num_samples: int = 1000,
        lr: float = 0.005,
        enumerate_discrete: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(probabilistic_model, graph_learner, *args, **kwargs)
        self.num_samples = num_samples
        self.lr = lr
        self.enumerate_discrete = enumerate_discrete

        # Validate that all variables have supported distributions
        validate_pyro_distributions(self.probabilistic_model.variables)

        # Build the Pyro model once; it accepts (evidence, obs_dict) at call time.
        self._pyro_model: Callable = build_pyro_model(self)

        # Populated by build_svi(); used by marginal() / query().
        self._guide: Optional[Callable] = None
        self._svi_built: bool = False

    # ------------------------------------------------------------------
    # ForwardInference hooks
    # ------------------------------------------------------------------

    def get_results(self, results: torch.Tensor, variable: Variable) -> torch.Tensor:
        """
        Return raw output (logits/parameters) for Pyro model building.

        For SVI inference, we return the raw CPD outputs which contain the
        distribution parameters needed to construct Pyro distributions.

        Args:
            results: Raw output tensor from the CPD (logits or parameters).
            variable: The variable being computed.

        Returns:
            torch.Tensor: Raw output tensor unchanged.
        """
        return results

    def ground_truth_to_evidence(
        self,
        value: torch.Tensor,
        cardinality: int,
    ) -> torch.Tensor:
        """
        Convert ground truth to evidence format for conditioning.

        For SVI with plate notation, evidence should be ``(batch_size,)`` for
        scalar variables (Bernoulli, Categorical class indices) so that it is
        compatible with ``pyro.plate``.

        Args:
            value: Ground truth tensor. Shape: ``(batch_size,)`` or ``(batch_size, 1)``.
            cardinality: Number of classes for this variable.

        Returns:
            torch.Tensor: Evidence tensor (hard values), shape ``(batch_size,)``.
        """
        if value.dim() == 2 and value.shape[-1] == 1:
            value = value.squeeze(-1)
        return value

    # ------------------------------------------------------------------
    # build_svi – create Pyro model, guide, and SVI for user training loop
    # ------------------------------------------------------------------

    def build_svi(
        self,
        lr: Optional[float] = None,
    ) -> tuple[Optional[Callable], SVI]:
        """
        Build a Pyro guide and SVI optimiser.

        Call this once before writing your own training loop.  The
        returned ``svi`` object exposes ``svi.step(evidence, obs_dict=obs)``
        which performs a single ELBO optimisation step.

        When ``enumerate_discrete=True``, no guide is needed: all
        unobserved discrete variables are summed out exactly via
        ``TraceEnum_ELBO`` and ``config_enumerate``.  In that case
        the first element of the returned tuple is ``None``.

        The guide (if any) is stored internally so that ``marginal()``
        / ``query()`` can be used after training.

        Args:
            lr: Learning rate for the ``ClippedAdam`` optimiser
                (overrides default).

        Returns:
            Tuple of ``(guide, svi)``.  ``guide`` is ``None`` when
            ``enumerate_discrete=True``.
        """
        if lr is None:
            lr = self.lr

        pyro.clear_param_store()

        if self.enumerate_discrete:
            self._guide = None
            svi = SVI(
                config_enumerate(self._pyro_model, "parallel"),
                lambda *args, **kwargs: None,
                ClippedAdam({"lr": lr}),
                loss=TraceEnum_ELBO(max_plate_nesting=1),
            )
        else:
            self._guide = AutoNormal(self._pyro_model)
            svi = SVI(
                self._pyro_model,
                self._guide,
                ClippedAdam({"lr": lr}),
                loss=Trace_ELBO(),
            )

        self._svi_built = True
        return self._guide, svi

    # ------------------------------------------------------------------
    # marginal – posterior predictive queries
    # ------------------------------------------------------------------

    def marginal(
        self,
        query: List[str],
        evidence: Dict[str, torch.Tensor],
        num_samples: Optional[int] = None,
        return_dict: bool = False,
    ) -> Union[torch.Tensor, Dict[str, Dict[str, torch.Tensor]]]:
        """
        Compute marginal probabilities using the trained SVI guide.

        This method computes ``p(query | evidence)`` by running
        ``Predictive`` with the learned guide, drawing posterior
        samples and aggregating them.

        Args:
            query: List of variable names to compute marginals for.
            evidence: Dictionary of observed variables ``{name: tensor}``.
                Must include at least the input variables.
            num_samples: Number of posterior samples (overrides default).
            return_dict: If True, return dict with detailed statistics.
                If False, return concatenated probability tensor.

        Returns:
            If *return_dict* is ``False`` (default):
                torch.Tensor — Marginal probabilities concatenated.
                    - Bernoulli: ``p(variable=1)``, shape ``(batch_size, 1)``
                    - Categorical: ``[p(class_0), …, p(class_K)]``, shape ``(batch_size, K)``
                    - Continuous: mean value, shape ``(batch_size, size)``

            If *return_dict* is ``True``:
                Dict[str, Dict[str, torch.Tensor]] — Per-variable statistics.
                    - Discrete: ``{'probs': tensor}``
                    - Continuous: ``{'mean': tensor, 'std': tensor}``

        Example:
            >>> # Single marginal
            >>> p_A = inference.marginal(['A'], {'input': x})
            >>>
            >>> # Multiple marginals
            >>> p_AB = inference.marginal(['A', 'B'], {'input': x})
            >>>
            >>> # Conditional marginal
            >>> p_C_given_A = inference.marginal(
            ...     ['C'],
            ...     {'input': x, 'A': torch.ones(batch, 1)}
            ... )
        """
        if num_samples is None:
            num_samples = self.num_samples

        # Validate query variables exist
        for var_name in query:
            if var_name not in self.probabilistic_model.concept_to_variable:
                raise ValueError(f"Variable '{var_name}' not found in model.")

        if not self._svi_built:
            raise RuntimeError(
                "SVI has not been initialised. Call build_svi() and "
                "run your training loop before querying marginals."
            )

        batch_size = next(iter(evidence.values())).shape[0]

        # Build obs dict from concept evidence for conditioning
        concept_evidence = build_obs_dict_from_evidence(
            self.probabilistic_model, evidence
        )

        # Determine which sites are latent (not in obs)
        latent_sites = [
            name for name in query if name not in concept_evidence
        ]

        # Use Predictive to draw posterior samples
        predictive = Predictive(
            self._pyro_model,
            guide=self._guide,
            num_samples=num_samples,
            return_sites=latent_sites,
        )
        samples = predictive(evidence, obs_dict=concept_evidence)
        # samples[site] shape: (num_samples, batch_size) or (num_samples, batch_size, K)

        results: Dict[str, Dict[str, torch.Tensor]] = {}
        for var_name in query:
            var = self.probabilistic_model.concept_to_variable[var_name]

            if var_name in concept_evidence:
                # Variable is observed — return the evidence directly
                val = concept_evidence[var_name]
                if var.distribution in (Bernoulli, RelaxedBernoulli):
                    if val.dim() == 1:
                        val = val.unsqueeze(-1)
                    results[var_name] = {'probs': val.float()}
                elif var.distribution in (Categorical, RelaxedOneHotCategorical):
                    num_classes = var.size
                    probs = torch.zeros(batch_size, num_classes)
                    indices = val.long()
                    if indices.dim() == 2:
                        indices = indices.squeeze(-1)
                    probs.scatter_(1, indices.unsqueeze(-1), 1.0)
                    results[var_name] = {'probs': probs}
                else:
                    if val.dim() == 1:
                        val = val.unsqueeze(-1)
                    results[var_name] = {'mean': val.float(), 'std': torch.zeros_like(val)}
                continue

            # Latent site — aggregate posterior samples
            site_samples = samples[var_name]
            # site_samples shape: (num_samples, batch_size) or (num_samples, batch_size, K)

            if var.distribution in (Bernoulli, RelaxedBernoulli):
                probs = site_samples.float().mean(dim=0).unsqueeze(-1)
                results[var_name] = {'probs': probs}

            elif var.distribution in (Categorical, RelaxedOneHotCategorical):
                num_classes = var.size
                probs = torch.zeros(batch_size, num_classes)
                for k in range(num_classes):
                    probs[:, k] = (site_samples == k).float().mean(dim=0)
                results[var_name] = {'probs': probs}

            else:
                # Continuous
                mean = site_samples.mean(dim=0)
                std = site_samples.std(dim=0)
                if mean.dim() == 1:
                    mean = mean.unsqueeze(-1)
                    std = std.unsqueeze(-1)
                results[var_name] = {'mean': mean, 'std': std}

        if return_dict:
            return results

        # Concatenate probabilities/means into single tensor
        tensors = []
        for var_name in query:
            var = self.probabilistic_model.concept_to_variable[var_name]
            if var.distribution in DISCRETE_DISTRIBUTIONS:
                tensors.append(results[var_name]['probs'])
            else:
                tensors.append(results[var_name]['mean'])
        return torch.cat(tensors, dim=-1)

    def _get_pyro_distribution(self, variable, params):
        """Convert distribution.  See :func:`.utils.get_pyro_distribution`."""
        return get_pyro_distribution(variable, params)

    def query(
        self,
        query: List[str],
        evidence: Dict[str, torch.Tensor],
        **kwargs,
    ) -> torch.Tensor:
        """Query interface — delegates to :meth:`marginal`."""
        return self.marginal(query, evidence, return_dict=False, **kwargs)


class LazySVIInference(LazyForwardInference, SVIInference):
    """
    Lazy SVI inference that only computes ancestor variables.

    Combines the lazy query strategy (computing only ancestors of queried
    concepts) with SVI inference via Pyro.

    Use this when:
        - You only need marginals for a subset of concepts
        - The graph has many independent branches
        - You want to avoid computing unused variables

    Args:
        probabilistic_model: The probabilistic model to perform inference on.
        graph_learner: Optional graph learner for weighted adjacency structure.
        num_samples: Number of posterior samples for marginal estimation.
        lr: Learning rate for the SVI optimiser.
        **kwargs: Additional arguments passed to parent classes.

    Example:
        >>> # Given model: input -> A -> B, input -> C -> D
        >>> inference = LazySVIInference(pgm, num_samples=1000)
        >>> obs = {'A': c[:, 0], 'B': c[:, 1], 'C': c[:, 2], 'D': c[:, 3]}
        >>> guide, svi = inference.build_svi()
        >>> evidence = {'input': x}
        >>> for step in range(2000):
        ...     svi.step(evidence, obs_dict=obs)
        >>> # Querying only ['B'] computes marginals for: input, A, B (not C, D)
        >>> p_B = inference.marginal(['B'], evidence={'input': x})
    """
    pass
