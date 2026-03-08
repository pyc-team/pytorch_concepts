"""
Shared utilities for Pyro-based inference engines.

This module provides functions and constants used by both
:class:`ImportanceSamplingInference` and :class:`SVIInference`
to avoid code duplication.
"""

from typing import Callable, Dict, Optional, Union

import torch
from torch.distributions import (
    Bernoulli, Categorical,
    RelaxedBernoulli, RelaxedOneHotCategorical,
    Normal, LogNormal, Beta, Gamma,
)

import pyro
import pyro.distributions as dist

from ..models.variable import Variable, ConceptVariable


# ------------------------------------------------------------------
# Supported distribution families
# ------------------------------------------------------------------

DISCRETE_DISTRIBUTIONS = (
    Bernoulli, RelaxedBernoulli,
    Categorical, RelaxedOneHotCategorical,
)

CONTINUOUS_DISTRIBUTIONS = (Normal, LogNormal, Beta, Gamma)


# ------------------------------------------------------------------
# Distribution helpers
# ------------------------------------------------------------------

def validate_pyro_distributions(variables) -> None:
    """
    Validate that every concept variable uses a Pyro-supported distribution.

    Args:
        variables: Iterable of :class:`Variable` instances from the
            probabilistic model.

    Raises:
        ValueError: If a variable has an unsupported distribution.
    """
    supported = DISCRETE_DISTRIBUTIONS + CONTINUOUS_DISTRIBUTIONS
    for var in variables:
        if var.distribution is not None and var.distribution not in supported:
            from torch_concepts.distributions import Delta
            if var.distribution != Delta:
                raise ValueError(
                    f"Variable '{var.concept}' has unsupported distribution: "
                    f"{var.distribution}. Supported: {supported}"
                )


def get_pyro_distribution(
    variable: Variable,
    params: Union[torch.Tensor, Dict[str, torch.Tensor]],
    batch_size: Optional[int] = None,
):
    """
    Convert a torch distribution type and CPD parameters to a Pyro distribution.

    Args:
        variable: Variable containing the distribution type.
        params: Parameters from the CPD.  Can be a raw tensor (treated as
            logits) or a dict with keys such as ``'logits'``, ``'probs'``,
            ``'loc'``, ``'scale'``, etc.
        batch_size: When provided, trailing singleton dimensions are only
            squeezed when ``shape[0] == batch_size``.  This prevents
            squeezing enumeration dimensions added by
            ``config_enumerate``.

    Returns:
        A ``pyro.distributions`` object, or ``None`` if the distribution
        is not supported.
    """
    if isinstance(params, torch.Tensor):
        params = {'logits': params}

    def _maybe_squeeze(t: torch.Tensor) -> torch.Tensor:
        """Squeeze trailing dim-1 only for standard (batch, 1) shapes."""
        if t.dim() > 1 and t.shape[-1] == 1:
            if batch_size is None or t.shape[0] == batch_size:
                return t.squeeze(-1)
        return t

    # Discrete distributions
    if variable.distribution in (Bernoulli, RelaxedBernoulli):
        logits = params.get('logits')
        if logits is not None:
            return dist.Bernoulli(logits=_maybe_squeeze(logits))
        probs = params.get('probs')
        if probs is not None:
            return dist.Bernoulli(probs=_maybe_squeeze(probs))

    elif variable.distribution in (Categorical, RelaxedOneHotCategorical):
        logits = params.get('logits')
        if logits is not None:
            return dist.Categorical(logits=logits)
        probs = params.get('probs')
        if probs is not None:
            return dist.Categorical(probs=probs)

    # Continuous distributions
    elif variable.distribution == Normal:
        loc = params.get('loc', params.get('mean', torch.zeros(1)))
        scale = params.get('scale', params.get('std', torch.ones(1)))
        return dist.Normal(loc=loc, scale=scale)

    elif variable.distribution == LogNormal:
        loc = params.get('loc', torch.zeros(1))
        scale = params.get('scale', torch.ones(1))
        return dist.LogNormal(loc=loc, scale=scale)

    elif variable.distribution == Beta:
        alpha = params.get('concentration1', params.get('alpha', torch.ones(1)))
        beta = params.get('concentration0', params.get('beta', torch.ones(1)))
        return dist.Beta(concentration1=alpha, concentration0=beta)

    elif variable.distribution == Gamma:
        concentration = params.get('concentration', params.get('alpha', torch.ones(1)))
        rate = params.get('rate', params.get('beta', torch.ones(1)))
        return dist.Gamma(concentration=concentration, rate=rate)

    return None


# ------------------------------------------------------------------
# Evidence helpers
# ------------------------------------------------------------------

def build_obs_dict_from_evidence(
    probabilistic_model,
    evidence: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    Extract concept observations from an evidence dictionary.

    Concept variables found in *evidence* are squeezed to
    ``(batch_size,)`` when they have a trailing dimension of 1, making
    them compatible with ``pyro.plate``.

    Args:
        probabilistic_model: The :class:`ProbabilisticModel`.
        evidence: Dict mapping variable names to tensors.

    Returns:
        Dict mapping concept names to observation tensors.
    """
    obs_dict: Dict[str, torch.Tensor] = {}
    for name, val in evidence.items():
        if name in probabilistic_model.concept_to_variable:
            var = probabilistic_model.concept_to_variable[name]
            if isinstance(var, ConceptVariable):
                if val.dim() == 2 and val.shape[-1] == 1:
                    val = val.squeeze(-1)
                obs_dict[name] = val
    return obs_dict


# ------------------------------------------------------------------
# Pyro model construction
# ------------------------------------------------------------------

def build_pyro_model(inference_engine) -> Callable:
    """
    Build a reusable Pyro model function for the given inference engine.

    The returned callable accepts ``(evidence, obs_dict=None)`` so it can
    be reused across training steps and marginal queries without
    rebuilding the model each time.  The *structural* parts of the
    model (DAG topology, CPDs, topological order) are captured once in
    the closure; the *data* parts (input tensors, observations) are
    provided at call time.

    Inside the model function:

    - Neural-network parameters are registered via ``pyro.module()``.
    - The batch dimension is parallelised via ``pyro.plate()``.
    - Observed variables are conditioned via ``obs=`` in
      ``pyro.sample()``.

    Args:
        inference_engine: A :class:`ForwardInference` subclass instance
            (must expose ``probabilistic_model``, ``sorted_variables``
            and ``get_parent_kwargs``).

    Returns:
        A callable ``model(evidence, obs_dict=None)`` that can be
        passed to ``SVI``, ``Importance``, ``Predictive``, etc.
    """
    probabilistic_model = inference_engine.probabilistic_model
    sorted_variables = inference_engine.sorted_variables
    _engine = inference_engine

    def model(evidence: Dict[str, torch.Tensor],
              obs_dict: Optional[Dict[str, torch.Tensor]] = None):
        if obs_dict is None:
            obs_dict = {}

        # Separate fixed evidence (inputs / latent) from concept evidence
        fixed_evidence: Dict[str, torch.Tensor] = {}
        for name, value in evidence.items():
            if name in probabilistic_model.concept_to_variable:
                var = probabilistic_model.concept_to_variable[name]
                if not isinstance(var, ConceptVariable):
                    fixed_evidence[name] = value
            else:
                fixed_evidence[name] = value

        batch_size = next(iter(evidence.values())).shape[0]

        # Register all CPD parameters with Pyro for gradient tracking
        pyro.module("probabilistic_model", probabilistic_model)

        # Start with fixed evidence (inputs / latent)
        values = dict(fixed_evidence)

        with pyro.plate("data", batch_size, dim=-1):
            for var in sorted_variables:
                concept_name = var.concept

                # Process root nodes through their CPDs (e.g., backbone)
                if not var.parents:
                    if concept_name in values:
                        cpd = probabilistic_model.get_module_of_concept(concept_name)
                        if cpd is not None:
                            parent_kwargs = _engine.get_parent_kwargs(
                                cpd, [values[concept_name]], []
                            )
                            values[concept_name] = cpd.forward(**parent_kwargs)
                    continue

                # Gather parent values
                parent_concepts = []
                parent_input = []
                all_parents_available = True

                for parent_var in var.parents:
                    parent_name = parent_var.concept
                    if parent_name not in values:
                        all_parents_available = False
                        break

                    if isinstance(parent_var, ConceptVariable):
                        parent_concepts.append(values[parent_name])
                    else:
                        parent_input.append(values[parent_name])

                if not all_parents_available:
                    continue

                # Get CPD and compute distribution parameters
                cpd = probabilistic_model.get_module_of_concept(concept_name)
                if cpd is None:
                    continue

                parent_kwargs = _engine.get_parent_kwargs(cpd, parent_input, parent_concepts)
                params = cpd.forward(**parent_kwargs)

                # Get observation for this variable if provided
                obs_value = obs_dict.get(concept_name, None)

                # Convert to Pyro distribution and sample
                pyro_dist = get_pyro_distribution(var, params, batch_size=batch_size)
                if pyro_dist is not None:
                    sampled_value = pyro.sample(concept_name, pyro_dist, obs=obs_value)

                    # Bernoulli samples are 0/1.  Downstream CPDs that apply
                    # sigmoid internally expect logits, so convert back to
                    # logit space.
                    if var.distribution in (Bernoulli, RelaxedBernoulli):
                        _LOGIT_SCALE = 10.0
                        sampled_value = (2 * sampled_value - 1) * _LOGIT_SCALE
                        if sampled_value.dim() == 1:
                            sampled_value = sampled_value.unsqueeze(-1)
                        values[concept_name] = sampled_value
                    else:
                        values[concept_name] = sampled_value

        return values

    return model
