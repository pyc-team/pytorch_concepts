r"""MutilatedNetworkProposal — likelihood weighting as an importance proposal.

The *mutilated network* :math:`B_z` is obtained from a Bayesian network by, for
every evidence variable :math:`Z_k = z_k`, severing its incoming arcs and pinning
it to the observed value — turning it into a constant. Every other node keeps its
original CPD. Sampling the non-evidence variables from their **own model CPDs**
(evidence clamped) and importance-correcting against the true joint is exactly
*likelihood weighting*.

Why this proposal needs no parameters
-------------------------------------
:class:`ImportanceSampling` forms the log-weight as
:math:`\log w = \log p(z, e) - \log q(z \mid e)`. The mutilated-network proposal
samples each non-evidence node :math:`X_j` from the model itself, so

.. math::

    \log q(z \mid e) = \sum_{j \notin Z} \log P(x_j \mid \mathrm{PA}_j),

while the model target is :math:`\log p = \sum_{i} \log P(x_i \mid \mathrm{PA}_i)`.
The non-evidence terms cancel and only the evidence likelihood survives:

.. math::

    w(\xi) = \frac{\prod_i P(x_i \mid \mathrm{PA}_i)}
                  {\prod_{j \notin Z} P(x_j \mid \mathrm{PA}_j)}
           = \prod_{Z_k} P(z_k \mid \mathrm{PA}(Z_k)),

i.e. each particle is weighted by the product of the CPDs whose children are the
evidence variables, evaluated at the evidence. This is the classic likelihood
weighting weight; the cancellation is realised numerically because the engine
scores the *same* sampled values under both :math:`p` and :math:`q`.

This proposal therefore carries no learnable parameters of its own — it reuses
the PGM's CPDs — and serves as the natural baseline against which a custom,
evidence-aware proposal should be compared. It works well when the evidence is
near the roots and degrades (low effective sample size) when the evidence is on
the leaves, which is the very regime a tailored proposal is meant to fix.
"""
from __future__ import annotations

from typing import Dict

import torch

from ....models.variable import Variable
from .base_proposal import BaseProposal


class MutilatedNetworkProposal(BaseProposal):
    """Likelihood-weighting proposal: sample non-evidence nodes from the model.

    Parameters
    ----------
    pgm : BayesianNetwork
        The network whose CPDs are reused as the proposal.
    """

    name = "MutilatedNetworkProposal"

    def propose(
        self,
        variable: Variable,
        parent_values: Dict[str, torch.Tensor],
        evidence: Dict[str, torch.Tensor],
        batch_size: int,
        temperature: torch.Tensor,
        layer_kwargs: Dict,
    ) -> Dict[str, torch.Tensor]:
        """Return the variable's **model** CPD parameters (no evidence conditioning).

        Roots return an unbatched parameter from the model CPD, broadcast here to
        the requested ``batch_size``; non-roots are evaluated on the already
        sampled / clamped parent values, exactly as the generative model would.
        """
        cpd = self.pgm.name_to_factor(variable.name)
        if cpd.is_root:
            params = cpd(parent_values={})
            return {
                key: value.unsqueeze(0).expand(batch_size, *value.shape)
                for key, value in params.items()
            }
        return cpd(parent_values=parent_values, **layer_kwargs)
