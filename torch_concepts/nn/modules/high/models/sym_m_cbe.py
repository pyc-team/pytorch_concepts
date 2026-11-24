"""Symbolic Mixture of Concept Bottleneck Experts (Sym-M-CBE).

Training has three stages:
1. train the model end-to-end with MLP experts (placeholders);
2. :meth:`~SymbolicMixtureOfConceptBottleneckExperts.distill_experts` — symbolic
   regression (SymTorch / PySR) of each expert on the samples routed to it;
3. :meth:`~SymbolicMixtureOfConceptBottleneckExperts.set_symbolic_experts`, then
   fine-tune the expression constants together with encoder and selector.

With ``lightning=True`` and ``mlp_epochs`` set, a single ``Trainer.fit`` runs
all three stages (like ``PostHocCBM``): stage 2 runs at the start of epoch
``mlp_epochs``, and stage 3 takes the remaining epochs.

References
----------
De Santis et al. "Mixture of Concept Bottleneck Experts", ICML 2026.
https://arxiv.org/abs/2602.02886
"""
import re
from typing import Dict, List, Optional, Union

import sympy
import torch
from pyro.distributions import RelaxedOneHotCategoricalStraightThrough
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch import nn
from torch.distributions import Bernoulli, OneHotCategorical

from .....annotations import Annotations
from .....distributions import Delta
from ...low.dense_layers import MLP
from ...low.encoders.linear import LinearEmbeddingToConcept
from ...low.lazy import LazyConstructor
from ...low.predictors.mixture import (
    MixtureOfExpertsConceptToConcept,
    SymbolicConceptToConcept,
)
from ...mid.distributions import DEFAULT_DIST_KWARGS
from ...mid.factors.cpd import ParametricCPD
from ...mid.graph.bayesian_network import BayesianNetwork
from ...mid.inference.base import BaseInference
from ...mid.inference.torch.ancestral import AncestralSamplingInference
from ...mid.inference.torch.map_forward import MAPForwardInference
from ...mid.variable import EmbeddingVariable
from ..base.bipartite import BipartiteModel


class SymbolicMixtureOfConceptBottleneckExperts(BipartiteModel):
    """Concept bottleneck whose task predictor is a mixture of symbolic experts.

    The PGM is ``input -> latent -> concepts``, ``latent -> selector`` and
    ``(concepts, selector) -> tasks``. ``selector`` is a categorical variable
    over the ``n_experts`` experts; the task CPD averages the outputs of all
    experts with the selector value. The inference engine decides how the
    selector is drawn: a hard sample (one-hot) selects one expert, while
    a soft sample give a weighted average.

    Args:
        input_size: Number of input features.
        annotations: Dataset annotations (concepts and tasks).
        task_names: Name or names of the task labels.
        n_experts: Number of experts ``M``.
        expert_hidden_size: Hidden units of each MLP expert.
        expert_hidden_layers: Hidden layers of each MLP expert; ``0`` gives
            linear experts.
        selector_hidden_layers: Hidden layers of the selector network.
        selector_entropy_weight: Weight of a training term that maximises the
            entropy of the batch-average selector distribution ``E_x[p(m|x)]``
            (equivalently, minimises its KL to the uniform), so that the
            experts do not collapse onto one. ``0`` (default) disables it.
            With ``lightning=False``, subtract
            ``selector_entropy_weight * model.selector_marginal_entropy()``
            from the loss yourself.
        inference: Evaluation engine. Defaults to ``MAPForwardInference``
            (the most probable expert).
        train_inference: Training engine. Defaults to
            ``AncestralSamplingInference``, with a hard (straight-through)
            Gumbel-Softmax selector annealed exponentially from 2 to 0.05
            over ~10000 steps.
        lightning, plate, **kwargs: As in the other high-level models.

    Lightning stages (only with ``lightning=True``):
        mlp_epochs: Epochs of stage 1 (MLP experts). The experts are then
            distilled and fine-tuned for the remaining ``max_epochs``.
            ``None`` (default) never switches: run the stages by hand.
        distill_kwargs: Keyword arguments of :meth:`distill_experts`
            (``target``, ``sr_params``, ``save_path``, ``max_samples``).
        finetune_lr_factor: Learning-rate multiplier of stage 3 (the paper
            uses 5).

    References:
        De Santis et al. "Mixture of Concept Bottleneck Experts", ICML 2026.
        https://arxiv.org/abs/2602.02886
    """

    supported_concept_types = frozenset({"binary", "categorical", "continuous"})
    param_for_discrete_var = "logits"
    # NOTE: continuous variables are point predictions (Delta + MSE), as in the paper.
    variable_distributions = {
        "binary": Bernoulli,
        "categorical": OneHotCategorical,
        "continuous": Delta,
    }
    variable_dist_kwargs = dict(DEFAULT_DIST_KWARGS)

    def __init__(
        self,
        input_size: int,
        annotations: Annotations,
        task_names: Union[List[str], str],
        n_experts: int = 2,
        expert_hidden_size: int = 64,
        expert_hidden_layers: int = 1,
        selector_hidden_layers: int = 1,
        selector_entropy_weight: float = 0.0,
        inference: Optional[BaseInference] = MAPForwardInference,
        inference_kwargs: Optional[dict] = None,
        train_inference: Optional[BaseInference] = AncestralSamplingInference,
        train_inference_kwargs: Optional[dict] = None,
        lightning: bool = False,
        plate: Optional[bool] = None,
        mlp_epochs: Optional[int] = None,
        distill_kwargs: Optional[dict] = None,
        finetune_lr_factor: float = 5.0,
        **kwargs,
    ):
        super().__init__(
            input_size=input_size,
            annotations=annotations,
            task_names=task_names,
            lightning=lightning,
            plate=plate,
            **kwargs,
        )
        if n_experts <= 0:
            raise ValueError("n_experts must be positive.")
        if expert_hidden_layers < 0 or selector_hidden_layers < 0:
            raise ValueError("Hidden layer counts must be non-negative.")
        self.n_experts = n_experts
        self.expert_hidden_size = expert_hidden_size
        self.expert_hidden_layers = expert_hidden_layers
        self.selector_hidden_layers = selector_hidden_layers
        self.mlp_epochs = mlp_epochs
        self.distill_kwargs = dict(distill_kwargs or {})
        self.finetune_lr_factor = finetune_lr_factor
        if selector_entropy_weight < 0:
            raise ValueError("selector_entropy_weight must be non-negative.")
        self.selector_entropy_weight = selector_entropy_weight
        self.pgm = self._build_model()
        # Keep the selector value the predictors received, for the entropy term.
        for mixture in self.experts().values():
            mixture.register_forward_pre_hook(self._keep_selector_value, with_kwargs=True)

        if train_inference_kwargs is None and train_inference is AncestralSamplingInference:
            # NOTE: Gumbel temperature 2 -> 0.05 in ~10k steps; exponential
            # (built-in) instead of the paper's cosine. The engine also draws
            # relaxed samples of the concepts.
            train_inference_kwargs = dict(
                initial_temperature=2.0, final_temperature=0.05,
                annealing="exponential", annealing_rate=3.7e-4,
            )
        self.setup_inference(
            inference, inference_kwargs, train_inference, train_inference_kwargs
        )

    def _expert(self, in_size: int, out_size: int) -> nn.Module:
        if self.expert_hidden_layers == 0:
            return nn.Linear(in_size, out_size)
        return MLP(in_size, self.expert_hidden_size, out_size,
                   n_layers=self.expert_hidden_layers)

    def _build_model(self) -> BayesianNetwork:
        input_var, latent_var, input_cpd, latent_cpd = self._input_latent_block()

        ## Variables
        self._concepts = self.build_concept_variables(
            self.intermediate_concept_names, plate_name="concepts"
        )
        self._tasks = self.build_concept_variables(self.task_names, plate_name="tasks")

        # NOTE: one expert index per sample; the engine samples it.
        # Contrary to the original implementation, we do not select an expert for each
        # target variable. The two versions are equivalent dispite the fact that selecting
        # one expert per target variable correspond to model with a larger memory (as in CMR).
        selector = EmbeddingVariable(
            "selector",
            distribution=RelaxedOneHotCategoricalStraightThrough,
            size=self.n_experts,
        )

        ## CPDs

        # Concept encoder
        encoders = ParametricCPD(
            variable=self._concepts,
            parents=[latent_var],
            parametrization=[
                self._flexible_parametrization(c, first=LazyConstructor(LinearEmbeddingToConcept))
                for c in self._concepts
            ],
        )

        # Selector
        selector_net = (
            nn.Linear(self.latent_size, self.n_experts)
            if self.selector_hidden_layers == 0
            else MLP(self.latent_size, self.latent_size, self.n_experts,
                     n_layers=self.selector_hidden_layers)
        )
        selector_cpd = ParametricCPD(
            selector, parents=[latent_var], parametrization={"logits": selector_net}
        )

        # Predictor
        n_in = sum(c.size for c in self._concepts)
        predictors = []
        for t in self._tasks:
            param = "value" if t.distribution is Delta else self.param_for_discrete_var
            experts = [self._expert(n_in, t.param_sizes[param]) for _ in range(self.n_experts)]
            predictors.append(ParametricCPD(
                t,
                parents=[*self._concepts, selector],
                parametrization={param: MixtureOfExpertsConceptToConcept(experts)},
            ))

        return BayesianNetwork(
            variables=[input_var, latent_var, *self._concepts, selector, *self._tasks],
            factors=[input_cpd, latent_cpd, *encoders, selector_cpd, *predictors],
        )

    @property
    def expert_input_names(self) -> List[str]:
        """Symbol name of each expert input column (one per concept state)."""
        names = []
        for c in self._concepts:
            for m in c.members:
                card = self.concept_annotations.concept(m).cardinality
                states = [m] if card == 1 else [f"{m}_{s}" for s in range(card)]
                # NOTE: sanitized, as PySR needs identifier-like variable names.
                names += [re.sub(r"\W", "_", s) for s in states]
        return names

    def experts(self) -> Dict[str, MixtureOfExpertsConceptToConcept]:
        """The expert mixture of each task variable, by name."""
        return {
            t.name: next(iter(self.pgm.factors[t.name].parametrization.values()))
            for t in self._tasks
        }

    @torch.no_grad()
    def distill_experts(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        target: str = "expert",
        sr_params: Optional[dict] = None,
        save_path: Optional[str] = None,
        max_samples: int = 2000,
    ) -> Dict[str, List[List[sympy.Expr]]]:
        """Stage 2: one symbolic expression per expert and task output.

        Each sample is routed to its most probable expert; SymTorch then fits
        every expert on its own samples. The expressions are returned, not
        installed in the model.

        Args:
            x: Inputs, ``(N, input_size)``.
            c: Ground-truth concepts and tasks, as in a training batch.
            target: ``'expert'`` distills the MLP outputs (paper);
                ``'ground_truth'`` fits the task labels directly.
            sr_params: PySR parameters (operators, ``niterations``,
                ``model_selection``, ...). The paper uses ``['+', '-', '*']``
                for classification and adds unary operators for regression.
            save_path: Directory for the PySR outputs.
            max_samples: Samples per expert kept for the regression.

        Returns:
            ``{task: [expressions of expert m for each output]}``.
        """
        from symtorch import SymbolicModel  # NOTE: optional (pip install torch-symbolic)

        # NOTE: multi-threaded PySR can deadlock while Julia compiles (seen with
        # torch loaded); serial search is the safe default.
        sr_params = {"parallelism": "serial", **(sr_params or {})}
        if target not in ("expert", "ground_truth"):
            raise ValueError("target must be 'expert' or 'ground_truth'.")
        was_training = self.training
        self.eval()
        route = self(["selector"], input=x).logits["selector"].argmax(-1)
        self.train(was_training)

        # NOTE: ground-truth concepts as SR inputs ("disjoint", as in the paper's code).
        values = self.fully_observed_query(c)
        concepts = torch.cat([values[v.name] for v in self._concepts], dim=-1)
        names = self.expert_input_names
        out = {}
        for t_name, mixture in self.experts().items():
            out[t_name] = []
            for m, expert in enumerate(mixture.experts):
                idx = (route == m).nonzero().flatten()
                idx = idx[torch.randperm(len(idx))[:max_samples]]
                out_dim = expert(concepts[:1]).shape[-1]
                if len(idx) == 0:
                    # NOTE: unused expert -> trainable constant 0.
                    out[t_name].append([sympy.Float(0.0)] * out_dim)
                    continue
                block = expert
                if target == "ground_truth":
                    labels = values[t_name][idx]
                    block = lambda _x, _y=labels: _y  
                regressors = SymbolicModel(block, block_name=f"{t_name}_expert_{m}").distill(
                    concepts[idx], sr_params=sr_params, save_path=save_path,
                    fit_params={"variable_names": names},
                )
                out[t_name].append([regressors[d].sympy() for d in range(out_dim)])
        return out

    def set_symbolic_experts(self, expressions: Dict[str, List[List]]) -> None:
        """Stage 3: replace the experts with the given expressions.

        The constants of the expressions become trainable parameters.
        """
        for t_name, mixture in self.experts().items():
            device = next(self.parameters()).device
            for m, exprs in enumerate(expressions[t_name]):
                mixture.experts[m] = SymbolicConceptToConcept(
                    exprs, self.expert_input_names
                ).to(device)

    def _keep_selector_value(self, module, args, kwargs) -> None:
        self._selector_value = kwargs["embeddings"]

    def selector_marginal_entropy(self) -> torch.Tensor:
        """Entropy of the batch average of the selector value the predictors saw."""
        p = self._selector_value.reshape(-1, self.n_experts).mean(0)
        return -(p * p.clamp_min(1e-12).log()).sum()

    def shared_step(self, batch, step):
        """The learner's step, minus the selector entropy term in training."""
        loss = super().shared_step(batch, step)
        if self.selector_entropy_weight > 0 and step == "train" and loss is not None:
            entropy = self.selector_marginal_entropy()
            self.log("train_selector_entropy", entropy.detach(), on_step=False,
                     on_epoch=True, batch_size=batch["inputs"]["x"].size(0))
            loss = loss - self.selector_entropy_weight * entropy
        return loss

    @property
    def is_symbolic(self) -> bool:
        """Whether the experts have been replaced by expressions."""
        return any(isinstance(e, SymbolicConceptToConcept)
                   for mixture in self.experts().values() for e in mixture.experts)

    def on_train_epoch_start(self) -> None:
        """Run stage 2 at epoch ``mlp_epochs``, then fine-tune (stage 3)."""
        parent_hook = getattr(super(), "on_train_epoch_start", None)
        if parent_hook is not None:
            parent_hook()
        if self.mlp_epochs is None or self.is_symbolic:
            return
        if self.current_epoch < self.mlp_epochs:
            # NOTE: stage 1 always lasts mlp_epochs; early stopping acts in stage 3.
            for cb in self.trainer.callbacks:
                if isinstance(cb, EarlyStopping):
                    cb.wait_count = 0
            return
        self._start_symbolic_stage()

    def _start_symbolic_stage(self) -> None:
        """Stage 2: distill the experts, then stage 3: fine-tune the expressions."""

        # Collect of the samples from the training set.
        xs, cs = [], []
        for batch in self.trainer.train_dataloader:
            inputs, concepts, transforms = self.unpack_batch(batch)
            xs.append(self.maybe_scale_inputs(inputs, transforms)["x"])
            c = self.maybe_scale_concepts(concepts, transforms)["c"]
            cs.append(c[self.concept_annotations].tensor)
        # Distill the MLPs into symbolic expressions.
        self.set_symbolic_experts(self.distill_experts(
            torch.cat(xs).to(self.device), torch.cat(cs).to(self.device),
            **self.distill_kwargs,
        ))

        # Update optimizer learning rate for stage 3: fine-tuning the constants 
        # of the symbolic expressions. 
        if (self.optim_kwargs or {}).get("lr") is not None:
            self.optim_kwargs = {**self.optim_kwargs,
                                 "lr": self.optim_kwargs["lr"] * self.finetune_lr_factor}
        self.trainer.strategy.setup_optimizers(self.trainer)

        # Best checkpoint and early stopping restart, so that they
        # refer to the symbolic model.
        for cb in self.trainer.callbacks:
            inf = torch.tensor(torch.inf if getattr(cb, "mode", "min") == "min" else -torch.inf)
            if isinstance(cb, ModelCheckpoint):
                cb.best_k_models, cb.best_model_path, cb.best_model_score = {}, "", None
                cb.kth_best_model_path, cb.kth_value = "", inf
            elif isinstance(cb, EarlyStopping):
                cb.wait_count, cb.best_score = 0, inf
