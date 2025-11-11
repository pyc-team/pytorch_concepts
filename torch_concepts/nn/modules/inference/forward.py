import inspect
from abc import abstractmethod

import torch
from torch.distributions import RelaxedBernoulli, Bernoulli, RelaxedOneHotCategorical

from torch_concepts import ConceptGraph, Variable
from torch_concepts.nn import BaseModel, BaseGraphLearner
from typing import List, Tuple, Dict, Union

from ..models.pgm import ProbabilisticGraphicalModel
from ...base.inference import BaseInference


class ForwardInference(BaseInference):
    def __init__(self, pgm: ProbabilisticGraphicalModel, graph_learner: BaseGraphLearner = None, *args, **kwargs):
        super().__init__()
        self.pgm = pgm
        self.graph_learner = graph_learner
        self.concept_map = {var.concepts[0]: var for var in pgm.variables}
        self.sorted_variables = self._topological_sort()

        if graph_learner is not None:
            self.row_labels2id = {var: idx for idx, var in enumerate(self.graph_learner.row_labels)}
            self.col_labels2id = {var: idx for idx, var in enumerate(self.graph_learner.col_labels)}

        if len(self.sorted_variables) != len(self.pgm.variables):
            raise RuntimeError("The PGM contains cycles and cannot be processed in topological order.")

    @abstractmethod
    def get_results(self, results: torch.tensor, parent_variable: Variable):
        pass

    def _topological_sort(self) -> List[Variable]:
        """
        Sorts the variables topologically (parents before children).
        """
        in_degree = {var.concepts[0]: 0 for var in self.pgm.variables}
        adj = {var.concepts[0]: [] for var in self.pgm.variables}

        for var in self.pgm.variables:
            child_name = var.concepts[0]
            for parent_var in var.parents:
                parent_name = parent_var.concepts[0]
                adj[parent_name].append(child_name)
                in_degree[child_name] += 1

        # Start with nodes having zero incoming edges (root nodes)
        queue = [self.concept_map[name] for name, degree in in_degree.items() if degree == 0]
        sorted_variables = []

        while queue:
            var = queue.pop(0)
            sorted_variables.append(var)

            for neighbor_name in adj[var.concepts[0]]:
                in_degree[neighbor_name] -= 1
                if in_degree[neighbor_name] == 0:
                    queue.append(self.concept_map[neighbor_name])

        return sorted_variables

    def predict(self, external_inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Performs a forward pass prediction across the entire PGM using the topological order.

        Args:
            external_inputs: A dictionary of {root_concept_name: input_tensor} for the root variables.
                           E.g., {'emb': torch.randn(87, 10)}.

        Returns:
            A dictionary of {concept_name: predicted_feature_tensor} for all concepts.
        """

        results = {}

        # Iterate in topological order
        for var in self.sorted_variables:
            concept_name = var.concepts[0]
            factor = self.pgm.get_factor_of_variable(concept_name)

            if factor is None:
                raise RuntimeError(f"Missing factor for variable/concept: {concept_name}")

            # 1. Handle Root Nodes (no parents)
            if not var.parents:
                if concept_name not in external_inputs:
                    raise ValueError(
                        f"Root variable '{concept_name}' requires an external input tensor in the 'external_inputs' dictionary.")

                input_tensor = external_inputs[concept_name]

                parent_kwargs = self.get_parent_kwargs(factor, [input_tensor], [])
                output_tensor = factor.forward(**parent_kwargs)
                output_tensor = self.get_results(output_tensor, var)

                # 2. Handle Child Nodes (has parents)
            else:
                parent_logits = []
                parent_latent = []
                for parent_var in var.parents:
                    parent_name = parent_var.concepts[0]
                    if parent_name not in results:
                        # Should not happen with correct topological sort
                        raise RuntimeError(
                            f"Parent data missing: Cannot compute {concept_name} because parent {parent_name} has not been computed yet.")

                    # Parent tensor is fed into the factor using the parent's concept name as the key
                    if parent_var.distribution in [Bernoulli, RelaxedBernoulli, RelaxedOneHotCategorical]:
                        # For probabilistic parents, pass logits
                        weight = 1
                        if self.graph_learner is not None:
                            weight = self.graph_learner.weighted_adj[self.row_labels2id[parent_name], self.col_labels2id[concept_name]]

                        parent_logits.append(results[parent_name] * weight)
                    else:
                        # For continuous parents, pass latent features
                        parent_latent.append(results[parent_name])

                parent_kwargs = self.get_parent_kwargs(factor, parent_latent, parent_logits)
                output_tensor = factor.forward(**parent_kwargs)
                output_tensor = self.get_results(output_tensor, var)

            results[concept_name] = output_tensor

        return results

    def get_parent_kwargs(self, factor,
                          parent_latent: Union[List[torch.Tensor], torch.Tensor] = None,
                          parent_logits: Union[List[torch.Tensor], torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        parent_kwargs = {}
        sig = inspect.signature(factor.module_class.forward)
        params = sig.parameters
        allowed = {
            name for name, p in params.items()
            if name != "self" and p.kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
        }
        if allowed not in [{'logits'}, {'logits', 'embedding'}, {'logits', 'exogenous'}, {'embedding'}, {'exogenous'}]:
            # this is a standard torch layer: concatenate all inputs into 'x'
            parent_kwargs[allowed.pop()] = torch.cat(parent_logits + parent_latent, dim=-1)
        else:
            # this is a PyC layer: separate logits and latent inputs
            if 'logits' in allowed:
                parent_kwargs['logits'] = torch.cat(parent_logits, dim=-1)
            if 'embedding' in allowed:
                parent_kwargs['embedding'] = torch.cat(parent_latent, dim=-1)
            elif 'exogenous' in allowed:
                parent_kwargs['exogenous'] = torch.cat(parent_latent, dim=1)

        return parent_kwargs

    def query(self, query_concepts: List[str], evidence: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Executes a forward pass and returns only the specified concepts concatenated
        into a single tensor, in the order requested.

        Args:
            query_concepts: A list of concept names to retrieve, e.g., ["c2", "c1", "xor_class"].
            evidence: A dictionary of {root_concept_name: input_tensor} for the root variables.

        Returns:
            A single torch.Tensor containing the concatenated predictions for the
            requested concepts, ordered as requested (Batch x TotalFeatures).
        """
        # 1. Run the full forward pass to get all necessary predictions
        all_predictions = self.predict(evidence)

        # 2. Filter and concatenate results
        result_tensors = []

        for concept_name in query_concepts:
            if concept_name not in all_predictions:
                raise ValueError(
                    f"Query concept '{concept_name}' was requested but could not be computed. "
                    f"Available predictions: {list(all_predictions.keys())}"
                )
            result_tensors.append(all_predictions[concept_name])

        if not result_tensors:
            return torch.empty(0)  # Return empty tensor if query list was empty

        # 3. Concatenate tensors along the last dimension (features)
        # Check if batch sizes match before concatenation
        batch_size = result_tensors[0].shape[0]
        if any(t.shape[0] != batch_size for t in result_tensors):
            raise RuntimeError("Batch size mismatch detected in query results before concatenation.")

        # Concatenate results into the final output tensor (Batch x TotalFeatures)
        final_tensor = torch.cat(result_tensors, dim=-1)

        # 4. Perform final check for expected shape
        expected_feature_dim = sum(self.concept_map[c].out_features for c in query_concepts)
        if final_tensor.shape[1] != expected_feature_dim:
            raise RuntimeError(
                f"Concatenation error. Expected total feature dimension of {expected_feature_dim}, "
                f"but got {final_tensor.shape[1]}. Check Variable.out_features logic."
            )

        return final_tensor


class DeterministicInference(ForwardInference):
    def get_results(self, results: torch.tensor, parent_variable: Variable) -> torch.Tensor:
        return results


class AncestralSamplingInference(ForwardInference):
    def __init__(self, pgm: ProbabilisticGraphicalModel, graph_learner: BaseGraphLearner = None, temperature: float = 1.):
        super().__init__(pgm, graph_learner)
        self.temperature = temperature

    def get_results(self, results: torch.tensor, parent_variable: Variable) -> torch.Tensor:
        if parent_variable.distribution in [Bernoulli]:
            return parent_variable.distribution(logits=results).sample()
        elif parent_variable.distribution in [RelaxedBernoulli, RelaxedOneHotCategorical]:
            return parent_variable.distribution(logits=results, temperature=self.temperature).rsample()
        return parent_variable.distribution(results).rsample()
