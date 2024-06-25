import torch
from collections import Counter

from .semantics import Logic, GodelTNorm


class BaseReasoner(torch.nn.Module):
    """
    BaseReasoner is an abstract base class for models that use concept to predict downstream tasks.

    Attributes:
        n_concepts (int): Number of concepts.
        n_classes (int): Number of classes.
    """
    def __init__(self, n_concepts, n_classes):
        super().__init__()
        self.n_concepts = n_concepts
        self.n_classes = n_classes

    def forward(self, x):
        raise NotImplementedError

    def __call__(self, x):
        return self.forward(x)


class DeepConceptReasoner(BaseReasoner):
    """
    DeepConceptReasoner is a reasoner that first learns a logic formula and then executes the formula using given concept activations to predict the downstream task.
    Main reference: `"Interpretable Neural-Symbolic Concept Reasoning" <https://arxiv.org/abs/2304.14068>`_

    Attributes:
        n_concepts (int): Number of concepts.
        n_classes (int): Number of classes.
        emb_size (int): Embedding size.
        logic (Logic): T-Norm to be used to for the logic semantics.
        temperature (float): Temperature to be used to compute the attention scores.
        set_level_rules (bool): Whether to learn set-level rules or instance-level rules.
    """
    def __init__(self, n_concepts, n_classes, emb_size, logic: Logic = GodelTNorm(), temperature: float = 1., set_level_rules: bool = False):
        super().__init__(n_concepts, n_classes)
        self.emb_size = emb_size
        self.logic = logic
        self.set_level_rules = set_level_rules
        self.emb_size_after_pool = emb_size
        if self.set_level_rules:
            self.emb_size_after_pool = emb_size * 4
        self.filter_nn = torch.nn.Sequential(
            torch.nn.Linear(self.emb_size_after_pool, emb_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(emb_size, n_concepts * n_classes),
        )
        self.sign_nn = torch.nn.Sequential(
            torch.nn.Linear(self.emb_size_after_pool, emb_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(emb_size, n_concepts * n_classes),
        )
        self.filter_nn_before_pool = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(emb_size, emb_size),
        )
        self.sign_nn_before_pool = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(emb_size, emb_size),
        )
        self.temperature = temperature

    def _softselect(self, values, temperature):
        softmax_scores = torch.log_softmax(values, dim=1)
        softscores = torch.sigmoid(softmax_scores - temperature * softmax_scores.mean(dim=1, keepdim=True))
        return softscores

    def forward(self, x, c, return_attn=False, sign_attn=None, filter_attn=None):
        values = c.unsqueeze(-1).repeat(1, 1, self.n_classes)

        sign_emb = filter_emb = x
        if sign_attn is None:
            if self.set_level_rules:
                sign_emb = self.sign_nn_before_pool(x)
                sign_emb = torch.concat([
                    torch.sum(sign_emb, dim=0, keepdim=True),
                    torch.mean(sign_emb, dim=0, keepdim=True),
                    torch.std(sign_emb, dim=0, keepdim=True),
                    torch.max(sign_emb, dim=0, keepdim=True)[0],
                ], dim=1)
                filter_emb = self.filter_nn_before_pool(x)
                filter_emb = torch.concat([
                    torch.sum(filter_emb, dim=0, keepdim=True),
                    torch.mean(filter_emb, dim=0, keepdim=True),
                    torch.std(filter_emb, dim=0, keepdim=True),
                    torch.max(filter_emb, dim=0, keepdim=True)[0],
                ], dim=1)

            # compute attention scores to build logic sentence
            # each attention score will represent whether the concept should be active or not in the logic sentence
            sign_attn = torch.sigmoid(self.sign_nn(sign_emb))
            sign_attn = sign_attn.view(sign_attn.shape[0], self.n_concepts, self.n_classes)

            if self.set_level_rules:
                sign_attn = sign_attn.expand(len(values), -1, -1)

        # attention scores need to be aligned with predicted concept truth values (attn <-> values)
        # (not A or V) and (A or not V) <-> (A <-> V)
        sign_terms = self.logic.iff_pair(sign_attn, values)

        if filter_attn is None:
            # compute attention scores to identify only relevant concepts for each class
            filtr = self.filter_nn(filter_emb)
            filtr = filtr.view(filtr.shape[0], self.n_concepts, self.n_classes)
            filter_attn = self._softselect(filtr, self.temperature)

            if self.set_level_rules:
                filter_attn = sign_attn.expand(len(sign_terms), -1, -1)

        # filter value
        # filtered implemented as "or(a, not b)", corresponding to "b -> a"
        filtered_values = self.logic.disj_pair(sign_terms, self.logic.neg(filter_attn))

        # generate minterm
        preds = self.logic.conj(filtered_values, dim=1).squeeze(1).float()

        if return_attn:
            return preds, sign_attn, filter_attn
        else:
            return preds

    def explain(self, x, c, mode, concept_names=None, class_names=None, filter_attn=None, sign_attn=None):
        assert mode in ['local', 'global', 'exact']

        if concept_names is None:
            concept_names = [f'c_{i}' for i in range(c.shape[1])]
        if class_names is None:
            class_names = [f'y_{i}' for i in range(self.n_classes)]

        # make a forward pass to get predictions and attention weights
        y_preds, sign_attn_mask, filter_attn_mask = self.forward(x, c, return_attn=True, filter_attn=filter_attn, sign_attn=sign_attn)

        explanations = []
        all_class_explanations = {cn: [] for cn in class_names}
        for sample_idx in range(len(x)):
            prediction = y_preds[sample_idx] > 0.5
            active_classes = torch.argwhere(prediction).ravel()

            if len(active_classes) == 0:
                # if no class is active for this sample, then we cannot extract any explanation
                explanations.append({
                    'class': -1,
                    'explanation': '',
                    'attention': [],
                })
            else:
                # else we can extract an explanation for each active class!
                for target_class in active_classes:
                    attentions = []
                    minterm = []
                    for concept_idx in range(len(concept_names)):
                        c_pred = c[sample_idx, concept_idx]
                        sign_attn = sign_attn_mask[sample_idx, concept_idx, target_class]
                        filter_attn = filter_attn_mask[sample_idx, concept_idx, target_class]

                        # we first check if the concept was relevant
                        # a concept is relevant <-> the filter attention score is lower than the concept probability
                        at_score = 0
                        sign_terms = self.logic.iff_pair(sign_attn, c_pred).item()
                        if self.logic.neg(filter_attn) < sign_terms:
                            if sign_attn >= 0.5:
                                # if the concept is relevant and the sign is positive we just take its attention score
                                at_score = filter_attn.item()
                                if mode == 'exact':
                                    minterm.append(f'{sign_terms:.3f} ({concept_names[concept_idx]})')
                                else:
                                    minterm.append(f'{concept_names[concept_idx]}')
                            else:
                                # if the concept is relevant and the sign is positive we take (-1) * its attention score
                                at_score = -filter_attn.item()
                                if mode == 'exact':
                                    minterm.append(f'{sign_terms:.3f} (~{concept_names[concept_idx]})')
                                else:
                                    minterm.append(f'~{concept_names[concept_idx]}')
                        attentions.append(at_score)

                    # add explanation to list
                    target_class_name = class_names[target_class]
                    minterm = ' & '.join(minterm)
                    all_class_explanations[target_class_name].append(minterm)
                    explanations.append({
                        'sample-id': sample_idx,
                        'class': target_class_name,
                        'explanation': minterm,
                        'attention': attentions,
                    })

        if mode == 'global':
            # count most frequent explanations for each class
            explanations = []
            for class_id, class_explanations in all_class_explanations.items():
                explanation_count = Counter(class_explanations)
                for explanation, count in explanation_count.items():
                    explanations.append({
                        'class': class_id,
                        'explanation': explanation,
                        'count': count,
                    })

        return explanations

    def __call__(self, x, c, return_attn=False, sign_attn=None, filter_attn=None):
        return self.forward(x, c, return_attn, sign_attn, filter_attn)
