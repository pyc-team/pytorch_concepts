import torch
from copy import deepcopy
from sklearn.metrics import accuracy_score
from torch.distributions import RelaxedOneHotCategorical, RelaxedBernoulli

from torch_concepts import Annotations, AxisAnnotation, ConceptGraph
from torch_concepts.data.datasets import ToyDataset
from torch_concepts.nn import DoIntervention, intervention, DeterministicInference, LazyConstructor, \
    LinearZU, LinearUC, GroundTruthIntervention, UniformPolicy, \
    HyperLinearCUC, GraphModel, WANDAGraphLearner


def main():
    latent_dims = 20
    n_epochs = 1000
    n_samples = 1000
    concept_reg = 0.5

    dataset = ToyDataset(dataset='xor', seed=42, n_gen=n_samples)
    x_train = dataset.input_data
    concept_idx = list(dataset.graph.edge_index[0].unique().numpy())
    task_idx = list(dataset.graph.edge_index[1].unique().numpy())
    c_train = dataset.concepts[:, concept_idx]
    y_train = dataset.concepts[:, task_idx]

    c_train = torch.cat([c_train, y_train], dim=1)
    y_train = deepcopy(c_train)
    cy_train = torch.cat([c_train, y_train], dim=1)
    c_train_one_hot = torch.cat([cy_train[:, :2], torch.nn.functional.one_hot(cy_train[:, 2].long(), num_classes=2).float()], dim=1)
    cy_train_one_hot = torch.cat([c_train_one_hot, c_train_one_hot], dim=1)

    concept_names = ['c1', 'c2', 'xor']
    task_names = ['c1_copy', 'c2_copy', 'xor_copy']
    cardinalities = [1, 1, 2, 1, 1, 2]
    metadata = {
        'c1': {'distribution': RelaxedBernoulli, 'type': 'binary', 'description': 'Concept 1'},
        'c2': {'distribution': RelaxedBernoulli, 'type': 'binary', 'description': 'Concept 2'},
        'xor': {'distribution': RelaxedOneHotCategorical, 'type': 'categorical', 'description': 'XOR Task'},
        'c1_copy': {'distribution': RelaxedBernoulli, 'type': 'binary', 'description': 'Concept 1 Copy'},
        'c2_copy': {'distribution': RelaxedBernoulli, 'type': 'binary', 'description': 'Concept 2 Copy'},
        'xor_copy': {'distribution': RelaxedOneHotCategorical, 'type': 'categorical', 'description': 'XOR Task Copy'},
    }
    annotations = Annotations({1: AxisAnnotation(concept_names + task_names, cardinalities=cardinalities, metadata=metadata)})

    model_graph = ConceptGraph(torch.tensor([[0, 0, 0, 0, 1, 1],
                                             [0, 0, 0, 1, 0, 1],
                                             [0, 0, 0, 1, 1, 0],
                                             [0, 0, 0, 0, 0, 0],
                                             [0, 0, 0, 0, 0, 0],
                                             [0, 0, 0, 0, 0, 0]]), list(annotations.get_axis_annotation(1).labels))

    # ProbabilisticModel Initialization
    encoder = torch.nn.Sequential(torch.nn.Linear(x_train.shape[1], latent_dims), torch.nn.LeakyReLU())
    concept_model = GraphModel(model_graph=model_graph,
                                   input_size=latent_dims,
                                   annotations=annotations,
                                   source_exogenous=LazyConstructor(LinearZU, exogenous_size=11),
                                   internal_exogenous=LazyConstructor(LinearZU, exogenous_size=7),
                                   encoder=LazyConstructor(LinearUC),
                                   predictor=LazyConstructor(HyperLinearCUC, embedding_size=20))

    # graph learning init
    graph_learner = WANDAGraphLearner(concept_names, task_names)

    # Inference Initialization
    inference_engine = DeterministicInference(concept_model.probabilistic_model, graph_learner)
    query_concepts = ["c1", "c2", "xor", "c1_copy", "c2_copy", "xor_copy"]

    model = torch.nn.Sequential(encoder, concept_model, graph_learner)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    model.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()

        # generate concept and task predictions
        emb = encoder(x_train)
        cy_pred = inference_engine.query(query_concepts, evidence={'input': emb}, debug=True)
        c_pred = cy_pred[:, :cy_train_one_hot.shape[1]//2]
        y_pred = cy_pred[:, cy_train_one_hot.shape[1]//2:]

        # compute loss
        concept_loss = loss_fn(c_pred, c_train_one_hot)
        task_loss = loss_fn(y_pred, c_train_one_hot)
        loss = concept_loss + concept_reg * task_loss

        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            task_accuracy = accuracy_score(c_train_one_hot.ravel(), y_pred.ravel() > 0.)
            concept_accuracy = accuracy_score(c_train_one_hot.ravel(), c_pred.ravel() > 0.)
            print(f"Epoch {epoch}: Loss {loss.item():.2f} | Task Acc: {task_accuracy:.2f} | Concept Acc: {concept_accuracy:.2f}")

    with torch.no_grad():
        print("=== Learned Graph ===")
        print(graph_learner.weighted_adj)
        print()

        concept_model_new = inference_engine.unrolled_probabilistic_model()
        # identify available query concepts in the unrolled model
        query_concepts = [c for c in query_concepts if c in inference_engine.available_query_vars]
        concept_idx = {v: i for i, v in enumerate(concept_names)}
        reverse_c2t_mapping = dict(zip(task_names, concept_names))
        query_concepts = sorted(query_concepts, key=lambda x: concept_idx[x] if x in concept_idx else concept_idx[reverse_c2t_mapping[x]])

        inference_engine = DeterministicInference(concept_model_new)

        print("=== Unrolled Model Predictions ===")
        # generate concept and task predictions
        emb = encoder(x_train)
        cy_pred = inference_engine.query(query_concepts, evidence={'input': emb})
        task_accuracy = accuracy_score(c_train_one_hot.ravel(), cy_pred.ravel() > 0.)
        print(f"Unrolling accuracies | Task Acc: {task_accuracy:.2f}")


        print("=== Interventions ===")
        intervened_concept = query_concepts[0]

        int_policy_c1 = UniformPolicy(out_features=concept_model.probabilistic_model.concept_to_variable[intervened_concept].size)
        int_strategy_c1 = DoIntervention(model=concept_model_new.parametric_cpds, constants=-10)
        with intervention(policies=int_policy_c1,
                          strategies=int_strategy_c1,
                          target_concepts=[intervened_concept]):
            cy_pred = inference_engine.query(query_concepts, evidence={'input': emb})
            task_accuracy = accuracy_score(c_train_one_hot.ravel(), cy_pred.ravel() > 0.)
            print(f"Do intervention on {intervened_concept} | Task Acc: {task_accuracy:.2f}")
            print(cy_pred[:5])
            print()

            int_policy_c1 = UniformPolicy(out_features=concept_model.probabilistic_model.concept_to_variable[intervened_concept].size)
            int_strategy_c1 = GroundTruthIntervention(model=concept_model_new.parametric_cpds, ground_truth=torch.logit(c_train[:, 0:1], eps=1e-6))
            with intervention(policies=int_policy_c1,
                              strategies=int_strategy_c1,
                              target_concepts=[intervened_concept]):
                cy_pred = inference_engine.query(query_concepts, evidence={'input': emb})
                task_accuracy = accuracy_score(c_train_one_hot.ravel(), cy_pred.ravel() > 0.)
                print(f"Ground truth intervention on {intervened_concept} | Task Acc: {task_accuracy:.2f}")
                print(cy_pred[:5])

    return


if __name__ == "__main__":
    main()
