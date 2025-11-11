from copy import deepcopy

import torch
from networkx.readwrite.json_graph.node_link import node_link_graph
from sklearn.metrics import accuracy_score
from torch.distributions import Bernoulli, Categorical, OneHotCategorical, RelaxedOneHotCategorical, RelaxedBernoulli
from twine import metadata

from torch_concepts import Annotations, AxisAnnotation, Variable, ConceptGraph
from torch_concepts.data import ToyDataset
from torch_concepts.distributions import Delta
from torch_concepts.nn import ProbEncoderFromEmb, ProbPredictor, Factor, ProbabilisticGraphicalModel, ForwardInference, \
    RandomPolicy, DoIntervention, intervention, DeterministicInference, BipartiteModel, Propagator, \
    MixProbExogPredictor, ExogEncoder, ProbEncoderFromExog, GroundTruthIntervention, UniformPolicy, \
    HyperLinearPredictor, GraphModel, AncestralSamplingInference, COSMOGraphLearner


def main():
    latent_dims = 20
    n_epochs = 1000
    n_samples = 1000
    concept_reg = 0.5
    data = ToyDataset('xor', size=n_samples, random_state=42)
    x_train, c_train, y_train, concept_names, task_names = data.data, data.concept_labels, data.target_labels, data.concept_attr_names, data.task_attr_names
    c_train = torch.cat([c_train, y_train], dim=1)
    y_train = deepcopy(c_train)
    cy_train = torch.cat([c_train, y_train], dim=1)
    c_train_one_hot = torch.cat([cy_train[:, :2], torch.nn.functional.one_hot(cy_train[:, 2].long(), num_classes=2).float()], dim=1)
    cy_train_one_hot = torch.cat([c_train_one_hot, c_train_one_hot], dim=1)

    concept_names = ('c1', 'c2', 'xor')
    task_names = ('copy_c1', 'copy_c2', 'copy_xor')
    cardinalities = (1, 1, 2, 1, 1, 2)
    metadata = {
        'c1': {'distribution': RelaxedBernoulli, 'type': 'binary', 'description': 'Concept 1'},
        'c2': {'distribution': RelaxedBernoulli, 'type': 'binary', 'description': 'Concept 2'},
        'xor': {'distribution': RelaxedOneHotCategorical, 'type': 'categorical', 'description': 'XOR Task'},
        'copy_c1': {'distribution': RelaxedBernoulli, 'type': 'binary', 'description': 'Concept 1 Copy'},
        'copy_c2': {'distribution': RelaxedBernoulli, 'type': 'binary', 'description': 'Concept 2 Copy'},
        'copy_xor': {'distribution': RelaxedOneHotCategorical, 'type': 'categorical', 'description': 'XOR Task Copy'},
    }
    annotations = Annotations({1: AxisAnnotation(concept_names + task_names, cardinalities=cardinalities, metadata=metadata)})

    model_graph = ConceptGraph(torch.tensor([[0, 0, 0, 0, 1, 1],
                                             [0, 0, 0, 1, 0, 1],
                                             [0, 0, 0, 1, 1, 0],
                                             [0, 0, 0, 0, 0, 0],
                                             [0, 0, 0, 0, 0, 0],
                                             [0, 0, 0, 0, 0, 0]]), list(annotations.get_axis_annotation(1).labels))

    # PGM Initialization
    encoder = torch.nn.Sequential(torch.nn.Linear(x_train.shape[1], latent_dims), torch.nn.LeakyReLU())
    concept_model = GraphModel(model_graph=model_graph,
                                   input_size=latent_dims,
                                   annotations=annotations,
                                   source_exogenous=Propagator(ExogEncoder, embedding_size=12),
                                   internal_exogenous=Propagator(ExogEncoder, embedding_size=13),
                                   encoder=Propagator(ProbEncoderFromExog),
                                   predictor=Propagator(HyperLinearPredictor, embedding_size=11))

    # graph learning init
    graph_learner = COSMOGraphLearner(concept_names, task_names, hard_threshold=False, temperature=0.01)

    # Inference Initialization
    inference_engine = DeterministicInference(concept_model.pgm, graph_learner)
    query_concepts = ["c1", "c2", "xor", "copy_c1", "copy_c2", "copy_xor"]

    model = torch.nn.Sequential(encoder, concept_model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    model.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()

        # generate concept and task predictions
        emb = encoder(x_train)
        cy_pred = inference_engine.query(query_concepts, evidence={'embedding': emb})
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

        # if epoch > 500:
        #     graph_learner.hard_threshold = True

    with torch.no_grad():
        # graph_learner.hard_threshold = True
        print(graph_learner.weighted_adj)

        print("=== Interventions ===")
        int_policy_c1 = UniformPolicy(out_annotations=Annotations({1: AxisAnnotation(["c1"])}), subset=["c1"])
        int_strategy_c1 = DoIntervention(model=concept_model.pgm.factor_modules, constants=-10)
        with intervention(policies=[int_policy_c1],
                          strategies=[int_strategy_c1],
                          on_layers=["c1.encoder"],
                          quantiles=[1]):
            cy_pred = inference_engine.query(query_concepts, evidence={'embedding': emb})
            c_pred = cy_pred[:, :cy_train_one_hot.shape[1]//2]
            y_pred = cy_pred[:, cy_train_one_hot.shape[1]//2:]
            task_accuracy = accuracy_score(c_train_one_hot.ravel(), y_pred.ravel() > 0.)
            concept_accuracy = accuracy_score(c_train_one_hot.ravel(), c_pred.ravel() > 0.)
            print(f"Do intervention on c1 | Task Acc: {task_accuracy:.2f} | Concept Acc: {concept_accuracy:.2f}")
            print(cy_pred[:5])
            print()

            int_policy_c1 = UniformPolicy(out_annotations=Annotations({1: AxisAnnotation(["c1"])}), subset=["c1"])
            int_strategy_c1 = GroundTruthIntervention(model=concept_model.pgm.factor_modules, ground_truth=torch.logit(c_train[:, 0:1], eps=1e-6))
            with intervention(policies=[int_policy_c1],
                              strategies=[int_strategy_c1],
                              on_layers=["c1.encoder"],
                              quantiles=[1]):
                cy_pred = inference_engine.query(query_concepts, evidence={'embedding': emb})
                c_pred = cy_pred[:, :cy_train_one_hot.shape[1]//2]
                y_pred = cy_pred[:, cy_train_one_hot.shape[1]//2:]
                task_accuracy = accuracy_score(c_train_one_hot.ravel(), y_pred.ravel() > 0.)
                concept_accuracy = accuracy_score(c_train_one_hot.ravel(), c_pred.ravel() > 0.)
                print(f"Ground truth intervention on c1 | Task Acc: {task_accuracy:.2f} | Concept Acc: {concept_accuracy:.2f}")
                print(cy_pred[:5])

    return


if __name__ == "__main__":
    main()
