import torch
from sklearn.metrics import accuracy_score
from torch.distributions import RelaxedOneHotCategorical, RelaxedBernoulli

from torch_concepts import Annotations, AxisAnnotation
from torch_concepts.data.datasets import ToyDataset
from torch_concepts.nn import RandomPolicy, DoIntervention, intervention, DeterministicInference, BipartiteModel, \
    LazyConstructor, \
    LinearZU, LinearUC, GroundTruthIntervention, UniformPolicy, HyperLinearCUC, \
    AncestralSamplingInference


def main():
    latent_dims = 10
    n_epochs = 200
    n_samples = 1000
    concept_reg = 0.5

    dataset = ToyDataset(dataset='xor', seed=42, n_gen=n_samples)
    x_train = dataset.input_data
    concept_idx = list(dataset.graph.edge_index[0].unique().numpy())
    task_idx = list(dataset.graph.edge_index[1].unique().numpy())
    c_train = dataset.concepts[:, concept_idx]
    y_train = dataset.concepts[:, task_idx]
    concept_names = ['c1', 'c2']
    task_names = ['xor']

    y_train = torch.cat([y_train, 1-y_train], dim=1)
    cy_train = torch.cat([c_train, y_train], dim=1)

    cardinalities = [1, 1, 2]
    metadata = {
        'c1': {'distribution': RelaxedBernoulli, 'type': 'binary', 'description': 'Concept 1'},
        'c2': {'distribution': RelaxedBernoulli, 'type': 'binary', 'description': 'Concept 2'},
        'xor': {'distribution': RelaxedOneHotCategorical, 'type': 'binary', 'description': 'XOR Task'},
    }
    annotations = Annotations({1: AxisAnnotation(concept_names + task_names, cardinalities=cardinalities, metadata=metadata)})

    # ProbabilisticModel Initialization
    encoder = torch.nn.Sequential(torch.nn.Linear(x_train.shape[1], latent_dims), torch.nn.LeakyReLU())
    concept_model = BipartiteModel(task_names=list(task_names),
                                   input_size=latent_dims,
                                   annotations=annotations,
                                   source_exogenous=LazyConstructor(LinearZU, exogenous_size=12),
                                   internal_exogenous=LazyConstructor(LinearZU, exogenous_size=13),
                                   encoder=LazyConstructor(LinearUC),
                                   predictor=LazyConstructor(HyperLinearCUC, embedding_size=11))

    # Inference Initialization
    inference_engine = AncestralSamplingInference(concept_model.probabilistic_model, temperature=1.0)
    query_concepts = ["c1", "c2", "xor"]
    int_policy_c = RandomPolicy(out_features=concept_model.probabilistic_model.concept_to_variable["c1"].size, scale=100)
    int_strategy_c1 = GroundTruthIntervention(model=concept_model.probabilistic_model.parametric_cpds, ground_truth=c_train[:, 0:1])
    int_strategy_c2 = GroundTruthIntervention(model=concept_model.probabilistic_model.parametric_cpds, ground_truth=c_train[:, 1:2])

    model = torch.nn.Sequential(encoder, concept_model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    loss_fn = torch.nn.BCELoss()
    model.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()

        # generate concept and task predictions
        emb = encoder(x_train)
        cy_pred = inference_engine.query(query_concepts, evidence={'input': emb})
        c_pred = cy_pred[:, :c_train.shape[1]]
        y_pred = cy_pred[:, c_train.shape[1]:]

        with intervention(policies=[int_policy_c, int_policy_c],
                          strategies=[int_strategy_c1, int_strategy_c2],
                          target_concepts=["c1", "c2"]):
            cy_pred = inference_engine.query(query_concepts, evidence={'input': emb})
            c_pred_int = cy_pred[:, :c_train.shape[1]]
            y_pred_int = cy_pred[:, c_train.shape[1]:]

        # compute loss
        concept_loss = loss_fn(c_pred, c_train)
        task_loss = loss_fn(y_pred, y_train)
        concept_loss_int = loss_fn(c_pred_int, c_train)
        task_loss_int = loss_fn(y_pred_int, y_train)
        loss = concept_loss + concept_reg * task_loss + concept_loss_int + concept_reg * task_loss_int

        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            task_accuracy = accuracy_score(y_train, y_pred > 0.5)
            concept_accuracy = accuracy_score(c_train, c_pred > 0.5)
            print(f"Epoch {epoch}: Loss {loss.item():.2f} | Task Acc: {task_accuracy:.2f} | Concept Acc: {concept_accuracy:.2f}")

    print("=== No Intervention ===")
    print(cy_train[:5])

    print("=== Interventions ===")

    int_policy_random = UniformPolicy(out_features=concept_model.probabilistic_model.concept_to_variable["c1"].size)
    int_strategy_random = DoIntervention(model=concept_model.probabilistic_model.parametric_cpds, constants=0)
    with intervention(policies=int_policy_random,
                      strategies=int_strategy_random,
                      target_concepts=["c1", "c2"]):
        cy_pred = inference_engine.query(query_concepts, evidence={'input': emb})
        c_pred = cy_pred[:, :c_train.shape[1]]
        y_pred = cy_pred[:, c_train.shape[1]:]
        task_accuracy = accuracy_score(y_train, y_pred > 0.5)
        concept_accuracy = accuracy_score(c_train, c_pred > 0.5)
        print(f"Do intervention on c1 | Task Acc: {task_accuracy:.2f} | Concept Acc: {concept_accuracy:.2f}")
        print(cy_pred[:5])
        print()

        with intervention(policies=[int_policy_c, int_policy_c],
                          strategies=[int_strategy_c1, int_strategy_c2],
                          target_concepts=["c1", "c2"]):
            cy_pred = inference_engine.query(query_concepts, evidence={'input': emb})
            c_pred = cy_pred[:, :c_train.shape[1]]
            y_pred = cy_pred[:, c_train.shape[1]:]
            task_accuracy = accuracy_score(y_train, y_pred > 0.5)
            concept_accuracy = accuracy_score(c_train, c_pred > 0.5)
            print(f"Ground truth intervention on c1 | Task Acc: {task_accuracy:.2f} | Concept Acc: {concept_accuracy:.2f}")
            print(cy_pred[:5])

    return


if __name__ == "__main__":
    main()
