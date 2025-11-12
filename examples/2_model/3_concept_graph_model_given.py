import torch
from sklearn.metrics import accuracy_score
from torch.distributions import RelaxedBernoulli

from torch_concepts import Annotations, AxisAnnotation, ConceptGraph
from torch_concepts.data import ToyDataset
from torch_concepts.nn import RandomPolicy, DoIntervention, intervention, Propagator, \
    ExogEncoder, ProbEncoderFromExog, GroundTruthIntervention, UniformPolicy, \
    HyperLinearPredictor, GraphModel, AncestralSamplingInference


def main():
    latent_dims = 10
    n_epochs = 200
    n_samples = 1000
    concept_reg = 0.5
    data = ToyDataset('xor', size=n_samples, random_state=42)
    x_train, c_train, y_train, concept_names, task_names = data.data, data.concept_labels, data.target_labels, data.concept_attr_names, data.task_attr_names
    y_train2 =  1 - y_train

    concept_names = ('c1', 'c2')
    task_names = ('xor',)
    task_names2 = ('not_xor',)
    cardinalities = (1, 1, 1, 1)
    metadata = {
        'c1': {'distribution': RelaxedBernoulli, 'type': 'binary', 'description': 'Concept 1'},
        'c2': {'distribution': RelaxedBernoulli, 'type': 'binary', 'description': 'Concept 2'},
        'xor': {'distribution': RelaxedBernoulli, 'type': 'binary', 'description': 'XOR Task'},
        'not_xor': {'distribution': RelaxedBernoulli, 'type': 'binary', 'description': 'NOT XOR Task'},
    }
    annotations = Annotations({1: AxisAnnotation(concept_names + task_names + task_names2, cardinalities=cardinalities, metadata=metadata)})

    model_graph = ConceptGraph(torch.tensor([[0, 0, 1, 0],
                                             [0, 0, 1, 0],
                                             [0, 0, 0, 1],
                                             [0, 0, 0, 0]]), list(annotations.get_axis_annotation(1).labels))

    # PGM Initialization
    encoder = torch.nn.Sequential(torch.nn.Linear(x_train.shape[1], latent_dims), torch.nn.LeakyReLU())
    concept_model = GraphModel(model_graph=model_graph,
                                   input_size=latent_dims,
                                   annotations=annotations,
                                   source_exogenous=Propagator(ExogEncoder, embedding_size=12),
                                   internal_exogenous=Propagator(ExogEncoder, embedding_size=13),
                                   encoder=Propagator(ProbEncoderFromExog),
                                   predictor=Propagator(HyperLinearPredictor, embedding_size=11))

    # Inference Initialization
    inference_engine = AncestralSamplingInference(concept_model.pgm, temperature=1.)
    query_concepts = ["c1", "c2", "xor", "not_xor"]

    model = torch.nn.Sequential(encoder, concept_model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    loss_fn = torch.nn.BCELoss()
    model.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()

        # generate concept and task predictions
        emb = encoder(x_train)
        cy_pred = inference_engine.query(query_concepts, evidence={'embedding': emb})
        c_pred = cy_pred[:, :c_train.shape[1]]
        y_pred = cy_pred[:, c_train.shape[1]:c_train.shape[1]+1]
        y2_pred = cy_pred[:, c_train.shape[1]+1:]

        # compute loss
        concept_loss = loss_fn(c_pred, c_train)
        task_loss = loss_fn(y_pred, y_train)
        task2_loss = loss_fn(y2_pred, y_train2)
        loss = concept_loss + concept_reg * task_loss + concept_reg * task2_loss

        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            task_accuracy = accuracy_score(y_train, y_pred > 0.5)
            task2_accuracy = accuracy_score(y_train2, y2_pred > 0.5)
            concept_accuracy = accuracy_score(c_train, c_pred > 0.5)
            print(f"Epoch {epoch}: Loss {loss.item():.2f} | Task Acc: {task_accuracy:.2f} | Task2 Acc: {task2_accuracy:.2f} | Concept Acc: {concept_accuracy:.2f}")

    print("=== Interventions ===")
    int_policy_c1 = UniformPolicy(out_annotations=Annotations({1: AxisAnnotation(["c1"])}), subset=["c1"])
    int_strategy_c1 = DoIntervention(model=concept_model.pgm.factor_modules, constants=-10)
    with intervention(policies=[int_policy_c1],
                      strategies=[int_strategy_c1],
                      on_layers=["c1.encoder"],
                      quantiles=[1]):
        cy_pred = inference_engine.query(query_concepts, evidence={'embedding': emb})
        c_pred = cy_pred[:, :c_train.shape[1]]
        y_pred = cy_pred[:, c_train.shape[1]:c_train.shape[1]+1]
        y2_pred = cy_pred[:, c_train.shape[1]+1:]
        task_accuracy = accuracy_score(y_train, y_pred > 0.5)
        task2_accuracy = accuracy_score(y_train2, y2_pred > 0.5)
        concept_accuracy = accuracy_score(c_train, c_pred > 0.5)
        print(f"Do intervention on c1 | Task Acc: {task_accuracy:.2f} | Task2 Acc: {task2_accuracy:.2f} | Concept Acc: {concept_accuracy:.2f}")
        print(cy_pred[:5])
        print()

        int_policy_c1 = RandomPolicy(out_annotations=Annotations({1: AxisAnnotation(["c1"])}), scale=100, subset=["c1"])
        int_strategy_c1 = GroundTruthIntervention(model=concept_model.pgm.factor_modules, ground_truth=torch.logit(c_train[:, 0:1], eps=1e-6))
        with intervention(policies=[int_policy_c1],
                          strategies=[int_strategy_c1],
                          on_layers=["c1.encoder"],
                          quantiles=[1]):
            cy_pred = inference_engine.query(query_concepts, evidence={'embedding': emb})
            c_pred = cy_pred[:, :c_train.shape[1]]
            y_pred = cy_pred[:, c_train.shape[1]:c_train.shape[1]+1]
            y2_pred = cy_pred[:, c_train.shape[1]+1:]
            task_accuracy = accuracy_score(y_train, y_pred > 0.5)
            task2_accuracy = accuracy_score(y_train2, y2_pred > 0.5)
            concept_accuracy = accuracy_score(c_train, c_pred > 0.5)
            print(f"Ground truth intervention on c1 | Task Acc: {task_accuracy:.2f} | Task2 Acc: {task2_accuracy:.2f} | Concept Acc: {concept_accuracy:.2f}")
            print(cy_pred[:5])

    return


if __name__ == "__main__":
    main()
