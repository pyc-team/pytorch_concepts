import torch

from torch_concepts import Annotations, AxisAnnotation
from torch_concepts.data.datasets import ToyDataset
from torch_concepts.nn import LinearZU, LinearUC, MixCUC


def main():
    latent_dims = 20
    n_epochs = 2000
    n_samples = 1000
    concept_reg = 0.5
    dataset = ToyDataset(dataset='xor', seed=42, n_gen=n_samples)
    x_train = dataset.input_data
    concept_idx = list(dataset.graph.edge_index[0].unique().numpy())
    task_idx = list(dataset.graph.edge_index[1].unique().numpy())
    c_train = dataset.concepts[:, concept_idx]
    y_train = dataset.concepts[:, task_idx]
    concept_names = [dataset.concept_names[i] for i in concept_idx]
    task_names = [dataset.concept_names[i] for i in task_idx]
    n_features = x_train.shape[1]

    y = torch.stack([
        torch.randint(0, 2, (n_samples,)),  # C1 labels
        torch.randint(0, 3, (n_samples,)),  # C2 labels
        torch.randint(0, 2, (n_samples,)),  # C3 binary targets
    ], dim=1)

    concept_names = ('C1', 'C2', 'C3')
    c_cardinalities = (2, 5, 1)
    c_annotations = Annotations({1: AxisAnnotation(concept_names, cardinalities=c_cardinalities, metadata={'C1': {'train_mode': 'classification'}, 'C2': {'train_mode': 'classification'}, 'C3': {'train_mode': 'regression'}})})
    c_train= torch.stack([
        torch.randint(0, 2, (n_samples,)),  # C1 labels
        torch.randint(0, 5, (n_samples,)),  # C2 labels
        torch.randn((n_samples,)),  # C3 labels
    ], dim=1)

    task_names = ('T1', 'T2')
    y_cardinalities = (1, 5)
    y_annotations = Annotations({1: AxisAnnotation(task_names, cardinalities=y_cardinalities, metadata={'T1': {'train_mode': 'classification'}, 'T2': {'train_mode': 'classification'}})})
    y_train = torch.stack([
        torch.randint(0, 2, (n_samples,)),  # T1 labels
        torch.randint(0, 5, (n_samples,)),  # T2 labels
    ], dim=1)

    encoder = torch.nn.Sequential(
        torch.nn.Linear(n_features, latent_dims),
        torch.nn.LeakyReLU(),
    )
    exog_encoder = LinearZU(in_features=latent_dims,
                               out_features=c_annotations.shape[1],
                               exogenous_size=latent_dims)
    c_encoder = LinearUC(in_features_exogenous=latent_dims)
    y_predictor = MixCUC(in_features_endogenous=c_annotations.shape[1],
                                       in_features_exogenous=latent_dims,
                                       out_features=y_annotations.shape[1],
                                       cardinalities=c_annotations.get_axis_annotation(1).cardinalities)


    model = torch.nn.Sequential(encoder, exog_encoder, c_encoder, y_predictor)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    loss_fn_binary = torch.nn.BCEWithLogitsLoss()
    loss_fn_categorical = torch.nn.CrossEntropyLoss()
    loss_fn_regression = torch.nn.MSELoss()
    model.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()

        # generate concept and task predictions
        emb = encoder(x_train)
        exog = exog_encoder(input=emb)
        c_pred = c_encoder(exogenous=exog)
        y_pred = y_predictor(endogenous=c_pred, exogenous=exog)

        # compute loss
        concept_loss = 0
        concept_tensors = torch.split(c_pred, c_annotations.get_axis_annotation(1).cardinalities, dim=1)
        for c_id, concept_tensor in enumerate(concept_tensors):
            c_true = c_train[:, c_id:c_id+1]
            c_name = c_annotations.get_axis_annotation(1).labels[c_id]
            meta = c_annotations.get_axis_annotation(1).metadata
            if meta[c_name]['train_mode'] == 'classification':
                if concept_tensor.shape[1] > 1:
                    concept_loss += loss_fn_categorical(concept_tensor, c_true.long().ravel())
                else:
                    concept_loss += loss_fn_binary(concept_tensor, c_true)
            elif meta[c_name]['train_mode'] == 'regression':
                concept_loss += loss_fn_regression(concept_tensor, c_true)

        # compute task loss
        task_loss = 0
        task_tensors = torch.split(y_pred, y_annotations.get_axis_annotation(1).cardinalities, dim=1)
        for y_id, task_tensor in enumerate(task_tensors):
            y_true = y_train[:, y_id:y_id+1]
            y_name = y_annotations.get_axis_annotation(1).labels[y_id]
            meta = y_annotations.get_axis_annotation(1).metadata
            if meta[y_name]['train_mode'] == 'classification':
                if task_tensor.shape[1] > 1:
                    task_loss += loss_fn_categorical(task_tensor, y_true.long().ravel())
                else:
                    task_loss += loss_fn_binary(task_tensor, y_true.float())
            elif meta[y_name]['train_mode'] == 'regression':
                task_loss += loss_fn_regression(task_tensor, y_true)

        loss = concept_loss + concept_reg * task_loss

        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss {loss.item():.2f}")

    return


if __name__ == "__main__":
    main()
