from torch_concepts.data.datasets import ToyDataset, CompletenessDataset


def main():
    data = ToyDataset('xor', size=1000, random_state=42)
    print(data.data.shape, data.concept_labels.shape, data.target_labels.shape,
          data.concept_attr_names, data.task_attr_names)

    data = ToyDataset('trigonometry', size=1000, random_state=42)
    print(data.data.shape, data.concept_labels.shape, data.target_labels.shape,
          data.concept_attr_names, data.task_attr_names)

    data = ToyDataset('dot', size=1000, random_state=42)
    print(data.data.shape, data.concept_labels.shape, data.target_labels.shape,
          data.concept_attr_names, data.task_attr_names)

    data = ToyDataset('checkmark', size=1000, random_state=42)
    print(data.data.shape, data.concept_labels.shape, data.target_labels.shape,
          data.concept_attr_names, data.task_attr_names, data.dag)

    data = CompletenessDataset(n_samples=1000, n_concepts=3, n_hidden_concepts=0, n_tasks=2, random_state=42)
    print(data.data.shape, data.concept_labels.shape, data.target_labels.shape,
          data.concept_attr_names, data.task_attr_names, data)

    data = CompletenessDataset(n_samples=1000, n_concepts=3, n_hidden_concepts=1, n_tasks=2, random_state=42)
    print(data.data.shape, data.concept_labels.shape, data.target_labels.shape,
          data.concept_attr_names, data.task_attr_names, data)
    return


if __name__ == "__main__":
    main()
