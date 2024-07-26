from torch_concepts.data import ToyDataset


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
    return


if __name__ == "__main__":
    main()
