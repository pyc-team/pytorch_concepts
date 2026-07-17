import torch
import torch_concepts as pyc


def main():
    # Inputs
    concepts = torch.randn(7, 3)
    embeddings = torch.randn(7, 5)

    # Annotations and AnnotatedTensor example
    in_concepts = pyc.Annotations(labels=['c1', 'c2', 'c3'])
    annotated_concepts = pyc.AnnotatedTensor(concepts, in_concepts)
    print(annotated_concepts.sum(dim=0, keepdims=True))
    print(annotated_concepts['c1', 'c3'])
    print(annotated_concepts.unsqueeze(2).permute(2, 1, 0))
    concepts2 = torch.randn(7, 3)
    in_concepts2 = pyc.Annotations(labels=['c1', 'c_test', 'c3'])
    annotated_concepts2 = pyc.AnnotatedTensor(concepts2, in_concepts2)
    print(annotated_concepts2)
    annotated_concepts2 = annotated_concepts2.union_with(annotated_concepts)
    print(annotated_concepts2)


    # Concept layer example
    class MyConceptLayer(pyc.nn.BaseConceptLayer):
        def __init__(self, in_concepts, in_embeddings, out_concepts):
            super().__init__(
                out_concepts=out_concepts,
                in_concepts=in_concepts,
                in_embeddings=in_embeddings
            )
            self.linear = torch.nn.Linear(self.in_concepts_shape+self.in_embeddings_shape, self.out_concepts_shape)

        def forward(self, concepts, embeddings):
            mix = torch.cat([concepts, embeddings], dim=-1)
            return torch.sigmoid(self.linear(mix))

    # Concept layer instantiation with integers
    layer = MyConceptLayer(in_concepts=3, in_embeddings=5, out_concepts=2)
    output = layer.forward(concepts, embeddings) # forward pass with tensors
    print(output)
    output = layer.forward(annotated_concepts, embeddings) # forward pass with annotated concepts
    print(output)

    # Concept layer instantiation with Annotations
    out_concepts = pyc.Annotations(labels=['c4', 'c5'])
    in_embeddings = pyc.Annotations(labels=['c1', 'c2', 'c3', 'c4', 'c5'])
    annotated_embeddings = pyc.AnnotatedTensor(embeddings, in_embeddings)
    layer = MyConceptLayer(in_concepts=in_concepts, in_embeddings=in_embeddings, out_concepts=out_concepts)
    output = layer.forward(annotated_concepts, annotated_embeddings) # forward pass with annotated concepts and annotated embeddings
    print(output)
    output = layer.forward(concepts, embeddings) # forward pass with tensors (should still work)
    print(output)

    # Concept layer instantiation with mixed types
    layer = MyConceptLayer(in_concepts=in_concepts, in_embeddings=5, out_concepts=out_concepts)
    output = layer.forward(annotated_concepts, embeddings)
    print(output)

    # Annotate the output
    annotated_output = layer.annotate(output)
    print(annotated_output)


if __name__ == "__main__":
    main()
