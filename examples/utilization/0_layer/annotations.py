"""
Concept Bottleneck Model (Low-Level Interface)
===============================================

This example demonstrates how to implement a Concept Bottleneck Model (CBM) using
the low-level interface of PyC, which provides pure PyTorch syntax.

Key Components:
- LinearEmbeddingToConcept: Maps latent embeddings (Z) to concept predictions (C)
- LinearConceptToConcept: Maps concept predictions (C) to task predictions (Y)
- Intervention API: Allows concept interventions at inference time

This low-level approach gives you full control over:
- Model architecture and layer composition
- Training loop and optimization
- Loss computation and weighting
- Intervention strategies during inference

Dataset: XOR toy dataset with 2 binary concepts and 1 binary task
"""
import torch

import torch_concepts as pyc


def main():
    # --------------------------------------------------------------------------
    # Concept annotations
    # --------------------------------------------------------------------------
    annotations = pyc.Annotations({
        1: pyc.AxisAnnotation(
            labels=["smoking", "genotype", "tar", "cancer"],
            cardinalities=[1, 3, 1, 1],
            types=["binary", "categorical", "continuous", "binary"]
        )
    })

    # Accessing concept annotations by name    
    annotations[1].concept("genotype")



    # --------------------------------------------------------------------------
    # Annotated tensors
    # --------------------------------------------------------------------------

    tensor = pyc.AnnotatedTensor(
        data=torch.randn(10, 6),    # (batch_size, sum(cardinalities))
        annotation=annotations[1]
    )

    # slice by concept name
    tensor["smoking"]  

    # slice by concept type
    tensor.split_by_type('binary')  

    return


if __name__ == "__main__":
    main()
