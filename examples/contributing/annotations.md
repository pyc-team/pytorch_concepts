# Creating an Annotation Object

This guide explains how to create and use an `Annotations` object in PyC. Annotations are essential for describing the structure, types, and metadata of concepts in your dataset.

## What is an Annotation?
An `Annotations` object organizes metadata for each concept axis in your data. It enables:
- Consistent handling of concept names, types, and cardinalities
- Integration with metrics, loss functions, and model logic
- Support for advanced features like causal graphs and interventions

## Key Classes
- `Annotations`: Describes the concept axis (labels, types, cardinalities, metadata)

## Minimal Example
```python
from torch_concepts.annotations import Annotations

concept_names = ['color', 'shape', 'size']
cardinalities = [3, 2, 1]  # 3 colors, 2 shapes, 1 binary size
types = ['categorical', 'categorical', 'binary']
metadata = {}

annotations = Annotations(
    labels=concept_names,
    cardinalities=cardinalities,
    types=types,
    metadata=metadata
)
```

## Annotations Arguments
- `labels`: List of concept names (required)
- `cardinalities`: List of number of states per concept (required)
- 'types': List of type per concept (i.e., 'binary', 'categorical' or 'continuous').
- `metadata`: (optional) Dict of extra metadata per concept
- `states`: (optional) List of state labels for each concept

## Example with States
```python
states = [
    ['red', 'green', 'blue'],
    ['circle', 'square'],
    ['large']
]
annotations = Annotations(
    labels=concept_names,
    cardinalities=cardinalities,
    types=types,
    metadata=metadata,
    states=states
)
```

## Accessing Annotation Info
```python
# Get concept names
print(annotations.labels)
# Get cardinalities
print(annotations.cardinalities)
# Get types
print(annotations.types)
# Get metadata for a concept
print(annotations.metadata['color'])
```

## Concepts and Tasks
Concepts and tasks share a single `Annotations` object; list them together as labels:
```python
annotations = Annotations(
    labels=['c1', 'c2', 'task1', 'task2'],
    cardinalities=[1, 1, 1, 1],
    types=['binary', 'binary', 'binary', 'binary']
)
```

## Best Practices
- Use **unique** and clear concept names
- Set correct cardinalities and types

## Reference
See the [API documentation](../../doc/modules/annotations.rst) for full details and advanced usage.
