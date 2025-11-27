# Creating an Annotation Object

This guide explains how to create and use an `Annotations` object in PyC. Annotations are essential for describing the structure, types, and metadata of concepts in your dataset.

## What is an Annotation?
An `Annotations` object organizes metadata for each concept axis in your data. It enables:
- Consistent handling of concept names, types, and cardinalities
- Integration with metrics, loss functions, and model logic
- Support for advanced features like causal graphs and interventions

## Key Classes
- `Annotations`: Container for all axis annotations
- `AxisAnnotation`: Describes one axis (usually concepts)

## Minimal Example
```python
from torch_concepts.annotations import Annotations, AxisAnnotation

concept_names = ['color', 'shape', 'size']
cardinalities = [3, 2, 1]  # 3 colors, 2 shapes, 1 binary size
metadata = {
    'color': {'type': 'discrete'},
    'shape': {'type': 'discrete'},
    'size': {'type': 'discrete'}
}

annotations = Annotations({
    1: AxisAnnotation(
        labels=concept_names,
        cardinalities=cardinalities,
        metadata=metadata
    )
})
```

## AxisAnnotation Arguments
- `labels`: List of concept names (required)
- `cardinalities`: List of number of states per concept (required)
- `metadata`: Dict of metadata for each concept (required, must include `'type'`)
- `states`: (optional) List of state labels for each concept

## Example with States
```python
states = [
    ['red', 'green', 'blue'],
    ['circle', 'square'],
    ['small', 'large']
]
annotations = Annotations({
    1: AxisAnnotation(
        labels=concept_names,
        cardinalities=cardinalities,
        metadata=metadata,
        states=states
    )
})
```

## Metadata Requirements
- Each concept in `metadata` must have a `'type'` field:
    - `'discrete'`: for binary/categorical concepts
    - `'continuous'`: for continuous concepts (not yet supported)
- You can add extra fields (e.g., `'distribution'`, `'description'`)

## Accessing Annotation Info
```python
# Get concept names
print(annotations.get_axis_labels(1))
# Get cardinalities
print(annotations.get_axis_cardinalities(1))
# Get metadata for a concept
print(annotations.get_axis_annotation(1).metadata['color'])
```

## Advanced: Multiple Axes
You can annotate multiple axes (e.g., concepts, tasks):
```python
annotations = Annotations({
    1: AxisAnnotation(labels=['c1', 'c2']),
    2: AxisAnnotation(labels=['task1', 'task2'])
})
```

## Best Practices
- Always annotate axis 1 (concepts)
- Use **unique** and clear concept names
- Set correct cardinalities and types

## Reference
See the [API documentation](../../doc/modules/annotations.rst) for full details and advanced usage.
