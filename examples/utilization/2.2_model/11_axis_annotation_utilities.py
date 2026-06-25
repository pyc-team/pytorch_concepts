"""
Annotations Utilities - Efficient cached index lookups for concept tensors.
"""

import torch
from torch_concepts.annotations import Annotations

# Mixed concept types: binary, categorical (3-class, 2-class), continuous
axis = Annotations(
    labels=['is_big', 'color', 'shape', 'temperature'],
    cardinalities=[1, 3, 2, 1],
    types=['binary', 'categorical', 'categorical', 'continuous'],
)

tensor = torch.arange(7).unsqueeze(0).float().expand(4, -1)  # Shape: [4, 7]

# 1. cumulative_cardinalities - O(1) position boundaries
# 0: start, 1: is_big, 4: color, 6: shape, 7: temperature
print("cumulative_cardinalities:", axis.cumulative_cardinalities)  # [0,1,4,6,7]

# 2. concept_slices - dict of slices for direct tensor indexing
print("concept_slices['color']:", axis.concept_slices['color'])  # slice(1,4)
print("color logits:", tensor[:, axis.concept_slices['color']])  # [[1,2,3]]

# 3. get_slice - unified: str→slice, list→indices
print("get_slice('shape'):", axis.get_slice('shape'))  # slice(4,6)
print("get_slice(['is_big','temperature']):", axis.get_slice(['is_big', 'temperature']))  # [0,6]

# 4. slice_tensor - extract/reorder columns by name
print("slice_tensor:", axis.slice_tensor(tensor, ['temperature', 'is_big']))  # [[6,0]]
# this is equivalent to:
# tensor[:, axis.get_slice(['temperature', 'is_big'])]

# 5. type_groups - grouped by binary/categorical/continuous
groups = axis.type_groups
print("binary labels:", groups['binary']['labels'])  # ['is_big']
print("categorical logits_idx:", groups['categorical']['logits_idx'])  # [1,2,3,4,5]
print("continuous logits_idx:", groups['continuous']['logits_idx'])  # [6]
