# Contributing a New Dataset

This guide will help you implement a new dataset in <img src="../../doc/_static/img/logos/pyc.svg" width="25px" align="center"/> PyC and also enable its usage in <img src="../../doc/_static/img/logos/conceptarium.svg" width="25px" align="center"/> Conceptarium. The process involves creating two main components:

1. **Dataset Class** (`dataset_name.py`) - handles data loading, downloading, and building
2. **DataModule Class** (`datamodule_name.py`) - handles data splitting, transformations, and PyTorch Lightning integration

## Prerequisites

Before implementing your dataset, ensure you have:
- Raw data files or a method to download/generate them
- Knowledge of the concept structure (concept names, types, cardinalities)
- (Optional) Causal graph structure between concepts

## Part 1: Implementing the Dataset Class

All <img src="../../doc/_static/img/logos/pyc.svg" width="25px" align="center"/> PyC dataset classes should extend `ConceptDataset` from `torch_concepts.data.base.dataset` and be placed in `torch_concepts/data/datasets/your_dataset.py`.

All datasets should provide 4 main objects to the base class `ConceptDataset`:
- `input_data`: raw input features as torch.Tensor
- `concepts`: concept labels as torch.Tensor or pandas DataFrame
- `annotations`: an Annotations object describing concepts' properties
- `graph`: optional causal graph as a pandas DataFrame

### 1.1 Init Structure

```python
import os
import torch
import pandas as pd
from typing import List, Mapping, Optional
from torch_concepts import Annotations, AxisAnnotation
from torch_concepts.data.base import ConceptDataset
from torch_concepts.data.io import download_url

class YourDataset(ConceptDataset):
    """Dataset class for [Your Dataset Name].
    
    [Brief description of what this dataset represents]
    
    Args:
        root: Root directory where the dataset is stored or will be downloaded.
        ...[Other dataset-specific parameters]
        concept_subset: Optional subset of concept labels to use.
        label_descriptions: Optional dict mapping concept names to descriptions.
        ...[Other dataset-specific optional parameters]
    """
    
    def __init__(
        self,
        root: str = None,
        # Add your dataset-specific parameters here
        # ...
        concept_subset: Optional[list] = None, # subset of concept labels
        label_descriptions: Optional[Mapping] = None,
        # Add your dataset-specific optional parameters here
        # ...
    ):
        self.root = root
        self.label_descriptions = label_descriptions
        # Store other parameters as needed
        # ...
        
        # Load data and annotations
        input_data, concepts, annotations, graph = self.load()
        
        # Initialize parent class
        super().__init__(
            input_data=input_data,
            concepts=concepts,
            annotations=annotations,
            graph=graph,
            concept_names_subset=concept_subset,
        )
```

### 1.2 Required Properties

#### `raw_filenames`
Defines which raw files need to be present in the root directory in order to skip download(). Returns a list of filenames. The download() method below should ensure these files are created.

```python
@property
def raw_filenames(self) -> List[str]:
    """List of raw filenames that must be present to skip downloading."""
    # Example: dataset needs a CSV file
    return ["dataset.csv"]
    
    # If nothing needs downloading (e.g., generated data):
    # return []
```

#### `processed_filenames`
Defines which processed files need to be present in the root directory in order to skip build(). Returns a list of filenames. The build() method below should ensure these files are created. 

If the dataset is synthetic and dependent on a seed, include the seed in the filenames to avoid conflicts.

```python
@property
def processed_filenames(self) -> List[str]:
    """List of processed filenames that will be created during build step."""
    return [
        "raw_data.pt",
        "concepts.h5",
        "annotations.pt",
        "graph.h5"
    ]
```

### 1.3 Required Methods

#### `download()`
Downloads raw data files from external sources. This should be skipped if data is already present in the root directory.

```python
def download(self):
    """Download raw data files to root directory."""
    # Example: Download from URL
    url = "https://example.com/dataset.zip"
    download_url(url, self.root_dir)
    
    # Decompress if needed
    # ...
```

#### `build()`
Processes raw data into a desired format. This is the most important method. This allow to store objects to avoid doing the processing at each loading. 
Importantly, this is were the `Annotations` object shold be created. See [annotations.md](./annotations.md) for details and advanced usage


```python
def build(self):
    """Build processed dataset from raw files."""
    # Step 1: Ensure raw data is available
    self.maybe_download()
    
    # Step 2: Load raw data
    # Example: Load from CSV
    df = pd.read_csv(self.raw_paths[0]) 
    
    # Step 3: Extract/generate embeddings (input features)
    embeddings = ...
    
    # Step 4: Extract concepts
    concepts = ...
    
    # Step 5: Create concept annotations
    concept_names = list(concept_columns)
    
    # Define metadata for each concept (REQUIRED: must include 'type')
    # type can be 'discrete' or 'continuous' ('continuous' is not yet supported)
    concept_metadata = {
        name: {'type': 'discrete'} for name in concept_names
    }
    
    # Define cardinalities (number of possible values)
    # For binary concepts: use 1
    # For categorical with K classes: use K
    # For continuous concepts: use 1 for scalars and >1 for vectors
    cardinalities = [3, 3, 1, 1]  # Example: 4 concepts with different cardinalities
    
    # State names can also be provided (this is optional)
    # if not, default is '0', '1', ...
    states = [[label_1, label_2, label_3], # state labels for concept 1 
              [label_1, label_2, label_3], # state labels for concept 2
              [label_1, label_2], # state labels for concept 3
              [label_1, label_2]] # state labels for concept 4

    # Create annotations object
    annotations = Annotations({
        # Axis 0 is batch (usually not annotated)
        # Axis 1 is concepts (MUST be annotated)
        1: AxisAnnotation(
            labels=concept_names,
            cardinalities=cardinalities,
            metadata=concept_metadata
        )
    })
    
    # Step 6 (optional): If the dataset has a causal graph structure, create it here
    # skip this step if no graph is available, graph defaults is `None`
    graph = pd.DataFrame(
        adjacency_matrix,  # your adj: numpy array or similar
        index=concept_names,
        columns=concept_names
    )
    graph = graph.astype(int)
    
    # Step 7: Save all components
    print(f"Saving dataset to {self.root_dir}")
    torch.save(embeddings, self.processed_paths[0])
    concepts.to_hdf(self.processed_paths[1], key="concepts", mode="w")
    torch.save(annotations, self.processed_paths[2])
    graph.to_hdf(self.processed_paths[3], key="graph", mode="w")
```

#### `load_raw()` and `load()`
Load the built dataset files. These functions can be kept very simple. Preprocessing steps on the stored datsets can be added in `load()` if needed. 

```python
def load_raw(self):
    """Load raw processed files."""
    self.maybe_build()  # Ensures build() is called if needed
    
    print(f"Loading dataset from {self.root_dir}")
    inputs = torch.load(self.processed_paths[0], weights_only=False)
    concepts = pd.read_hdf(self.processed_paths[1], "concepts")
    annotations = torch.load(self.processed_paths[2], weights_only=False)
    graph = pd.read_hdf(self.processed_paths[3], "graph")
    
    return embeddings, concepts, annotations, graph

def load(self):
    """Load and optionally preprocess dataset."""
    inputs, concepts, annotations, graph = self.load_raw()
    
    # Add any additional preprocessing here if needed
    # For most cases, just return raw data
    
    return inputs, concepts, annotations, graph
```



### 1.4 Implementing custom __get_item__()

At this level, you can customize how individual samples are retrieved from the dataset. The default implementation returns a dictionary with 'inputs' and 'concepts' keys. If your dataset has multiple input or concept modalities, you can modify this method accordingly.

```python
def __getitem__(self, idx: int) -> dict:
    """Retrieve a single sample from the dataset.
    Args:
        idx: Index of the sample to retrieve.
    Returns:
        A dictionary with keys:
            'inputs': dict with key 'x' for input features tensor
            'concepts': dict with key 'c' for concept labels tensor
    """
    # example implementation
    sample = {
        'inputs': {
            'x': self.input_data[idx]
            # ... add other input modalities if needed
        },
        'concepts': {
            'c': self.concepts[idx]
            # ... add other concept modalities if needed
        }
    }
    return sample
```





### 1.5 Complete Example Template

See `torch_concepts/data/datasets/bnlearn.py` for a complete reference implementation.




## Part 2: Implementing the DataModule Class

The DataModule handles data splitting, transformations, and integration with PyTorch Lightning. Place it in `torch_concepts/data/datamodules/your_datamodule.py`.

### 2.1 Basic DataModule (Extends Default)

Your datamodule should extend `ConceptDataModule` from `torch_concepts.data.base.datamodule`.

```python
from env import DATA_ROOT
from torch_concepts.data.datasets import YourDataset
from torch_concepts.data.base.datamodule import ConceptDataModule
from torch_concepts.typing import BackboneType


class YourDataModule(ConceptDataModule):
    """DataModule for Your Dataset.
    
    Handles data loading, splitting, and batching for your dataset
    with support for concept-based learning.
    
    Args:
        seed: Random seed for splitting and eventually data generation
        val_size: Validation set size (fraction or absolute count)
        test_size: Test set size (fraction or absolute count)
        ftune_size: Fine-tuning set size (fraction or absolute count)
        ftune_val_size: Fine-tuning validation set size (fraction or absolute count)
        batch_size: Batch size for dataloaders
        backbone: Model backbone to use (if applicable)
        precompute_embs: Whether to precompute embeddings from backbone
        force_recompute: Force recomputation of cached embeddings
        workers: Number of workers for dataloaders
        [dataset-specific parameters]
    """
    
    def __init__(
        self,
        seed: int = 42,
        val_size: int | float = 0.1,
        test_size: int | float = 0.2,
        ftune_size: int | float = 0.0,
        ftune_val_size: int | float = 0.0,
        batch_size: int = 512,
        backbone: BackboneType = None,
        precompute_embs: bool = False,
        force_recompute: bool = False,
        workers: int = 0,
        # Add your dataset-specific parameters
        concept_subset: list | None = None,
        label_descriptions: dict | None = None,
        **kwargs
    ):
        # Instantiate your dataset
        dataset = YourDataset(
            root=str(DATA_ROOT / "your_dataset_name"),
            seed=seed,
            concept_subset=concept_subset,
            label_descriptions=label_descriptions,
            # Pass other dataset-specific parameters
        )
        
        # Initialize parent class with default behavior
        super().__init__(
            dataset=dataset,
            val_size=val_size,
            test_size=test_size,
            ftune_size=ftune_size,
            ftune_val_size=ftune_val_size,
            batch_size=batch_size,
            backbone=backbone,
            precompute_embs=precompute_embs,
            force_recompute=force_recompute,
            workers=workers,
        )
```

### 2.2 Available Default Components
The following default components will be used if the corresponding parameters are not specified.

#### Default Scalers
- **None**: No scaling is applied by default. You can provide custom scalers via the `scalers` parameter if normalization is needed (e.g., `StandardScaler` for Z-score normalization, located in `torch_concepts/data/scalers/standard.py`).

#### Default Splitters
- `RandomSplitter`: Random train/val/test split (default). Located in `torch_concepts/data/splitters/random.py`.


### 2.3 Implementing Custom Scalers

If you need a custom scaler, you can extend the `Scaler` class from `torch_concepts.data.base.scaler` and place the new scaler in `torch_concepts/data/scalers/your_scaler.py`.

```python
class YourCustomScaler:
    """Custom scaler for your specific preprocessing needs."""
    
    def __init__(self, axis=0):
        self.axis = axis
        # Initialize any parameters
        
    def fit(self, data, dim=0):
        """Compute scaling parameters from training data."""
        # Calculate statistics needed for scaling
        # Store them as instance variables
        pass
        
    def transform(self, data):
        """Apply scaling to data."""
        # Apply transformation using stored parameters
        pass
        
    def fit_transform(self, data, dim=0):
        """Fit and transform in one step."""
        self.fit(data, dim)
        return self.transform(data)
        
    def inverse_transform(self, data):
        """Reverse the scaling transformation."""
        pass
```

### 2.4 Implementing Custom Splitters

If you need a custom splitter, you can extend the `Splitter` class from `torch_concepts.data.base.splitter` and place the new splitter in `torch_concepts/data/splitters/your_splitter.py`:

```python
import numpy as np


class YourCustomSplitter:
    """Custom splitter for your specific splitting logic."""
    
    def __init__(self, val_size=0.1, test_size=0.2):
        self.val_size = val_size
        self.test_size = test_size
        # Initialize split parameters
        
    def split(self, dataset):
        """Split dataset into train/val/test indices.
        
        Args:
            dataset: The ConceptDataset to split
            
        Sets:
            self.train_idxs: Training set indices
            self.val_idxs: Validation set indices  
            self.test_idxs: Test set indices
            self.ftune_idxs: Fine-tuning set indices (optional)
            self.ftune_val_idxs: Fine-tuning validation indices (optional)
        """
        n = len(dataset)
        indices = np.arange(n)
        
        # Implement your splitting logic
        # Example: stratified split, temporal split, etc.
        
        # Set the indices
        self.train_idxs = indices[:train_end]
        self.val_idxs = indices[train_end:val_end]
        self.test_idxs = indices[val_end:]
        self.ftune_idxs = []
        self.ftune_val_idxs = []
```

## Part 3: Creating the Configuration File

A YAML configuration file is **required** for integrating your dataset with the Hydra-based configuration system used in Conceptarium. This file defines default parameters and allows users to easily customize dataset settings.

### 3.1 Configuration File Structure

Create a configuration file at `conceptarium/conf/dataset/your_dataset.yaml`.

#### Basic Configuration Template

```yaml
defaults:
  - _commons
  - _self_

# Target class for Hydra instantiation
_target_: torch_concepts.data.datamodules.your_datamodule.YourDataModule       # Path to your datamodule class

# Random seed (typically inherited from global config)
seed: ${seed}

# Dataset-specific parameters
# Add all customizable parameters from your DataModule here
param1: default_value1
param2: default_value2

# Backbone configuration (if applicable)
backbone: null
precompute_embs: false
force_recompute: false

# Concept descriptions (optional but recommended)
label_descriptions:
  concept_1: "Description of concept 1"
  concept_2: "Description of concept 2"
  concept_3: "Description of concept 3"

# Default task concept names (optional)
# Use this if your dataset has specific target concepts
default_task_names: [target_concept_name]
```

### 3.2 Understanding Configuration Components

#### `defaults`
Specifies configuration inheritance:
- `_commons`: Includes common datamodule parameters (batch_size, val_size, test_size, etc.)
- `_self_`: Ensures this file's settings override inherited defaults

#### `_target_`
The fully qualified path to your DataModule class. This tells Hydra which class to instantiate.

```yaml
_target_: torch_concepts.data.datamodules.your_datamodule.YourDataModule
```

#### `seed`
Usually inherited from the global configuration using Hydra's variable interpolation:

```yaml
seed: ${seed}
```

#### Dataset-Specific Parameters
Include **all** parameters that users might want to customize from your DataModule's `__init__` method:

#### `label_descriptions`
A dictionary mapping concept names to human-readable descriptions. This is **highly recommended** for documentation and interpretability:

```yaml
label_descriptions:
  age: "Patient age in years"
  gender: "Patient gender (0=female, 1=male)"
  diagnosis: "Primary diagnosis code"
```

#### `default_task_names`
List here the concepts that will be treated as target/task concepts by certain concept-based models, e.g., standard CBMs.

```yaml
default_task_names: [outcome, severity]
```


## Part 4: Testing Your Implementation

### 4.1 Basic Test Script

Create a test script to verify your implementation:

```python
from torch_concepts.data import YourDataset
from torch_concepts.data.datamodules import YourDataModule

# Test dataset loading
dataset = YourDataset(
    root="/path/to/data",
    seed=42,
    ...
)

print(f"Dataset: {dataset}")
print(f"Number of samples: {len(dataset)}")
print(f"Number of features: {dataset.n_features}")
print(f"Number of concepts: {dataset.n_concepts}")
print(f"Concept names: {dataset.concept_names}")

# Test sample access
sample = dataset[0]
print(f"Sample structure: {sample.keys()}")
print(f"Input shape: {sample['inputs']['x'].shape}")
print(f"Concepts shape: {sample['concepts']['c'].shape}")

# Test datamodule
datamodule = YourDataModule(
    seed=42,
    batch_size=128,
    val_size=0.1,
    test_size=0.2,
)

datamodule.setup()
print(f"\nDataModule: {datamodule}")
print(f"Train size: {datamodule.train_len}")
print(f"Val size: {datamodule.val_len}")
print(f"Test size: {datamodule.test_len}")

# Test dataloader
train_loader = datamodule.train_dataloader()
batch = next(iter(train_loader))
print(f"\nBatch structure: {batch.keys()}")
print(f"Batch input shape: {batch['inputs']['x'].shape}")
print(f"Batch concepts shape: {batch['concepts']['c'].shape}")
```

## Part 5: Integration & Submission

### 5.1 Contacting the Authors

**Important**: Contact the library authors before submitting to ensure your dataset fits the library's scope and get guidance on:
- Dataset naming conventions
- Integration with existing infrastructure
- Documentation requirements
- Testing requirements

### 5.2 Documentation

Provide the following documentation:
1. **Dataset docstring**: Clear description of data source, structure, and usage
2. **Citation**: If based on a paper, include proper citation
3. **Example usage**: If the dataset is somewhat peculiar, please create example in `torch_concepts/examples/loading-data/your_dataset.py`
4. **README entry**: Add entry and description to the torch_concepts `README.md`
