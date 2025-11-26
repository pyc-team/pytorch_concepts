"""
Concept graph representation and utilities.

This module provides a memory-efficient implementation of concept graphs using
sparse tensor representations. It includes utilities for graph analysis, conversions,
and topological operations.
"""
import torch

import pandas as pd
from typing import List, Tuple, Union, Optional, Set

from torch import Tensor
import networkx as nx


def _dense_to_sparse_pytorch(adj_matrix: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Convert dense adjacency matrix to sparse COO format using pure PyTorch.

    This is a differentiable alternative to torch_geometric's dense_to_sparse.

    Args:
        adj_matrix: Dense adjacency matrix of shape (n_nodes, n_nodes)

    Returns:
        edge_index: Tensor of shape (2, num_edges) with [source, target] indices
        edge_weight: Tensor of shape (num_edges,) with edge weights
    """
    # Get non-zero indices using torch.nonzero (differentiable)
    indices = torch.nonzero(adj_matrix, as_tuple=False)

    if indices.numel() == 0:
        # Empty graph - return empty tensors with proper shape
        device = adj_matrix.device
        dtype = adj_matrix.dtype
        return (torch.empty((2, 0), dtype=torch.long, device=device),
                torch.empty(0, dtype=dtype, device=device))

    # Transpose to get shape (2, num_edges) for edge_index
    edge_index = indices.t().contiguous()

    # Extract edge weights at non-zero positions
    edge_weight = adj_matrix[indices[:, 0], indices[:, 1]]

    return edge_index, edge_weight


class ConceptGraph:
    """
    Memory-efficient concept graph representation using sparse COO format.

    This class stores graphs in sparse format (edge list) internally, making it
    efficient for large sparse graphs. It provides utilities for graph analysis
    and conversions to dense/NetworkX/pandas formats.

    The graph is stored as:
        - edge_index: Tensor of shape (2, num_edges) with [source, target] indices
        - edge_weight: Tensor of shape (num_edges,) with edge weights
        - node_names: List of node names

    Attributes:
        edge_index (Tensor): Edge list of shape (2, num_edges)
        edge_weight (Tensor): Edge weights of shape (num_edges,)
        node_names (List[str]): Names of nodes in the graph
        n_nodes (int): Number of nodes in the graph

    Args:
        data (Tensor): Dense adjacency matrix of shape (n_nodes, n_nodes)
        node_names (List[str], optional): Node names. If None, generates default names.

    Example:
        >>> import torch
        >>> from torch_concepts import ConceptGraph
        >>>
        >>> # Create a simple directed graph
        >>> # A -> B -> C
        >>> # A -> C
        >>> adj = torch.tensor([[0., 1., 1.],
        ...                     [0., 0., 1.],
        ...                     [0., 0., 0.]])
        >>> graph = ConceptGraph(adj, node_names=['A', 'B', 'C'])
        >>>
        >>> # Get root nodes (no incoming edges)
        >>> print(graph.get_root_nodes())  # ['A']
        >>>
        >>> # Get leaf nodes (no outgoing edges)
        >>> print(graph.get_leaf_nodes())  # ['C']
        >>>
        >>> # Check edge existence
        >>> print(graph.has_edge('A', 'B'))  # True
        >>> print(graph.has_edge('B', 'A'))  # False
        >>>
        >>> # Get edge weight
        >>> print(graph.get_edge_weight('A', 'C'))  # 1.0
        >>>
        >>> # Get successors and predecessors
        >>> print(graph.get_successors('A'))  # ['B', 'C']
        >>> print(graph.get_predecessors('C'))  # ['A', 'B']
        >>>
        >>> # Check if DAG
        >>> print(graph.is_dag())  # True
        >>>
        >>> # Topological sort
        >>> print(graph.topological_sort())  # ['A', 'B', 'C']
        >>>
        >>> # Convert to NetworkX for visualization
        >>> nx_graph = graph.to_networkx()
        >>>
        >>> # Convert to pandas DataFrame
        >>> df = graph.to_pandas()
        >>> print(df)
        >>>
        >>> # Create from sparse format directly
        >>> edge_index = torch.tensor([[0, 0, 1], [1, 2, 2]])
        >>> edge_weight = torch.tensor([1.0, 1.0, 1.0])
        >>> graph2 = ConceptGraph.from_sparse(
        ...     edge_index, edge_weight, n_nodes=3,
        ...     node_names=['X', 'Y', 'Z']
        ... )
    """

    def __init__(self, data: Tensor, node_names: Optional[List[str]] = None):
        """Create new ConceptGraph instance from dense adjacency matrix."""
        # Validate shape
        if data.dim() != 2:
            raise ValueError(f"Adjacency matrix must be 2D, got {data.dim()}D")
        if data.shape[0] != data.shape[1]:
            raise ValueError(f"Adjacency matrix must be square, got shape {data.shape}")

        self._n_nodes = data.shape[0]
        self.node_names = node_names if node_names is not None else [f"node_{i}" for i in range(self._n_nodes)]

        if len(self.node_names) != self._n_nodes:
            raise ValueError(f"Number of node names ({len(self.node_names)}) must match matrix size ({self._n_nodes})")

        # Pre-compute node name to index mapping for O(1) lookup
        self._node_name_to_index = {name: idx for idx, name in enumerate(self.node_names)}

        # Convert to sparse format and store
        self.edge_index, self.edge_weight = _dense_to_sparse_pytorch(data)

        # Cache networkx graph for faster repeated access
        self._nx_graph_cache = None

    @classmethod
    def from_sparse(cls, edge_index: Tensor, edge_weight: Tensor, n_nodes: int, node_names: Optional[List[str]] = None):
        """
        Create ConceptGraph directly from sparse format (more efficient).
        
        Args:
            edge_index: Tensor of shape (2, num_edges) with [source, target] indices
            edge_weight: Tensor of shape (num_edges,) with edge weights
            n_nodes: Number of nodes in the graph
            node_names: Optional node names
            
        Returns:
            ConceptGraph instance
            
        Example:
            >>> import torch
            >>> from torch_concepts import ConceptGraph
            >>> edge_index = torch.tensor([[0, 0, 1], [1, 2, 2]])
            >>> edge_weight = torch.tensor([1.0, 1.0, 1.0])
            >>> graph = ConceptGraph.from_sparse(edge_index, edge_weight, n_nodes=3)
        """
        # Create instance without going through __init__
        instance = cls.__new__(cls)
        instance._n_nodes = n_nodes
        instance.node_names = node_names if node_names is not None else [f"node_{i}" for i in range(n_nodes)]
        
        if len(instance.node_names) != n_nodes:
            raise ValueError(f"Number of node names ({len(instance.node_names)}) must match n_nodes ({n_nodes})")
        
        # Pre-compute node name to index mapping for O(1) lookup
        instance._node_name_to_index = {name: idx for idx, name in enumerate(instance.node_names)}

        instance.edge_index = edge_index
        instance.edge_weight = edge_weight

        # Cache networkx graph for faster repeated access
        instance._nx_graph_cache = None

        return instance

    @property
    def n_nodes(self) -> int:
        """Get number of nodes in the graph."""
        return self._n_nodes

    @property
    def data(self) -> Tensor:
        """
        Get dense adjacency matrix representation.
        
        Note: This reconstructs the dense matrix from sparse format.
        For frequent dense access, consider caching the result.
        
        Returns:
            Dense adjacency matrix of shape (n_nodes, n_nodes)
        """
        # Reconstruct dense matrix from sparse format
        adj = torch.zeros(self._n_nodes, self._n_nodes, dtype=self.edge_weight.dtype, device=self.edge_weight.device)
        adj[self.edge_index[0], self.edge_index[1]] = self.edge_weight
        return adj

    def _node_to_index(self, node: Union[str, int]) -> int:
        """Convert node name or index to index."""
        if isinstance(node, int):
            if node < 0 or node >= self.n_nodes:
                raise IndexError(f"Node index {node} out of range [0, {self.n_nodes})")
            return node
        elif isinstance(node, str):
            # Use pre-computed dictionary for O(1) lookup instead of O(n) list search
            idx = self._node_name_to_index.get(node)
            if idx is None:
                raise ValueError(f"Node '{node}' not found in graph")
            return idx
        else:
            raise TypeError(f"Node must be str or int, got {type(node)}")

    def __getitem__(self, key):
        """
        Allow indexing like graph[i, j] or graph['A', 'B'].
        
        For single edge queries (tuple of 2), uses sparse lookup.
        For slice/advanced indexing, falls back to dense representation.
        """
        if isinstance(key, tuple) and len(key) == 2:
            # Optimized path for single edge lookup
            row = self._node_to_index(key[0])
            col = self._node_to_index(key[1])
            
            # Search in sparse edge list
            mask = (self.edge_index[0] == row) & (self.edge_index[1] == col)
            if mask.any():
                return self.edge_weight[mask]
            return torch.tensor(0.0, dtype=self.edge_weight.dtype, device=self.edge_weight.device)
        
        # For advanced indexing, use dense representation
        return self.data[key]

    def get_edge_weight(self, source: Union[str, int], target: Union[str, int]) -> float:
        """
        Get the weight of an edge.

        Args:
            source: Source node name or index
            target: Target node name or index

        Returns:
            Edge weight value (0.0 if edge doesn't exist)
        """
        source_idx = self._node_to_index(source)
        target_idx = self._node_to_index(target)
        
        # Search in sparse edge list
        mask = (self.edge_index[0] == source_idx) & (self.edge_index[1] == target_idx)
        if mask.any():
            return self.edge_weight[mask].item()
        return 0.0

    def has_edge(self, source: Union[str, int], target: Union[str, int], threshold: float = 0.0) -> bool:
        """
        Check if an edge exists between two nodes.

        Args:
            source: Source node name or index
            target: Target node name or index
            threshold: Minimum weight to consider as edge

        Returns:
            True if edge exists, False otherwise
        """
        weight = self.get_edge_weight(source, target)
        return abs(weight) > threshold

    def to_pandas(self) -> pd.DataFrame:
        """
        Convert adjacency matrix to pandas DataFrame.

        Returns:
            pd.DataFrame with node names as index and columns
        """
        return pd.DataFrame(
            self.data.cpu().numpy(),
            index=self.node_names,
            columns=self.node_names
        )

    @property
    def _nx_graph(self) -> nx.DiGraph:
        """
        Get cached NetworkX graph (lazy initialization).

        This property caches the NetworkX graph for faster repeated access.
        The cache is created on first access.

        Returns:
            nx.DiGraph: Cached NetworkX directed graph
        """
        if self._nx_graph_cache is None:
            self._nx_graph_cache = self.to_networkx()
        return self._nx_graph_cache

    def to_networkx(self, threshold: float = 0.0) -> nx.DiGraph:
        """
        Convert to NetworkX directed graph.

        Args:
            threshold: Minimum absolute value to consider as an edge

        Returns:
            nx.DiGraph: NetworkX directed graph
        """
        # If threshold is 0.0 and we have a cache, return it
        if threshold == 0.0 and self._nx_graph_cache is not None:
            return self._nx_graph_cache

        # Create empty directed graph
        G = nx.DiGraph()
        
        # Add all nodes with their names
        G.add_nodes_from(self.node_names)
        
        # Add edges from sparse representation
        edge_index_np = self.edge_index.cpu().numpy()
        edge_weight_np = self.edge_weight.cpu().numpy()
        
        for i in range(edge_index_np.shape[1]):
            source_idx = edge_index_np[0, i]
            target_idx = edge_index_np[1, i]
            weight = edge_weight_np[i]
            
            # Apply threshold
            if abs(weight) > threshold:
                source_name = self.node_names[source_idx]
                target_name = self.node_names[target_idx]
                G.add_edge(source_name, target_name, weight=weight)
        
        # Cache if threshold is 0.0
        if threshold == 0.0 and self._nx_graph_cache is None:
            self._nx_graph_cache = G

        return G

    def dense_to_sparse(self, threshold: float = 0.0) -> Tuple[Tensor, Tensor]:
        """
        Get sparse COO format (edge list) representation.

        Args:
            threshold: Minimum value to consider as an edge (default: 0.0)

        Returns:
            edge_index: Tensor of shape (2, num_edges) with source and target indices
            edge_weight: Tensor of shape (num_edges,) with edge weights
        """
        if threshold > 0.0:
            # Filter edges by threshold
            mask = torch.abs(self.edge_weight) > threshold
            return self.edge_index[:, mask], self.edge_weight[mask]
        return self.edge_index, self.edge_weight

    def get_root_nodes(self) -> List[str]:
        """
        Get nodes with no incoming edges (in-degree = 0).

        Returns:
            List of root node names
        """
        G = self._nx_graph
        return [node for node, degree in G.in_degree() if degree == 0]

    def get_leaf_nodes(self) -> List[str]:
        """
        Get nodes with no outgoing edges (out-degree = 0).

        Returns:
            List of leaf node names
        """
        G = self._nx_graph
        return [node for node, degree in G.out_degree() if degree == 0]

    def topological_sort(self) -> List[str]:
        """
        Compute topological ordering of nodes.

        Only valid for directed acyclic graphs (DAGs).

        Returns:
            List of node names in topological order

        Raises:
            nx.NetworkXError: If graph contains cycles
        """
        G = self._nx_graph
        return list(nx.topological_sort(G))

    def get_predecessors(self, node: Union[str, int]) -> List[str]:
        """
        Get immediate predecessors (parents) of a node.

        Args:
            node: Node name (str) or index (int)

        Returns:
            List of predecessor node names
        """
        G = self._nx_graph
        node_name = self.node_names[node] if isinstance(node, int) else node
        return list(G.predecessors(node_name))

    def get_successors(self, node: Union[str, int]) -> List[str]:
        """
        Get immediate successors (children) of a node.

        Args:
            node: Node name (str) or index (int)

        Returns:
            List of successor node names
        """
        G = self._nx_graph
        node_name = self.node_names[node] if isinstance(node, int) else node
        return list(G.successors(node_name))

    def get_ancestors(self, node: Union[str, int]) -> Set[str]:
        """
        Get all ancestors of a node (transitive predecessors).

        Args:
            node: Node name (str) or index (int)

        Returns:
            Set of ancestor node names
        """
        G = self._nx_graph
        node_name = self.node_names[node] if isinstance(node, int) else node
        return nx.ancestors(G, node_name)

    def get_descendants(self, node: Union[str, int]) -> Set[str]:
        """
        Get all descendants of a node (transitive successors).

        Args:
            node: Node name (str) or index (int)

        Returns:
            Set of descendant node names
        """
        G = self._nx_graph
        node_name = self.node_names[node] if isinstance(node, int) else node
        return nx.descendants(G, node_name)

    def is_directed_acyclic(self) -> bool:
        """
        Check if the graph is a directed acyclic graph (DAG).

        Returns:
            True if graph is a DAG, False otherwise
        """
        G = self._nx_graph
        return nx.is_directed_acyclic_graph(G)

    def is_dag(self) -> bool:
        """
        Check if the graph is a directed acyclic graph (DAG).

        Alias for is_directed_acyclic() for convenience.

        Returns:
            True if graph is a DAG, False otherwise
        """
        return self.is_directed_acyclic()


def dense_to_sparse(
        adj_matrix: Union[ConceptGraph, Tensor],
        threshold: float = 0.0
) -> Tuple[Tensor, Tensor]:
    """
    Convert dense adjacency matrix to sparse COO format (edge list).

    Uses PyTorch Geometric's native dense_to_sparse function.

    Args:
        adj_matrix: Dense adjacency matrix (ConceptGraph or Tensor) of shape (n_nodes, n_nodes)
        threshold: Minimum absolute value to consider as an edge (only used in fallback)

    Returns:
        edge_index: Tensor of shape (2, num_edges) with [source_indices, target_indices]
        edge_weight: Tensor of shape (num_edges,) with edge weights

    Example:
        >>> import torch
        >>> from torch_concepts.nn.modules.mid.constructors.concept_graph import dense_to_sparse
        >>> adj = torch.tensor([[0., 1., 0.],
        ...                     [0., 0., 1.],
        ...                     [0., 0., 0.]])
        >>> edge_index, edge_weight = dense_to_sparse(adj)
        >>> print(edge_index)
        tensor([[0, 1],
                [1, 2]])
        >>> print(edge_weight)
        tensor([1., 1.])
    """
    # Extract tensor data
    if isinstance(adj_matrix, ConceptGraph):
        adj_tensor = adj_matrix.data
    else:
        adj_tensor = adj_matrix

    return _dense_to_sparse_pytorch(adj_tensor)


def to_networkx_graph(
        adj_matrix: Union[ConceptGraph, Tensor],
        node_names: Optional[List[str]] = None,
        threshold: float = 0.0
) -> nx.DiGraph:
    """
    Convert adjacency matrix to NetworkX directed graph.

    Uses NetworkX's native from_numpy_array function for conversion.

    Args:
        adj_matrix: Adjacency matrix (ConceptGraph or Tensor)
        node_names: Optional node names. If adj_matrix is ConceptGraph,
                   uses its node_names. Otherwise uses integer indices.
        threshold: Minimum absolute value to consider as an edge

    Returns:
        nx.DiGraph: NetworkX directed graph

    Example:
        >>> import torch
        >>> from torch_concepts.nn.modules.mid.constructors.concept_graph import to_networkx_graph
        >>> adj = torch.tensor([[0., 1., 1.],
        ...                     [0., 0., 1.],
        ...                     [0., 0., 0.]])
        >>> G = to_networkx_graph(adj, node_names=['A', 'B', 'C'])
        >>> print(list(G.nodes()))  # ['A', 'B', 'C']
        >>> print(list(G.edges()))  # [('A', 'B'), ('A', 'C'), ('B', 'C')]
    """
    # Extract node names and tensor data
    if isinstance(adj_matrix, ConceptGraph):
        if node_names is None:
            node_names = adj_matrix.node_names
        adj_tensor = adj_matrix.data
    else:
        adj_tensor = adj_matrix
        if node_names is None:
            node_names = list(range(adj_tensor.shape[0]))

    # Apply threshold if needed
    if threshold > 0.0:
        adj_tensor = adj_tensor.clone()
        adj_tensor[torch.abs(adj_tensor) <= threshold] = 0.0

    # Convert to numpy for NetworkX
    adj_numpy = adj_tensor.detach().cpu().numpy()

    # Use NetworkX's native conversion
    G = nx.from_numpy_array(adj_numpy, create_using=nx.DiGraph)

    # Relabel nodes with custom names if provided
    if node_names != list(range(len(node_names))):
        mapping = {i: name for i, name in enumerate(node_names)}
        G = nx.relabel_nodes(G, mapping)

    return G


def get_root_nodes(
        adj_matrix: Union[ConceptGraph, Tensor, nx.DiGraph],
        node_names: Optional[List[str]] = None
) -> List[str]:
    """
    Get nodes with no incoming edges (in-degree = 0).

    Args:
        adj_matrix: Adjacency matrix (ConceptGraph, Tensor) or NetworkX graph
        node_names: Optional node names (only needed if adj_matrix is Tensor)

    Returns:
        List of root node names

    Example:
        >>> import torch
        >>> from torch_concepts.nn.modules.mid.constructors.concept_graph import get_root_nodes
        >>> adj = torch.tensor([[0., 1., 1.],
        ...                     [0., 0., 1.],
        ...                     [0., 0., 0.]])
        >>> roots = get_root_nodes(adj, node_names=['A', 'B', 'C'])
        >>> print(roots)  # ['A']
    """
    if isinstance(adj_matrix, nx.DiGraph):
        G = adj_matrix
    else:
        if isinstance(adj_matrix, ConceptGraph):
            node_names = adj_matrix.node_names
        G = to_networkx_graph(adj_matrix, node_names=node_names)

    return [node for node, degree in G.in_degree() if degree == 0]


def get_leaf_nodes(
        adj_matrix: Union[ConceptGraph, Tensor, nx.DiGraph],
        node_names: Optional[List[str]] = None
) -> List[str]:
    """
    Get nodes with no outgoing edges (out-degree = 0).

    Args:
        adj_matrix: Adjacency matrix (ConceptGraph, Tensor) or NetworkX graph
        node_names: Optional node names (only needed if adj_matrix is Tensor)

    Returns:
        List of leaf node names

    Example:
        >>> import torch
        >>> from torch_concepts.nn.modules.mid.constructors.concept_graph import get_leaf_nodes
        >>> adj = torch.tensor([[0., 1., 1.],
        ...                     [0., 0., 1.],
        ...                     [0., 0., 0.]])
        >>> leaves = get_leaf_nodes(adj, node_names=['A', 'B', 'C'])
        >>> print(leaves)  # ['C']
    """
    if isinstance(adj_matrix, nx.DiGraph):
        G = adj_matrix
    else:
        if isinstance(adj_matrix, ConceptGraph):
            node_names = adj_matrix.node_names
        G = to_networkx_graph(adj_matrix, node_names=node_names)

    return [node for node, degree in G.out_degree() if degree == 0]


def topological_sort(
        adj_matrix: Union[ConceptGraph, Tensor, nx.DiGraph],
        node_names: Optional[List[str]] = None
) -> List[str]:
    """
    Compute topological ordering of nodes (only for DAGs).

    Uses NetworkX's native topological_sort function.

    Args:
        adj_matrix: Adjacency matrix (ConceptGraph, Tensor) or NetworkX graph
        node_names: Optional node names (only needed if adj_matrix is Tensor)

    Returns:
        List of node names in topological order

    Raises:
        nx.NetworkXError: If graph contains cycles

    Example:
        >>> import torch
        >>> from torch_concepts.nn.modules.mid.constructors.concept_graph import topological_sort
        >>> adj = torch.tensor([[0., 1., 1.],
        ...                     [0., 0., 1.],
        ...                     [0., 0., 0.]])
        >>> ordered = topological_sort(adj, node_names=['A', 'B', 'C'])
        >>> print(ordered)  # ['A', 'B', 'C']
    """
    if isinstance(adj_matrix, nx.DiGraph):
        G = adj_matrix
    else:
        if isinstance(adj_matrix, ConceptGraph):
            node_names = adj_matrix.node_names
        G = to_networkx_graph(adj_matrix, node_names=node_names)

    return list(nx.topological_sort(G))


def get_predecessors(
        adj_matrix: Union[ConceptGraph, Tensor, nx.DiGraph],
        node: Union[str, int],
        node_names: Optional[List[str]] = None
) -> List[str]:
    """
    Get immediate predecessors (parents) of a node.

    Uses NetworkX's native predecessors method.

    Args:
        adj_matrix: Adjacency matrix (ConceptGraph, Tensor) or NetworkX graph
        node: Node name (str) or index (int)
        node_names: Optional node names (only needed if adj_matrix is Tensor)

    Returns:
        List of predecessor node names

    Example:
        >>> import torch
        >>> from torch_concepts.nn.modules.mid.constructors.concept_graph import get_predecessors
        >>> adj = torch.tensor([[0., 1., 1.],
        ...                     [0., 0., 1.],
        ...                     [0., 0., 0.]])
        >>> preds = get_predecessors(adj, 'C', node_names=['A', 'B', 'C'])
        >>> print(preds)  # ['A', 'B']
    """
    if isinstance(adj_matrix, nx.DiGraph):
        G = adj_matrix
        if isinstance(node, int) and node_names:
            node = node_names[node]
    else:
        if isinstance(adj_matrix, ConceptGraph):
            node_names = adj_matrix.node_names
        G = to_networkx_graph(adj_matrix, node_names=node_names)
        if isinstance(node, int):
            node = node_names[node]

    return list(G.predecessors(node))


def get_successors(
        adj_matrix: Union[ConceptGraph, Tensor, nx.DiGraph],
        node: Union[str, int],
        node_names: Optional[List[str]] = None
) -> List[str]:
    """
    Get immediate successors (children) of a node.

    Uses NetworkX's native successors method.

    Args:
        adj_matrix: Adjacency matrix (ConceptGraph, Tensor) or NetworkX graph
        node: Node name (str) or index (int)
        node_names: Optional node names (only needed if adj_matrix is Tensor)

    Returns:
        List of successor node names

    Example:
        >>> import torch
        >>> from torch_concepts.nn.modules.mid.constructors.concept_graph import get_successors
        >>> adj = torch.tensor([[0., 1., 1.],
        ...                     [0., 0., 1.],
        ...                     [0., 0., 0.]])
        >>> succs = get_successors(adj, 'A', node_names=['A', 'B', 'C'])
        >>> print(succs)  # ['B', 'C']
    """
    if isinstance(adj_matrix, nx.DiGraph):
        G = adj_matrix
        if isinstance(node, int) and node_names:
            node = node_names[node]
    else:
        if isinstance(adj_matrix, ConceptGraph):
            node_names = adj_matrix.node_names
        G = to_networkx_graph(adj_matrix, node_names=node_names)
        if isinstance(node, int):
            node = node_names[node]

    return list(G.successors(node))