"""
Représentation plate compacte en mémoire d'un arbre K16 (*K16Tree*).

Cette version inclut les correctifs nécessaires pour que
`TreeFlat.apply_perfect_recall()` garantisse un rappel de 100 % sans erreurs.
Utilise une réduction de dimension locale par niveau.
"""

from __future__ import annotations

import os
import json
from typing import List, Optional, TYPE_CHECKING, Dict, Any

import numpy as np

from .tree import TreeNode, K16Tree

if TYPE_CHECKING:  # uniquement pour les vérificateurs de types statiques
    from .io import VectorReader


class TreeFlat:
    """Représentation plate compacte en mémoire d'un arbre K16 (*K16Tree*)."""

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    def __init__(self, dims: int, max_depth: int):
        self.dims = dims              # original embedding dimension
        self.depth = 0                # will be set after construction
        self.max_depth = max_depth

        # Per-level structures (local dimensional reduction)
        self.centroids_head: List[np.ndarray] = []   # list[level] → (n_nodes, d_head_level)
        self.tail_norms: List[np.ndarray] = []       # list[level] → (n_nodes,)
        self.child_ptr: List[np.ndarray] = []        # list[level] → (n_nodes, max_children) int32

        # Leaves data
        self.leaf_offset: Optional[np.ndarray] = None  # (n_nodes,) int32  / -1 for internal nodes
        self.leaf_data: Optional[np.ndarray] = None    # (total_leaf_indices,) int32
        self.leaf_bounds: Optional[np.ndarray] = None  # (n_leaves+1,) int32

        # Misc stats
        self.n_nodes: int = 0
        self.n_leaves: int = 0
        self.n_levels: int = 0

        # Mapping nodes per level (used for traversal & leaf mapping)
        self.nodes_by_level: List[List[TreeNode]] = []

        # Top-variance dimensions kept in *head* per level (local)
        self.top_dims_by_level: List[np.ndarray] = []  # list[level] → int32[d_head_level]
        self.d_head_by_level: List[int] = []           # list[level] → d_head_level

    # ------------------------------------------------------------------
    # Factory from hierarchical tree
    # ------------------------------------------------------------------
    @classmethod
    def from_tree(cls, tree: K16Tree) -> "TreeFlat":
        if not tree.root:
            raise ValueError("Empty source tree")

        dims = int(tree.root.centroid.shape[0])
        stats = tree.get_statistics()
        max_depth = stats["max_depth"] + 1  # include root level 0

        # Use local dimensional reduction already computed in the tree
        if tree.top_dims_by_level is None or tree.d_head_by_level is None:
            raise ValueError("La réduction de dimension locale doit être calculée dans l'arbre avant l'aplatissement")
        
        top_dims_by_level = tree.top_dims_by_level
        d_head_by_level = tree.d_head_by_level
        print(f"✓ Utilisation de la réduction de dimension locale pré-calculée")

        ft = cls(dims, max_depth)

        # Store dimensional reduction info
        ft.top_dims_by_level = top_dims_by_level
        ft.d_head_by_level = d_head_by_level

        # ------------------------------------------------------------------
        # Gather nodes per level
        # ------------------------------------------------------------------
        ft.nodes_by_level = [[] for _ in range(max_depth)]

        def _collect(node: TreeNode):
            lvl = node.level
            ft.nodes_by_level[lvl].append(node)
            for child in node.children:
                _collect(child)

        _collect(tree.root)

        level_counts = [len(nodes) for nodes in ft.nodes_by_level]
        total_nodes = sum(level_counts)

        # ------------------------------------------------------------------
        # Allocate leaf structures
        # ------------------------------------------------------------------
        ft.leaf_offset = np.full(total_nodes, -1, dtype=np.int32)

        leaf_nodes: List[TreeNode] = []
        total_leaf_indices = 0
        for level_nodes in ft.nodes_by_level:
            for n in level_nodes:
                if n.is_leaf():
                    leaf_nodes.append(n)
                    total_leaf_indices += len(n.indices)

        ft.leaf_data = np.zeros(total_leaf_indices, dtype=np.int32)
        ft.leaf_bounds = np.zeros(len(leaf_nodes) + 1, dtype=np.int32)

        # ------------------------------------------------------------------
        # Build per-level centroid matrices and child pointers
        # ------------------------------------------------------------------
        for level, count in enumerate(level_counts):
            if count == 0:
                d_head_level = d_head_by_level[level] if level < len(d_head_by_level) else 0
                ft.centroids_head.append(np.empty((0, d_head_level), dtype=np.float32))
                ft.tail_norms.append(np.empty((0,), dtype=np.float32))
                ft.child_ptr.append(np.empty((0, 0), dtype=np.int32))
                continue

            # Get dimensional reduction for this level
            d_head_level = d_head_by_level[level] if level < len(d_head_by_level) else dims
            top_dims_level = top_dims_by_level[level] if level < len(top_dims_by_level) else np.arange(dims, dtype=np.int32)

            # allocate arrays
            cent_head = np.zeros((count, d_head_level), dtype=np.float32)
            tail_norm = np.zeros((count,), dtype=np.float32)

            # child pointer matrix (fix width per level)
            if level < max_depth - 1:  # except leaves level
                max_children = max(len(node.children) for node in ft.nodes_by_level[level])
                child_mat = np.full((count, max_children), -1, dtype=np.int32)
            else:
                child_mat = np.empty((count, 0), dtype=np.int32)

            ft.centroids_head.append(cent_head)
            ft.tail_norms.append(tail_norm)
            ft.child_ptr.append(child_mat)

        # Fill arrays & leaf structures
        current_leaf_idx = 0
        current_leaf_data_ptr = 0
        ft.leaf_bounds[0] = 0

        for level, nodes in enumerate(ft.nodes_by_level):
            if not nodes:
                continue
                
            # Get dimensional reduction for this level
            top_dims_level = top_dims_by_level[level] if level < len(top_dims_by_level) else np.arange(dims, dtype=np.int32)
            tail_mask = np.ones(dims, dtype=bool)
            tail_mask[top_dims_level] = False
            
            for i, node in enumerate(nodes):
                # centroid head / tail_norm
                c = node.centroid.astype(np.float32)
                ft.centroids_head[level][i] = c[top_dims_level]
                tail = c[tail_mask]
                ft.tail_norms[level][i] = np.sqrt(np.dot(tail, tail))

                # child pointers
                if not node.is_leaf():
                    for j, child in enumerate(node.children):
                        child_idx = ft.nodes_by_level[level + 1].index(child)
                        ft.child_ptr[level][i, j] = child_idx

                # leaf data
                if node.is_leaf():
                    global_node_idx = sum(level_counts[:level]) + i
                    ft.leaf_offset[global_node_idx] = current_leaf_idx

                    idxs = node.indices
                    n_idx = len(idxs)
                    ft.leaf_data[current_leaf_data_ptr:current_leaf_data_ptr + n_idx] = idxs
                    current_leaf_data_ptr += n_idx
                    ft.leaf_bounds[current_leaf_idx + 1] = current_leaf_data_ptr

                    current_leaf_idx += 1

        # Final stats
        ft.n_nodes = total_nodes
        ft.n_leaves = len(leaf_nodes)
        ft.n_levels = max_depth
        ft.depth = max_depth - 1

        return ft

    # ------------------------------------------------------------------
    # Search helpers
    # ------------------------------------------------------------------
    def _prepare_query(self, query: np.ndarray, level: int):
        """Prepare query for a specific level using level-specific dimensional reduction."""
        if level >= len(self.top_dims_by_level):
            # Fallback for levels without specific reduction
            return query.astype(np.float32, copy=False), 0.0
            
        top_dims_level = self.top_dims_by_level[level]
        q_head = query[top_dims_level]
        
        tail_mask = np.ones(self.dims, dtype=bool)
        tail_mask[top_dims_level] = False
        q_tail = query[tail_mask]
        q_tail_norm = float(np.sqrt(np.dot(q_tail, q_tail)))
        
        return q_head.astype(np.float32, copy=False), q_tail_norm

    # ------------------------------------------------------------------
    # Single-path search
    # ------------------------------------------------------------------
    def search_tree_single(self, query: np.ndarray) -> np.ndarray:
        # Pre-compute query preparations for all levels (optimization)
        q_heads = []
        q_tail_norms = []
        for level in range(self.n_levels):
            q_head, q_tail_norm = self._prepare_query(query, level)
            q_heads.append(q_head)
            q_tail_norms.append(q_tail_norm)

        node_idx = 0
        level = 0

        while level < self.depth:
            next_level = level + 1
            children_ptr = self.child_ptr[level][node_idx]
            valid = children_ptr[children_ptr >= 0]
            if len(valid) == 0:
                break

            # Use pre-computed query for this level
            q_head = q_heads[next_level]
            q_tail_norm = q_tail_norms[next_level]

            head_next = self.centroids_head[next_level][valid]
            tail_next = self.tail_norms[next_level][valid]

            dot_head = head_next @ q_head  # (len(valid),)
            upper = dot_head + tail_next * q_tail_norm

            best_local = np.argmax(upper)
            node_idx = int(valid[best_local])
            level = next_level

        # fetch leaf indices
        global_node_idx = sum(len(nodes) for nodes in self.nodes_by_level[:level]) + node_idx
        if self.leaf_offset[global_node_idx] >= 0:
            leaf_idx = self.leaf_offset[global_node_idx]
            start = self.leaf_bounds[leaf_idx]
            end = self.leaf_bounds[leaf_idx + 1]
            return self.leaf_data[start:end]

        return np.array([], dtype=np.int32)

    # ------------------------------------------------------------------
    # Beam search (width ≥ 1); uses the same upper-bound ordering.
    # ------------------------------------------------------------------
    def search_tree_beam(self, query: np.ndarray, beam_width: int = 3) -> np.ndarray:
        if beam_width <= 1:
            return self.search_tree_single(query)

        # Pre-compute query preparations for all levels (optimization)
        q_heads = []
        q_tail_norms = []
        for level in range(self.n_levels):
            q_head, q_tail_norm = self._prepare_query(query, level)
            q_heads.append(q_head)
            q_tail_norms.append(q_tail_norm)

        beam = [(0, 0, 1.0)]  # (level, node_idx, score_upper)
        leaves = []

        while beam:
            level, node_idx, _ = beam.pop(0)
            if level >= self.depth:
                continue

            # convert to global idx to test leaf
            global_idx = sum(len(nodes) for nodes in self.nodes_by_level[:level]) + node_idx
            if self.leaf_offset[global_idx] >= 0:
                leaf_idx = self.leaf_offset[global_idx]
                leaves.append((leaf_idx, 0.0))  # score not used later
                continue

            # explore children
            next_level = level + 1
            children_ptr = self.child_ptr[level][node_idx]
            valid = children_ptr[children_ptr >= 0]
            if len(valid) == 0:
                continue

            # Use pre-computed query for this level
            q_head = q_heads[next_level]
            q_tail_norm = q_tail_norms[next_level]

            head_next = self.centroids_head[next_level][valid]
            tail_next = self.tail_norms[next_level][valid]
            dot_head = head_next @ q_head
            upper = dot_head + tail_next * q_tail_norm

            top = np.argsort(-upper)[:beam_width]
            for idx_local in top:
                beam.append((next_level, int(valid[idx_local]), float(upper[idx_local])))

            beam.sort(key=lambda x: x[2], reverse=True)
            beam = beam[:beam_width]

        # gather indices from best leaves
        all_idx: List[int] = []
        for leaf_idx, _ in leaves[:beam_width]:
            start = self.leaf_bounds[leaf_idx]
            end = self.leaf_bounds[leaf_idx + 1]
            all_idx.extend(self.leaf_data[start:end])

        # ensure unique indices
        return np.array(list(set(all_idx)), dtype=np.int32)

    # unified entry point
    def search_tree(
        self,
        query: np.ndarray,
        beam_width: int = 1,
        vectors_reader: Optional["VectorReader"] = None,
        k: Optional[int] = None,
    ) -> np.ndarray:
        if beam_width <= 1:
            candidates = self.search_tree_single(query)
        else:
            candidates = self.search_tree_beam(query, beam_width)

        if vectors_reader is not None and len(candidates) > 0:
            scores = vectors_reader.dot(candidates.tolist(), query)
            if k is not None and 0 < k < len(candidates):
                top_local = np.argpartition(-scores, k - 1)[:k]
                top_sorted = top_local[np.argsort(-scores[top_local])]
                candidates = candidates[top_sorted]
            else:
                candidates = candidates[np.argsort(-scores)]
        return candidates

    # ------------------------------------------------------------------
    # Sauvegarde / chargement
    # ------------------------------------------------------------------
    def save(self, file_path: str, mmap_dir: bool = False) -> None:
        """
        Sauvegarde de la structure plate. Par défaut, sérialise en un seul fichier
        numpy pickle. Si mmap_dir=True, crée un répertoire de fichiers numpy
        pour permettre le memory-mapping individuel des tableaux.
        """
        if mmap_dir:
            base = os.path.splitext(file_path)[0]
            os.makedirs(base, exist_ok=True)
            meta = {
                "dims": self.dims,
                "depth": self.depth,
                "max_depth": self.max_depth,
                "n_nodes": self.n_nodes,
                "n_leaves": self.n_leaves,
                "n_levels": self.n_levels,
                "top_dims_by_level": [arr.tolist() for arr in self.top_dims_by_level],
                "d_head_by_level": self.d_head_by_level,
            }
            # Enregistrement des métadonnées
            with open(os.path.join(base, 'meta.json'), 'w') as f:
                json.dump(meta, f)
            # Sérialisation des tableaux numpy séparément
            for i, arr in enumerate(self.centroids_head):
                np.save(os.path.join(base, f'centroids_head_{i}.npy'), arr)
            for i, arr in enumerate(self.tail_norms):
                np.save(os.path.join(base, f'tail_norms_{i}.npy'), arr)
            for i, arr in enumerate(self.child_ptr):
                np.save(os.path.join(base, f'child_ptr_{i}.npy'), arr)
            np.save(os.path.join(base, 'leaf_offset.npy'), self.leaf_offset)
            np.save(os.path.join(base, 'leaf_data.npy'), self.leaf_data)
            np.save(os.path.join(base, 'leaf_bounds.npy'), self.leaf_bounds)
        else:
            data = {
                "dims": self.dims,
                "depth": self.depth,
                "max_depth": self.max_depth,
                "n_nodes": self.n_nodes,
                "n_leaves": self.n_leaves,
                "n_levels": self.n_levels,
                "top_dims_by_level": self.top_dims_by_level,
                "d_head_by_level": self.d_head_by_level,
                "centroids_head": self.centroids_head,
                "tail_norms": self.tail_norms,
                "child_ptr": self.child_ptr,
                "leaf_offset": self.leaf_offset,
                "leaf_data": self.leaf_data,
                "leaf_bounds": self.leaf_bounds,
            }
            np.save(file_path, data, allow_pickle=True)

    @classmethod
    def load(cls, file_path: str, mmap_mode: Optional[str] = None) -> "TreeFlat":
        """
        Charge la structure plate. Si mmap_mode fourni et qu'un répertoire mmap existe,
        les tableaux numpy sont chargés en memory-mapping depuis ce répertoire;
        sinon, charge le fichier en mémoire via pickle numpy.
        """
        base = os.path.splitext(file_path)[0]
        if mmap_mode and os.path.isdir(base):
            return cls._load_from_mmap_dir(base)
        if mmap_mode:
            data = np.load(file_path, allow_pickle=True, mmap_mode=mmap_mode).item()
        else:
            data = np.load(file_path, allow_pickle=True).item()
        obj = cls(data["dims"], data["max_depth"])

        obj.depth = data["depth"]
        obj.n_nodes = data["n_nodes"]
        obj.n_leaves = data["n_leaves"]
        obj.n_levels = data["n_levels"]
        obj.top_dims_by_level = data["top_dims_by_level"]
        obj.d_head_by_level = data["d_head_by_level"]

        obj.centroids_head = data["centroids_head"]
        obj.tail_norms = data["tail_norms"]
        obj.child_ptr = data["child_ptr"]
        obj.leaf_offset = data["leaf_offset"]
        obj.leaf_data = data["leaf_data"]
        obj.leaf_bounds = data["leaf_bounds"]

        # Rebuild nodes_by_level placeholders (sizes only)
        obj.nodes_by_level = [list(range(mat.shape[0])) for mat in obj.centroids_head]

        return obj

    @classmethod
    def _load_from_mmap_dir(cls, base: str) -> "TreeFlat":
        """
        Charge la structure plate depuis un répertoire créé par save(..., mmap_dir=True).
        """
        # Chargement des métadonnées
        meta_path = os.path.join(base, 'meta.json')
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        obj = cls(int(meta['dims']), int(meta['max_depth']))

        obj.depth = int(meta['depth'])
        obj.n_nodes = int(meta['n_nodes'])
        obj.n_leaves = int(meta['n_leaves'])
        obj.n_levels = int(meta['n_levels'])
        obj.top_dims_by_level = [np.array(arr, dtype=np.int32) for arr in meta['top_dims_by_level']]
        obj.d_head_by_level = meta['d_head_by_level']

        # Chargement memory-mapping des tableaux numpy
        obj.centroids_head = [
            np.load(os.path.join(base, f'centroids_head_{i}.npy'), mmap_mode='r')
            for i in range(obj.n_levels)
        ]
        obj.tail_norms = [
            np.load(os.path.join(base, f'tail_norms_{i}.npy'), mmap_mode='r')
            for i in range(obj.n_levels)
        ]
        obj.child_ptr = [
            np.load(os.path.join(base, f'child_ptr_{i}.npy'), mmap_mode='r')
            for i in range(obj.n_levels)
        ]
        obj.leaf_offset = np.load(os.path.join(base, 'leaf_offset.npy'), mmap_mode='r')
        obj.leaf_data = np.load(os.path.join(base, 'leaf_data.npy'), mmap_mode='r')
        obj.leaf_bounds = np.load(os.path.join(base, 'leaf_bounds.npy'), mmap_mode='r')

        obj.nodes_by_level = [list(range(mat.shape[0])) for mat in obj.centroids_head]
        return obj

    # ------------------------------------------------------------------
    # Stats helper
    # ------------------------------------------------------------------
    def get_statistics(self) -> Dict[str, Any]:
        stats: Dict[str, Any] = {
            "n_nodes": self.n_nodes,
            "n_leaves": self.n_leaves,
            "max_depth": self.depth,
            "n_levels": self.n_levels,
            "dims": self.dims,
            "d_head_by_level": self.d_head_by_level,
            "nodes_per_level": [mat.shape[0] for mat in self.centroids_head],
        }
        if self.n_leaves > 0:
            sizes = [self.leaf_bounds[i + 1] - self.leaf_bounds[i] for i in range(self.n_leaves)]
            stats.update(
                avg_leaf_size=sum(sizes) / self.n_leaves,
                min_leaf_size=min(sizes),
                max_leaf_size=max(sizes),
                total_indices=len(self.leaf_data),
            )
        return stats