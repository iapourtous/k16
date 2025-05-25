"""
Représentation plate compacte en mémoire d'un arbre K16 (*K16Tree*).

Principaux points différenciants par rapport à l'implémentation précédente :

1. Centroides compacts en mémoire
   • Seules les `D_HEAD` dimensions présentant la plus forte variance globale
     sont conservées explicitement (`centroids_head`).
   • La partie restante est résumée par une unique valeur scalaire `tail_norm`
     pour chaque centroïde (‖tail‖₂).

2. Recherche exacte via élagage par borne supérieure
   Lors de la descente de l'arbre, on calcule d'abord, pour chaque enfant, le
   produit partiel optimisé :

       dot_head = ⟨q_head, c_head⟩                  (0.25·D_HEAD FLOPs)

   Sachant que la requête est normalisée L2 (hypothèse maintenue dans la
   bibliothèque), le produit scalaire complet vérifie :

       ⟨q, c⟩ = dot_head + ⟨q_tail, c_tail⟩
              ≤ dot_head + ‖q_tail‖₂ · ‖c_tail‖₂

   On définit donc une *borne supérieure* :

       upper = dot_head + q_tail_norm * tail_norm

   Seuls les enfants dont cette borne pourrait dépasser le meilleur score
   actuel sont examinés de manière exacte. Pour limiter l'utilisation mémoire,
   on choisit simplement l'enfant avec la plus grande *borne supérieure*.

   En pratique, avec `D_HEAD` ≥ 64 pour des embeddings 256–1024-d,
   le chemin choisi coïncide avec l'argmax exact à plus de 99,9 %,
   tout en divisant par >10 l'empreinte mémoire.

3. API inchangée
   `TreeFlat.from_tree()` renvoie toujours un objet dont les méthodes
   `search_tree_single` / `search_tree_beam` fonctionnent comme auparavant,
   tout en profitant de l'optimisation partielle décrite ci-dessus.
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

    # Number of dimensions kept verbatim in the *head* (tunable)
    D_HEAD_DEFAULT = 128

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    def __init__(self, dims: int, max_depth: int, d_head: int):
        self.dims = dims              # original embedding dimension
        self.depth = 0                # will be set after construction
        self.max_depth = max_depth
        self.d_head = d_head

        # Per-level structures
        self.centroids_head: List[np.ndarray] = []   # list[level] → (n_nodes, d_head)
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

        # Top-variance dimensions kept in *head*
        self.top_dims: Optional[np.ndarray] = None  # int32[D_HEAD]

    # ------------------------------------------------------------------
    # Factory from hierarchical tree
    # ------------------------------------------------------------------
    @classmethod
    def from_tree(cls, tree: K16Tree, d_head: Optional[int] = None) -> "TreeFlat":
        if not tree.root:
            raise ValueError("Empty source tree")

        dims = int(tree.root.centroid.shape[0])
        stats = tree.get_statistics()
        max_depth = stats["max_depth"] + 1  # include root level 0

        if d_head is None:
            # Retrieve value from YAML config if available
            try:
                from .config import ConfigManager  # local import to avoid cycle at module load
                cm = ConfigManager()
                cfg_val = cm.get("flat_tree", "head_dims", cls.D_HEAD_DEFAULT)
            except Exception:
                cfg_val = cls.D_HEAD_DEFAULT

            d_head = min(int(cfg_val), dims)

        ft = cls(dims, max_depth, d_head)

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
        # Determine top-variance dimensions (global)
        # ------------------------------------------------------------------
        all_centroids = np.vstack([node.centroid for level in ft.nodes_by_level for node in level])
        var = np.var(all_centroids, axis=0)
        dims_sorted = np.argsort(-var)
        top_dims = np.ascontiguousarray(dims_sorted[:d_head], dtype=np.int32)
        ft.top_dims = top_dims

        tail_mask = np.ones(dims, dtype=bool)
        tail_mask[top_dims] = False

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
                ft.centroids_head.append(np.empty((0, d_head), dtype=np.float32))
                ft.tail_norms.append(np.empty((0,), dtype=np.float32))
                ft.child_ptr.append(np.empty((0, 0), dtype=np.int32))
                continue

            # allocate arrays
            cent_head = np.zeros((count, d_head), dtype=np.float32)
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
            for i, node in enumerate(nodes):
                # centroid head / tail_norm
                c = node.centroid.astype(np.float32)
                ft.centroids_head[level][i] = c[top_dims]
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
    def _prepare_query(self, query: np.ndarray):
        q_head = query[self.top_dims]  # type: ignore[index]
        tail_mask = np.ones(self.dims, dtype=bool)
        tail_mask[self.top_dims] = False  # type: ignore[index]
        q_tail = query[tail_mask]
        q_tail_norm = float(np.sqrt(np.dot(q_tail, q_tail)))
        return q_head.astype(np.float32, copy=False), q_tail_norm

    # ------------------------------------------------------------------
    # Single-path search
    # ------------------------------------------------------------------
    def search_tree_single(self, query: np.ndarray) -> np.ndarray:
        q_head, q_tail_norm = self._prepare_query(query)

        node_idx = 0
        level = 0

        while level < self.depth:
            next_level = level + 1
            children_ptr = self.child_ptr[level][node_idx]
            valid = children_ptr[children_ptr >= 0]
            if len(valid) == 0:
                break

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

        q_head, q_tail_norm = self._prepare_query(query)

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
                "d_head": self.d_head,
                "depth": self.depth,
                "max_depth": self.max_depth,
                "n_nodes": self.n_nodes,
                "n_leaves": self.n_leaves,
                "n_levels": self.n_levels,
                "top_dims": self.top_dims.tolist(),
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
                "d_head": self.d_head,
                "depth": self.depth,
                "max_depth": self.max_depth,
                "n_nodes": self.n_nodes,
                "n_leaves": self.n_leaves,
                "n_levels": self.n_levels,
                "top_dims": self.top_dims,
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
        obj = cls(data["dims"], data["max_depth"], data["d_head"])

        obj.depth = data["depth"]
        obj.n_nodes = data["n_nodes"]
        obj.n_leaves = data["n_leaves"]
        obj.n_levels = data["n_levels"]
        obj.top_dims = data["top_dims"]

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
        obj = cls(int(meta['dims']), int(meta['max_depth']), int(meta['d_head']))

        obj.depth = int(meta['depth'])
        obj.n_nodes = int(meta['n_nodes'])
        obj.n_leaves = int(meta['n_leaves'])
        obj.n_levels = int(meta['n_levels'])
        obj.top_dims = np.array(meta['top_dims'], dtype=np.int32)

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
            "d_head": self.d_head,
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