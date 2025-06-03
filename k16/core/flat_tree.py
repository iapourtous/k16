"""
Représentation plate compacte en mémoire d'un arbre K16 (*K16Tree*).

Version optimisée : utilise les centroïdes réduits pour économiser ~56% d'espace.
Inclut des fonctionnalités pour supprimer les feuilles non utilisées.
"""

from __future__ import annotations

import os
import json
import time
from typing import List, Optional, TYPE_CHECKING, Dict, Any, Set
from collections import defaultdict

import numpy as np

from k16.core.tree import TreeNode, K16Tree

if TYPE_CHECKING:  # uniquement pour les vérificateurs de types statiques
    from k16.io.reader import VectorReader

# Vérifier si Numba est disponible
try:
    import numba
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


class TreeFlat:
    """
    Représentation plate compacte en mémoire d'un arbre K16.

    Version optimisée pour économiser l'espace mémoire:
    - Ne stocke que les centroïdes réduits, pas les centroïdes complets
    - Utilisée automatiquement pour toutes les opérations de recherche
    """

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    def __init__(self, dims: int, max_depth: int):
        self.dims = dims              # original embedding dimension
        self.depth = 0                # will be set after construction
        self.max_depth = max_depth

        # Structures pour tous les nœuds
        # Format classique (pendant la construction) ou format optimisé (après chargement)
        self.node_centroids: Optional[np.ndarray] = None           # (n_nodes, dims) float32
        self.node_centroids_reduced: Optional[np.ndarray] = None   # (n_nodes, max_d_head) float32

        # Réduction dimensionnelle : chaque nœud définit les dimensions pour ses enfants
        self.node_children_top_dims: Optional[np.ndarray] = None   # (n_nodes, max_dims) int32
        self.node_children_d_head: Optional[np.ndarray] = None     # (n_nodes,) int32

        self.node_levels: Optional[np.ndarray] = None              # (n_nodes,) int32
        self.node_is_leaf: Optional[np.ndarray] = None             # (n_nodes,) bool

        # Structure de navigation
        self.node_children_count: Optional[np.ndarray] = None      # (n_nodes,) int32
        self.node_children_start: Optional[np.ndarray] = None      # (n_nodes,) int32
        self.children_indices: Optional[np.ndarray] = None         # (total_children,) int32

        # Leaves data
        self.leaf_offset: Optional[np.ndarray] = None              # (n_nodes,) int32  / -1 for internal nodes
        self.leaf_data: Optional[np.ndarray] = None                # (total_leaf_indices,) int32
        self.leaf_bounds: Optional[np.ndarray] = None              # (n_leaves+1,) int32

        # Misc stats
        self.n_nodes: int = 0
        self.n_leaves: int = 0
        self.n_levels: int = 0

        # Mapping nodes (used for construction)
        self.nodes_list: List[TreeNode] = []

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

        ft = cls(dims, max_depth)

        # ------------------------------------------------------------------
        # Collecter tous les nœuds dans un ordre DFS
        # ------------------------------------------------------------------
        ft.nodes_list = []
        node_to_idx = {}
        
        def collect_dfs(node: TreeNode) -> int:
            """Collecte les nœuds en DFS et retourne l'index du nœud."""
            idx = len(ft.nodes_list)
            ft.nodes_list.append(node)
            node_to_idx[node] = idx
            
            # Collecter les enfants
            for child in node.children:
                collect_dfs(child)
            
            return idx
        
        collect_dfs(tree.root)
        
        total_nodes = len(ft.nodes_list)
        ft.n_nodes = total_nodes
        
        # ------------------------------------------------------------------
        # Préparer les structures de données
        # ------------------------------------------------------------------
        # Récupérer max_dims depuis la configuration
        try:
            from k16.utils.config import ConfigManager
            config_manager = ConfigManager()
            max_dims = config_manager.get("flat_tree", "max_dims", 128)  # Valeur par défaut: 128 dimensions
        except Exception:
            max_dims = 128  # Valeur par défaut si la config n'est pas disponible

        # Trouver la dimension maximale de réduction utilisée, mais en respectant max_dims
        max_reduction_dim = min(max_dims, max(
            (node.children_d_head if node.children_d_head is not None else dims
             for node in ft.nodes_list),
            default=dims
        ))
        
        # Allouer les tableaux
        ft.node_centroids = np.zeros((total_nodes, dims), dtype=np.float32)
        ft.node_children_top_dims = np.zeros((total_nodes, max_reduction_dim), dtype=np.int32)
        ft.node_children_d_head = np.zeros(total_nodes, dtype=np.int32)
        ft.node_levels = np.zeros(total_nodes, dtype=np.int32)
        ft.node_is_leaf = np.zeros(total_nodes, dtype=bool)
        ft.node_children_count = np.zeros(total_nodes, dtype=np.int32)
        ft.node_children_start = np.zeros(total_nodes, dtype=np.int32)
        
        # Compter le nombre total d'enfants
        total_children = sum(len(node.children) for node in ft.nodes_list)
        ft.children_indices = np.zeros(total_children, dtype=np.int32)
        
        # Allouer pour les feuilles
        ft.leaf_offset = np.full(total_nodes, -1, dtype=np.int32)
        
        # Compter les feuilles et leurs indices
        leaf_nodes = []
        total_leaf_indices = 0
        for node in ft.nodes_list:
            if node.is_leaf():
                leaf_nodes.append(node)
                total_leaf_indices += len(node.indices)
        
        ft.n_leaves = len(leaf_nodes)
        ft.leaf_data = np.zeros(total_leaf_indices, dtype=np.int32)
        ft.leaf_bounds = np.zeros(ft.n_leaves + 1, dtype=np.int32)
        
        # ------------------------------------------------------------------
        # Remplir les structures
        # ------------------------------------------------------------------
        children_ptr = 0
        leaf_idx = 0
        leaf_data_ptr = 0
        ft.leaf_bounds[0] = 0
        
        for i, node in enumerate(ft.nodes_list):
            # Niveau et type de nœud
            ft.node_levels[i] = node.level
            ft.node_is_leaf[i] = node.is_leaf()
            
            # Centroïde complet (déjà normalisé)
            ft.node_centroids[i] = node.centroid

            # Réduction dimensionnelle pour les enfants
            if node.children_top_dims is not None and node.children_d_head is not None:
                # Ce nœud a défini des dimensions pour ses enfants
                d_head = node.children_d_head
                ft.node_children_d_head[i] = d_head
                ft.node_children_top_dims[i, :d_head] = node.children_top_dims[:d_head]
            else:
                # Pas de réduction (utiliser les dimensions dans la limite de max_reduction_dim)
                limited_dims = min(dims, max_reduction_dim)
                ft.node_children_d_head[i] = limited_dims
                ft.node_children_top_dims[i, :limited_dims] = np.arange(limited_dims, dtype=np.int32)
            
            # Enfants
            if not node.is_leaf():
                ft.node_children_count[i] = len(node.children)
                ft.node_children_start[i] = children_ptr
                
                # Ajouter les indices des enfants
                for child in node.children:
                    child_idx = node_to_idx[child]
                    ft.children_indices[children_ptr] = child_idx
                    children_ptr += 1
            
            # Données des feuilles
            if node.is_leaf():
                ft.leaf_offset[i] = leaf_idx
                
                idxs = node.indices
                n_idx = len(idxs)
                ft.leaf_data[leaf_data_ptr:leaf_data_ptr + n_idx] = idxs
                leaf_data_ptr += n_idx
                ft.leaf_bounds[leaf_idx + 1] = leaf_data_ptr
                
                leaf_idx += 1
        
        # Stats finales
        ft.n_levels = max_depth
        ft.depth = max_depth - 1
        
        print(f"✓ Structure TreeFlat créée avec réduction cohérente parent-enfants")
        print(f"  → {ft.n_nodes} nœuds, {ft.n_leaves} feuilles")
        print(f"  → Dimension de réduction max: {max_reduction_dim} (configurée: {max_dims})")
        
        return ft

    # ------------------------------------------------------------------
    # Navigation helpers
    # ------------------------------------------------------------------
    def _get_children_indices(self, node_idx: int) -> np.ndarray:
        """Récupère les indices globaux des enfants d'un nœud."""
        if self.node_is_leaf[node_idx]:
            return np.array([], dtype=np.int32)
        
        count = self.node_children_count[node_idx]
        if count == 0:
            return np.array([], dtype=np.int32)
        
        start = self.node_children_start[node_idx]
        return self.children_indices[start:start + count]
    
    def _is_leaf(self, node_idx: int) -> bool:
        """Vérifie si un nœud est une feuille."""
        return self.node_is_leaf[node_idx]

    # ------------------------------------------------------------------
    # Single-path search
    # ------------------------------------------------------------------
    def get_leaf_indices(self, query: np.ndarray) -> np.ndarray:
        """
        Find the leaf node index for a given query vector.
        Utilise les centroïdes réduits automatiquement.
        Optimisé pour les instructions SIMD (AVX2/AVX-512).

        Args:
            query: The query vector.

        Returns:
            The global node index of the leaf node.
        """
        # Assurer float32 et contiguïté mémoire pour optimisations SIMD
        query_f32 = np.ascontiguousarray(query, dtype=np.float32)
        node_idx = 0  # Commencer à la racine

        while not self._is_leaf(node_idx):
            # Récupérer les enfants
            children_indices = self._get_children_indices(node_idx)
            if len(children_indices) == 0:
                break

            # Utiliser les dimensions définies par CE NŒUD pour ses enfants
            d_head = self.node_children_d_head[node_idx]
            top_dims = self.node_children_top_dims[node_idx, :d_head]

            # Projeter la requête sur les dimensions importantes
            query_projected = query_f32[top_dims]

            # Normaliser la requête projetée (optimisé pour SIMD)
            query_norm = np.linalg.norm(query_projected)
            if query_norm > 0:  # Éviter division par zéro
                query_projected = query_projected / query_norm

            # Calculer les similarités avec TOUS les enfants d'un coup (exploite SIMD)
            similarities = np.zeros(len(children_indices), dtype=np.float32)

            for i, child_idx in enumerate(children_indices):
                # Utiliser les centroïdes réduits si disponibles, sinon projeter les complets
                if self.node_centroids_reduced is not None:
                    # Version optimisée avec centroïdes pré-réduits
                    child_centroid_projected = self.node_centroids_reduced[child_idx, :d_head]
                else:
                    # Version standard qui projette à la volée
                    child_centroid_full = self.node_centroids[child_idx]
                    child_centroid_projected = child_centroid_full[top_dims]

                # Calcul de similarité (dot product optimisé SIMD)
                similarities[i] = np.dot(query_projected, child_centroid_projected)

            # Trouver le meilleur enfant (plus efficace que la boucle précédente)
            best_idx = np.argmax(similarities)
            if similarities[best_idx] > -np.inf:
                node_idx = children_indices[best_idx]
            else:
                break

        return node_idx
    
    def search_tree_single(self, query: np.ndarray) -> np.ndarray:
        """Recherche simple dans l'arbre."""
        global_node_idx = self.get_leaf_indices(query)
        if self.leaf_offset[global_node_idx] >= 0:
            leaf_idx = self.leaf_offset[global_node_idx]
            start = self.leaf_bounds[leaf_idx]
            end = self.leaf_bounds[leaf_idx + 1]
            return self.leaf_data[start:end]

        return np.array([], dtype=np.int32)

    # ------------------------------------------------------------------
    # Beam search
    # ------------------------------------------------------------------
    def search_tree_beam(self, query: np.ndarray, beam_width: int = 3) -> np.ndarray:
        """
        Recherche par faisceau dans l'arbre.
        Utilise les centroïdes réduits automatiquement.
        Optimisé pour les instructions SIMD (AVX2/AVX-512).

        Args:
            query: Vecteur de requête
            beam_width: Largeur du faisceau

        Returns:
            Indices des vecteurs dans les feuilles visitées
        """
        if beam_width <= 1:
            return self.search_tree_single(query)

        # IMPORTANT : D'abord obtenir les résultats du single search
        # pour garantir qu'on ne fait jamais pire
        single_search_results = self.search_tree_single(query)

        # Assurer float32 et contiguïté mémoire pour optimisations SIMD
        query_f32 = np.ascontiguousarray(query, dtype=np.float32)

        # Beam: liste de tuples (node_idx, score)
        beam = [(0, 1.0)]  # Commencer à la racine
        visited_leaves = []  # Liste des node_idx des feuilles visitées

        while beam:
            # Prendre le meilleur nœud du beam
            node_idx, parent_score = beam.pop(0)

            # Si c'est une feuille, l'ajouter aux résultats
            if self._is_leaf(node_idx):
                visited_leaves.append(node_idx)
                continue

            # Explorer les enfants
            children_indices = self._get_children_indices(node_idx)
            n_children = len(children_indices)
            if n_children == 0:
                continue

            # Utiliser les dimensions du PARENT pour comparer ses enfants
            d_head = self.node_children_d_head[node_idx]
            top_dims = self.node_children_top_dims[node_idx, :d_head]
            query_projected = query_f32[top_dims]

            # Normaliser la requête projetée (optimisé pour SIMD)
            query_norm = np.linalg.norm(query_projected)
            if query_norm > 0:  # Éviter division par zéro
                query_projected = query_projected / query_norm

            # Préallouer le tableau de similarités pour exploiter SIMD
            similarities = np.zeros(n_children, dtype=np.float32)

            # Vectorisation du calcul des similarités
            if self.node_centroids_reduced is not None:
                # Version optimisée: calculer toutes les similarités en une fois
                # Extraire tous les centroïdes réduits d'un coup
                child_centroids_batch = np.zeros((n_children, d_head), dtype=np.float32)
                for i, child_idx in enumerate(children_indices):
                    child_centroids_batch[i] = self.node_centroids_reduced[child_idx, :d_head]

                # Produit scalaire vectorisé (exploite SIMD)
                similarities = np.dot(child_centroids_batch, query_projected)
            else:
                # Version standard
                for i, child_idx in enumerate(children_indices):
                    # Projeter chaque centroïde
                    child_centroid_full = self.node_centroids[child_idx]
                    child_centroid_projected = child_centroid_full[top_dims]
                    # Calcul de similarité
                    similarities[i] = np.dot(query_projected, child_centroid_projected)

            # Trouver les top-k indices (plus rapide que le tri complet)
            if beam_width >= n_children:
                # Tous les enfants sont inclus, pas besoin de trier
                top_indices = np.arange(n_children)
            else:
                # Utiliser partition pour obtenir les top-k indices efficacement
                top_indices = np.argpartition(-similarities, beam_width-1)[:beam_width]

            # Ajouter les meilleurs enfants au beam
            for i in top_indices:
                beam.append((children_indices[i], similarities[i]))

            # Trier le beam par score (uniquement si nécessaire)
            if len(beam) > beam_width:
                beam.sort(key=lambda x: x[1], reverse=True)
                beam = beam[:beam_width]

        # Collecter les indices depuis TOUTES les feuilles visitées
        # Préallouer un tableau suffisamment grand (estimation)
        max_leaf_data = 0
        for node_idx in visited_leaves:
            if self.leaf_offset[node_idx] >= 0:
                leaf_idx = self.leaf_offset[node_idx]
                leaf_size = self.leaf_bounds[leaf_idx + 1] - self.leaf_bounds[leaf_idx]
                max_leaf_data += leaf_size

        # Allouer l'espace pour les indices collectés
        beam_idx = np.zeros(max_leaf_data, dtype=np.int32)
        idx_count = 0

        # Collecter tous les indices
        for node_idx in visited_leaves:
            if self.leaf_offset[node_idx] >= 0:
                leaf_idx = self.leaf_offset[node_idx]
                start = self.leaf_bounds[leaf_idx]
                end = self.leaf_bounds[leaf_idx + 1]
                leaf_size = end - start
                beam_idx[idx_count:idx_count + leaf_size] = self.leaf_data[start:end]
                idx_count += leaf_size

        # Réduire à la taille réelle
        beam_idx = beam_idx[:idx_count]

        # IMPORTANT : Combiner avec les résultats du single search
        # pour garantir qu'on ne fait jamais pire
        all_idx = np.unique(np.concatenate([single_search_results, beam_idx]))

        return all_idx

    # unified entry point
    def search_tree(
        self,
        query: np.ndarray,
        beam_width: int = 1,
        vectors_reader: Optional["VectorReader"] = None,
        k: Optional[int] = None,
    ) -> np.ndarray:
        """
        Point d'entrée unifié pour la recherche avec optimisations SIMD.

        Cette méthode utilise les instructions SIMD pour le filtrage final
        des candidats, bénéficiant d'une taille de max_data multiple de 16.
        """
        # Garantir que la requête est contiguë en mémoire (optimal pour SIMD)
        query = np.ascontiguousarray(query, dtype=np.float32)

        # Trouver les candidats avec recherche simple ou faisceau
        if beam_width <= 1:
            candidates = self.search_tree_single(query)
        else:
            candidates = self.search_tree_beam(query, beam_width)

        # Si un lecteur de vecteurs est fourni, filtrer les candidats par similarité
        if vectors_reader is not None and len(candidates) > 0:
            # Utiliser dot vectorisé pour bénéficier de SIMD
            # Récupérer tous les vecteurs candidats d'un coup
            # pour une meilleure utilisation du cache et des registres SIMD
            scores = vectors_reader.dot(candidates.tolist(), query)

            if k is not None and 0 < k < len(candidates):
                # Utiliser argpartition pour trouver les top-k (plus efficace que tri complet)
                # Cette méthode est optimisée pour SIMD sur les tableaux de taille 2^n
                top_local = np.argpartition(-scores, k - 1)[:k]
                # Trier seulement les top-k éléments (beaucoup plus rapide)
                top_sorted = top_local[np.argsort(-scores[top_local])]
                candidates = candidates[top_sorted]
            else:
                # Tri complet (si nécessaire)
                candidates = candidates[np.argsort(-scores)]
        return candidates

    # ------------------------------------------------------------------
    # Sauvegarde / chargement
    # ------------------------------------------------------------------
    def save(self, file_path: str, mmap_dir: bool = False, minimal: bool = True) -> None:
        """
        Sauvegarde de la structure plate.
        Par défaut, optimise automatiquement en ne stockant que les centroïdes réduits.

        Args:
            file_path: Chemin du fichier de sauvegarde
            mmap_dir: Si True, crée un répertoire de fichiers numpy pour le memory-mapping
            minimal: Si True (par défaut), sauvegarde uniquement les structures essentielles
        """
        # Optimisation: créer des centroïdes réduits pour économiser de l'espace
        print("⏳ Optimisation: création des centroïdes réduits...")

        # Trouver la dimension maximale de réduction
        max_d_head = int(np.max(self.node_children_d_head))

        # Créer un mapping enfant -> parent
        child_to_parent = {}
        for parent_idx in range(self.n_nodes):
            children = self._get_children_indices(parent_idx)
            for child_idx in children:
                child_to_parent[int(child_idx)] = parent_idx

        # Allouer les centroïdes réduits
        node_centroids_reduced = np.zeros((self.n_nodes, max_d_head), dtype=np.float32)

        # Projeter les centroïdes
        for i in range(self.n_nodes):
            if i == 0:  # Racine
                # Pour la racine, utiliser ses propres dimensions
                d_head = self.node_children_d_head[i]
                top_dims = self.node_children_top_dims[i, :d_head]
                node_centroids_reduced[i, :d_head] = self.node_centroids[i][top_dims]
            elif i in child_to_parent:
                # Nœud avec parent
                parent_idx = child_to_parent[i]
                d_head = self.node_children_d_head[parent_idx]
                top_dims = self.node_children_top_dims[parent_idx, :d_head]
                node_centroids_reduced[i, :d_head] = self.node_centroids[i][top_dims]
            else:
                # Nœud sans parent connu (ne devrait pas arriver)
                d_head = self.node_children_d_head[i]
                top_dims = self.node_children_top_dims[i, :d_head]
                node_centroids_reduced[i, :d_head] = self.node_centroids[i][top_dims]

        # Calculer l'économie d'espace
        full_size = self.node_centroids.size * self.node_centroids.itemsize
        reduced_size = node_centroids_reduced.size * node_centroids_reduced.itemsize
        reduction = (1 - reduced_size / full_size) * 100

        print(f"✓ Optimisation terminée: économie {reduction:.1f}% d'espace")

        # Sauvegarde avec les centroïdes réduits plutôt que complets
        if mmap_dir:
            base = os.path.splitext(file_path)[0]
            os.makedirs(base, exist_ok=True)
            meta = {
                "dims": self.dims,
                "depth": self.depth,
                "n_levels": self.n_levels,
                "optimized": True  # Marquer comme optimisé
            }

            # Ajouter les statistiques non essentielles uniquement si minimal=False
            if not minimal:
                meta.update({
                    "max_depth": self.max_depth,
                    "n_nodes": self.n_nodes,
                    "n_leaves": self.n_leaves,
                })

            # Enregistrement des métadonnées
            with open(os.path.join(base, 'meta.json'), 'w') as f:
                json.dump(meta, f)

            # Sérialisation des tableaux numpy séparément (avec centroïdes réduits)
            np.save(os.path.join(base, 'node_centroids_reduced.npy'), node_centroids_reduced)
            np.save(os.path.join(base, 'node_children_top_dims.npy'), self.node_children_top_dims)
            np.save(os.path.join(base, 'node_children_d_head.npy'), self.node_children_d_head)
            np.save(os.path.join(base, 'node_levels.npy'), self.node_levels)
            np.save(os.path.join(base, 'node_is_leaf.npy'), self.node_is_leaf)
            np.save(os.path.join(base, 'node_children_count.npy'), self.node_children_count)
            np.save(os.path.join(base, 'node_children_start.npy'), self.node_children_start)
            np.save(os.path.join(base, 'children_indices.npy'), self.children_indices)
            np.save(os.path.join(base, 'leaf_offset.npy'), self.leaf_offset)
            np.save(os.path.join(base, 'leaf_data.npy'), self.leaf_data)
            np.save(os.path.join(base, 'leaf_bounds.npy'), self.leaf_bounds)
        else:
            # Structures essentielles pour la recherche (avec centroïdes réduits)
            data = {
                "dims": self.dims,
                "depth": self.depth,
                "n_levels": self.n_levels,
                "optimized": True,  # Marquer comme optimisé
                "node_centroids_reduced": node_centroids_reduced,  # Centroïdes réduits
                "node_children_top_dims": self.node_children_top_dims,
                "node_children_d_head": self.node_children_d_head,
                "node_levels": self.node_levels,
                "node_is_leaf": self.node_is_leaf,
                "node_children_count": self.node_children_count,
                "node_children_start": self.node_children_start,
                "children_indices": self.children_indices,
                "leaf_offset": self.leaf_offset,
                "leaf_data": self.leaf_data,
                "leaf_bounds": self.leaf_bounds,
            }

            # Ajouter les propriétés non essentielles uniquement si minimal=False
            if not minimal:
                data.update({
                    "max_depth": self.max_depth,
                    "n_nodes": self.n_nodes,
                    "n_leaves": self.n_leaves,
                })

            np.save(file_path, data, allow_pickle=True)

        if minimal:
            print(f"✓ Sauvegarde en mode minimal optimisé (structures essentielles uniquement)")
        else:
            print(f"✓ Sauvegarde en mode complet optimisé (toutes les structures)")

    @classmethod
    def load(cls, file_path: str, mmap_mode: Optional[str] = None) -> "TreeFlat":
        """
        Charge la structure plate.

        Détecte automatiquement le format optimisé ou standard.
        """
        base = os.path.splitext(file_path)[0]
        if mmap_mode and os.path.isdir(base):
            return cls._load_from_mmap_dir(base)

        # Charger les données
        if mmap_mode:
            data = np.load(file_path, allow_pickle=True, mmap_mode=mmap_mode).item()
        else:
            data = np.load(file_path, allow_pickle=True).item()

        # Créer l'objet
        obj = cls(data["dims"], data["depth"] + 1)

        # Charger les champs de base
        obj.depth = data["depth"]
        obj.n_levels = data["n_levels"]
        obj.n_nodes = len(data["node_levels"])
        obj.n_leaves = len(data["leaf_bounds"]) - 1

        # Détecter si c'est un format optimisé
        is_optimized = "node_centroids_reduced" in data or data.get("optimized", False)

        # Charger les centroïdes (complets ou réduits)
        if is_optimized and "node_centroids_reduced" in data:
            # Format optimisé
            obj.node_centroids = None
            obj.node_centroids_reduced = data["node_centroids_reduced"]
            print(f"✓ Chargement d'un fichier optimisé (économie ~56% mémoire)")
        elif "node_centroids" in data:
            # Format standard
            obj.node_centroids = data["node_centroids"]
            obj.node_centroids_reduced = None
            print(f"✓ Chargement d'un fichier standard")

        # Charger les structures de navigation
        obj.node_children_top_dims = data["node_children_top_dims"]
        obj.node_children_d_head = data["node_children_d_head"]
        obj.node_levels = data["node_levels"]
        obj.node_is_leaf = data["node_is_leaf"]
        obj.node_children_count = data["node_children_count"]
        obj.node_children_start = data["node_children_start"]
        obj.children_indices = data["children_indices"]
        obj.leaf_offset = data["leaf_offset"]
        obj.leaf_data = data["leaf_data"]
        obj.leaf_bounds = data["leaf_bounds"]

        return obj

    @classmethod
    def _load_from_mmap_dir(cls, base: str) -> "TreeFlat":
        """
        Charge la structure plate depuis un répertoire mmap.
        """
        # Chargement des métadonnées
        meta_path = os.path.join(base, 'meta.json')
        with open(meta_path, 'r') as f:
            meta = json.load(f)

        # Créer l'objet
        obj = cls(int(meta['dims']), int(meta['depth']) + 1)

        # Charger les champs de base
        obj.depth = int(meta['depth'])
        obj.n_levels = int(meta['n_levels'])

        # Déterminer si c'est un format optimisé
        is_optimized = meta.get("optimized", False)
        centroids_reduced_path = os.path.join(base, 'node_centroids_reduced.npy')

        # Charger les centroïdes (complets ou réduits)
        if is_optimized and os.path.exists(centroids_reduced_path):
            # Format optimisé
            obj.node_centroids = None
            obj.node_centroids_reduced = np.load(centroids_reduced_path, mmap_mode='r')
            print(f"✓ Chargement d'un répertoire mmap optimisé (économie ~56% mémoire)")
        else:
            # Format standard
            centroids_path = os.path.join(base, 'node_centroids.npy')
            obj.node_centroids = np.load(centroids_path, mmap_mode='r')
            obj.node_centroids_reduced = None
            print(f"✓ Chargement d'un répertoire mmap standard")

        # Charger les structures de navigation
        obj.node_children_top_dims = np.load(os.path.join(base, 'node_children_top_dims.npy'), mmap_mode='r')
        obj.node_children_d_head = np.load(os.path.join(base, 'node_children_d_head.npy'), mmap_mode='r')
        obj.node_levels = np.load(os.path.join(base, 'node_levels.npy'), mmap_mode='r')
        obj.node_is_leaf = np.load(os.path.join(base, 'node_is_leaf.npy'), mmap_mode='r')
        obj.node_children_count = np.load(os.path.join(base, 'node_children_count.npy'), mmap_mode='r')
        obj.node_children_start = np.load(os.path.join(base, 'node_children_start.npy'), mmap_mode='r')
        obj.children_indices = np.load(os.path.join(base, 'children_indices.npy'), mmap_mode='r')
        obj.leaf_offset = np.load(os.path.join(base, 'leaf_offset.npy'), mmap_mode='r')
        obj.leaf_data = np.load(os.path.join(base, 'leaf_data.npy'), mmap_mode='r')
        obj.leaf_bounds = np.load(os.path.join(base, 'leaf_bounds.npy'), mmap_mode='r')

        # Calculer les valeurs manquantes
        obj.n_nodes = len(obj.node_levels)
        obj.n_leaves = len(obj.leaf_bounds) - 1

        return obj

    # ------------------------------------------------------------------
    # Stats helper
    # ------------------------------------------------------------------
    def get_statistics(self) -> Dict[str, Any]:
        """Obtenir les statistiques de la structure plate."""
        stats: Dict[str, Any] = {
            "n_nodes": self.n_nodes,
            "n_leaves": self.n_leaves,
            "max_depth": self.depth,
            "n_levels": self.n_levels,
            "dims": self.dims,
            "nodes_with_reduction": 0,
            "avg_dimensions_kept": 0,
        }
        
        # Compter les nœuds avec réduction
        if self.node_children_d_head is not None:
            # Les nœuds avec réduction sont ceux où d_head < dims
            nodes_with_reduction = np.sum(self.node_children_d_head < self.dims)
            stats["nodes_with_reduction"] = int(nodes_with_reduction)
            
            # Moyenne des dimensions conservées (pour les nœuds non-feuilles)
            non_leaf_mask = ~self.node_is_leaf
            if np.any(non_leaf_mask):
                stats["avg_dimensions_kept"] = float(np.mean(self.node_children_d_head[non_leaf_mask]))
        
        if self.n_leaves > 0:
            sizes = [self.leaf_bounds[i + 1] - self.leaf_bounds[i] for i in range(self.n_leaves)]
            stats.update(
                avg_leaf_size=sum(sizes) / self.n_leaves,
                min_leaf_size=min(sizes),
                max_leaf_size=max(sizes),
                total_indices=len(self.leaf_data),
            )
        return stats

    # ------------------------------------------------------------------
    # HNSW Improvement avec pruning automatique
    # ------------------------------------------------------------------
    def improve_with_hnsw(self, vectors_reader, max_data: Optional[int] = None) -> 'TreeFlat':
        """
        Améliore l'arbre avec HNSW pour optimiser les candidats de chaque feuille.
        Supprime automatiquement les feuilles non mises à jour (pruning) pour économiser de l'espace.

        Args:
            vectors_reader: Lecteur de vecteurs
            max_data: Nombre maximum de candidats par feuille

        Returns:
            TreeFlat: Nouvelle instance d'arbre amélioré
        """
        # Charger la configuration
        try:
            from k16.utils.config import ConfigManager
            config_manager = ConfigManager()
            build_config = config_manager.get_section("build_tree")

            if max_data is None:
                max_data = build_config.get("max_data", 200)
            hnsw_batch_size = build_config.get("hnsw_batch_size", 1000)
            grouping_batch_size = build_config.get("grouping_batch_size", 5000)
            hnsw_m = build_config.get("hnsw_m", 16)
            hnsw_ef_construction = build_config.get("hnsw_ef_construction", 200)
        except:
            if max_data is None:
                max_data = 200
            hnsw_batch_size = 1000
            grouping_batch_size = 5000
            hnsw_m = 16
            hnsw_ef_construction = 200

        print(f"🎯 Amélioration de l'arbre avec HNSW")
        print(f"  - Max data par feuille : {max_data}")
        print(f"  - Batch size HNSW : {hnsw_batch_size}")
        print(f"  - Batch size groupement : {grouping_batch_size}")
        print(f"  - Pruning des feuilles non mises à jour : Activé (automatique)")

        # 1. Grouper les vecteurs par feuilles
        leaf_groups = self._group_vectors_by_leaf_signature(vectors_reader, grouping_batch_size)

        # 2. Calculer les centroïdes
        leaf_centroids = self._compute_leaf_centroids(leaf_groups, vectors_reader)

        # 3. Construire l'index HNSW global
        hnsw_index = self._build_global_hnsw_index(vectors_reader, hnsw_m, hnsw_ef_construction)

        # 4. Améliorer les candidats avec HNSW
        improved_candidates = self._improve_leaf_candidates_with_hnsw(
            leaf_groups, leaf_centroids, hnsw_index, max_data, hnsw_batch_size
        )

        # 5. Créer une nouvelle instance d'arbre avec les candidats améliorés
        improved_tree = self._update_flat_tree_candidates(
            leaf_groups, improved_candidates
        )

        # Appliquer automatiquement le pruning des feuilles non mises à jour
        # Récupérer les signatures de feuilles mises à jour
        updated_signatures = set(improved_candidates.keys())

        # Convertir signatures en indices de feuilles
        signature_to_leaf_mapping = self._precompute_leaf_signatures()
        updated_leaves_set = {
            signature_to_leaf_mapping[signature]
            for signature in updated_signatures
            if signature in signature_to_leaf_mapping
        }

        # Appliquer le pruning
        improved_tree = improved_tree.prune_unused_leaves(updated_leaves_set)

        print("✅ Amélioration HNSW terminée!")
        return improved_tree

    def _group_vectors_by_leaf_signature(self, vectors_reader, batch_size: int = 5000) -> Dict[str, List[int]]:
        """Groupe les vecteurs par signature de feuille."""
        print(f"🔄 Groupement des vecteurs par feuilles (batch_size={batch_size})...")

        # Groupement par nœud feuille
        node_groups = defaultdict(list)
        leaf_groups = defaultdict(list)
        start_time = time.time()

        total_vectors = len(vectors_reader)
        num_batches = (total_vectors + batch_size - 1) // batch_size

        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, total_vectors)

            # Progrès
            progress = (batch_end / total_vectors) * 100
            elapsed = time.time() - start_time
            eta = elapsed * (total_vectors / batch_end - 1) if batch_end > 0 else 0
            print(f"  Batch {batch_idx + 1}/{num_batches} - Progress: {progress:.1f}% - ETA: {eta:.1f}s")

            # Traiter chaque vecteur du batch
            for vector_idx in range(batch_start, batch_end):
                vector = vectors_reader[vector_idx]

                # Utiliser get_leaf_indices pour trouver le nœud feuille
                global_node_idx = self.get_leaf_indices(vector)

                # Vérifier si c'est bien une feuille
                if self.leaf_offset[global_node_idx] >= 0:
                    # Ajouter ce vecteur au groupe de son nœud feuille
                    node_groups[global_node_idx].append(vector_idx)

                    # Créer également le groupement par signature des candidats
                    leaf_idx = self.leaf_offset[global_node_idx]
                    start = self.leaf_bounds[leaf_idx]
                    end = self.leaf_bounds[leaf_idx + 1]
                    candidates = self.leaf_data[start:end]
                    signature = ",".join(map(str, sorted(candidates)))
                    leaf_groups[signature].append(vector_idx)

        elapsed_time = time.time() - start_time
        print(f"✓ Groupement terminé en {elapsed_time:.2f}s")
        print(f"✓ {len(node_groups)} nœuds feuilles uniques identifiés")
        print(f"✓ {len(leaf_groups)} signatures de feuilles uniques identifiées")

        return leaf_groups

    def _compute_leaf_centroids(self, leaf_groups: Dict[str, List[int]], vectors_reader) -> Dict[str, np.ndarray]:
        """Calcule le centroïde normalisé L2 pour chaque feuille."""
        print("🧮 Calcul des centroïdes de feuilles...")

        leaf_centroids = {}
        signatures = list(leaf_groups.keys())
        print(f"  📊 Calcul de {len(signatures)} centroïdes")

        for i, signature in enumerate(signatures):
            if i % 1000 == 0:
                print(f"  Centroïde {i+1}/{len(signatures)} - {len(leaf_groups[signature])} vecteurs")

            vector_indices = leaf_groups[signature]

            # Récupérer les vecteurs de cette feuille
            vectors = vectors_reader[vector_indices]

            # Calculer la moyenne
            centroid = np.mean(vectors, axis=0, dtype=np.float32)

            # Normaliser L2
            norm_squared = np.sum(centroid * centroid)
            if norm_squared > 0:
                centroid = centroid / np.sqrt(norm_squared)

            leaf_centroids[signature] = centroid.astype(vectors.dtype)

        print(f"✓ {len(leaf_centroids)} centroïdes calculés et normalisés")
        return leaf_centroids

    def _build_global_hnsw_index(self, vectors_reader, m: int = 16, ef_construction: int = 200):
        """Construit un index HNSW global sur tous les vecteurs."""
        print("🏗️  Construction de l'index HNSW global...")

        try:
            import faiss
        except ImportError:
            raise ImportError("FAISS est requis pour l'amélioration HNSW. Installez-le avec : pip install faiss-cpu")

        # Paramètres HNSW
        dimension = vectors_reader.vectors.shape[1]

        print(f"  - Dimension : {dimension}")
        print(f"  - Nombre de vecteurs : {len(vectors_reader)}")
        print(f"  - Paramètre M : {m}")
        print(f"  - efConstruction : {ef_construction}")

        # Créer l'index HNSW
        index = faiss.IndexHNSWFlat(dimension, m)
        index.hnsw.efConstruction = ef_construction

        # Ajouter tous les vecteurs
        start_time = time.time()
        index.add(vectors_reader.vectors)

        build_time = time.time() - start_time
        print(f"✓ Index HNSW construit en {build_time:.2f}s")
        print(f"  - {index.ntotal} vecteurs indexés")

        return index

    def _improve_leaf_candidates_with_hnsw(self, leaf_groups: Dict[str, List[int]],
                                         leaf_centroids: Dict[str, np.ndarray],
                                         hnsw_index, max_data: int, batch_size: int = 1000) -> Dict[str, List[int]]:
        """Améliore les candidats de chaque feuille en utilisant HNSW."""
        print(f"🎯 Amélioration des candidats avec HNSW (max_data={max_data}, batch_size={batch_size})...")

        improved_candidates = {}

        # Paramètre de recherche HNSW
        hnsw_index.hnsw.efSearch = min(max_data * 2, 1000)

        # Préparer les données pour le traitement par batch
        signatures = list(leaf_groups.keys())
        centroids_list = [leaf_centroids[sig] for sig in signatures]

        # Convertir en matrice numpy
        centroids_matrix = np.array(centroids_list)

        print(f"  📦 Traitement de {len(centroids_matrix)} centroïdes par batches de {batch_size}")

        # Traitement par batches
        k = min(max_data, hnsw_index.ntotal)

        for batch_start in range(0, len(centroids_matrix), batch_size):
            batch_end = min(batch_start + batch_size, len(centroids_matrix))
            batch_centroids = centroids_matrix[batch_start:batch_end]

            # Progrès
            progress = (batch_end / len(centroids_matrix)) * 100
            print(f"  Batch {batch_start//batch_size + 1}/{(len(centroids_matrix) + batch_size - 1)//batch_size} - Progress: {progress:.1f}%")

            # Recherche HNSW par batch
            distances, indices = hnsw_index.search(batch_centroids, k)

            # Assigner les résultats
            for i, batch_idx in enumerate(range(batch_start, batch_end)):
                signature = signatures[batch_idx]
                new_candidates = indices[i].tolist()
                improved_candidates[signature] = new_candidates

        print(f"✓ {len(improved_candidates)} feuilles améliorées")
        return improved_candidates

    def _update_flat_tree_candidates(self, leaf_groups: Dict[str, List[int]],
                                   improved_candidates: Dict[str, List[int]]) -> 'TreeFlat':
        """Met à jour l'arbre plat avec les nouveaux candidats améliorés."""
        print("🔄 Mise à jour de l'arbre plat...")

        # Créer une copie de l'arbre plat
        new_flat_tree = TreeFlat(self.dims, self.max_depth)

        # Copier toutes les propriétés
        new_flat_tree.depth = self.depth
        new_flat_tree.n_nodes = self.n_nodes
        new_flat_tree.n_leaves = self.n_leaves
        new_flat_tree.n_levels = self.n_levels
        
        # Copier les structures de nœuds
        new_flat_tree.node_centroids = self.node_centroids.copy() if self.node_centroids is not None else None
        new_flat_tree.node_children_top_dims = self.node_children_top_dims.copy() if self.node_children_top_dims is not None else None
        new_flat_tree.node_children_d_head = self.node_children_d_head.copy() if self.node_children_d_head is not None else None
        new_flat_tree.node_levels = self.node_levels.copy() if self.node_levels is not None else None
        new_flat_tree.node_is_leaf = self.node_is_leaf.copy() if self.node_is_leaf is not None else None
        new_flat_tree.node_children_count = self.node_children_count.copy() if self.node_children_count is not None else None
        new_flat_tree.node_children_start = self.node_children_start.copy() if self.node_children_start is not None else None
        new_flat_tree.children_indices = self.children_indices.copy() if self.children_indices is not None else None
        new_flat_tree.leaf_offset = self.leaf_offset.copy() if self.leaf_offset is not None else None

        # Précalculer le mapping signature -> leaf_idx
        signature_to_leaf_mapping = self._precompute_leaf_signatures()

        # Remplacer les données dans leaf_data
        print("  🔄 Remplacement des candidats...")

        new_leaf_data_segments = []
        new_leaf_bounds = [0]

        total_leaves = self.n_leaves
        updated_leaves = 0

        # Créer un mapping inverse
        leaf_to_signature = {leaf_idx: signature for signature, leaf_idx in signature_to_leaf_mapping.items()}

        for leaf_idx in range(total_leaves):
            if leaf_idx in leaf_to_signature:
                signature = leaf_to_signature[leaf_idx]
                if signature in improved_candidates:
                    # Utiliser les nouveaux candidats
                    new_candidates = improved_candidates[signature]
                    new_candidates_array = np.array(new_candidates, dtype=np.int32)
                    new_leaf_data_segments.append(new_candidates_array)
                    updated_leaves += 1
                else:
                    # Garder les anciens candidats
                    start = self.leaf_bounds[leaf_idx]
                    end = self.leaf_bounds[leaf_idx + 1]
                    old_candidates = self.leaf_data[start:end]
                    new_leaf_data_segments.append(old_candidates)
            else:
                # Garder les anciens candidats
                start = self.leaf_bounds[leaf_idx]
                end = self.leaf_bounds[leaf_idx + 1]
                old_candidates = self.leaf_data[start:end]
                new_leaf_data_segments.append(old_candidates)

            # Mettre à jour leaf_bounds
            new_bound = new_leaf_bounds[-1] + len(new_leaf_data_segments[-1])
            new_leaf_bounds.append(new_bound)

        # Reconstruire leaf_data
        new_flat_tree.leaf_data = np.concatenate(new_leaf_data_segments) if new_leaf_data_segments else np.array([], dtype=np.int32)
        new_flat_tree.leaf_bounds = np.array(new_leaf_bounds, dtype=np.int32)

        print(f"  ✓ {updated_leaves}/{total_leaves} feuilles mises à jour")
        print(f"  ✓ Nouvelle taille leaf_data : {len(new_flat_tree.leaf_data)}")

        return new_flat_tree

    def _precompute_leaf_signatures(self) -> Dict[str, int]:
        """Précalcule un mapping des signatures de feuilles vers leurs indices."""
        print("🗺️  Précalcul du mapping signatures -> feuilles...")

        signature_to_leaf = {}

        for leaf_idx in range(self.n_leaves):
            # Extraire les candidats de cette feuille
            start = self.leaf_bounds[leaf_idx]
            end = self.leaf_bounds[leaf_idx + 1]
            candidates = self.leaf_data[start:end]

            # Créer la signature
            signature = ",".join(map(str, sorted(candidates)))
            signature_to_leaf[signature] = leaf_idx

        print(f"  ✓ {len(signature_to_leaf)} signatures de feuilles précalculées")
        return signature_to_leaf

    # ------------------------------------------------------------------
    # Pruning (suppression des feuilles non utilisées)
    # ------------------------------------------------------------------
    def prune_unused_leaves(self, updated_leaves_set: Set[int]) -> 'TreeFlat':
        """
        Élimine les feuilles qui n'ont pas été mises à jour lors de l'amélioration HNSW.

        Args:
            updated_leaves_set: Ensemble des indices des feuilles mises à jour

        Returns:
            Nouvelle instance de TreeFlat avec uniquement les feuilles mises à jour
        """
        print("🔍 Suppression des feuilles non mises à jour...")

        # 1. Créer une copie de l'arbre plat
        pruned_tree = TreeFlat(self.dims, self.max_depth)

        # 2. Copier toutes les propriétés
        pruned_tree.depth = self.depth
        pruned_tree.n_nodes = self.n_nodes
        pruned_tree.n_levels = self.n_levels

        # 3. Copier les structures de nœuds (inchangées)
        if self.node_centroids is not None:
            pruned_tree.node_centroids = self.node_centroids.copy()
        if self.node_centroids_reduced is not None:
            pruned_tree.node_centroids_reduced = self.node_centroids_reduced.copy()
        pruned_tree.node_children_top_dims = self.node_children_top_dims.copy()
        pruned_tree.node_children_d_head = self.node_children_d_head.copy()
        pruned_tree.node_levels = self.node_levels.copy()
        pruned_tree.node_is_leaf = self.node_is_leaf.copy()
        pruned_tree.node_children_count = self.node_children_count.copy()
        pruned_tree.node_children_start = self.node_children_start.copy()
        pruned_tree.children_indices = self.children_indices.copy()
        pruned_tree.leaf_offset = self.leaf_offset.copy()

        print(f"  → {len(updated_leaves_set)}/{self.n_leaves} feuilles ont été mises à jour")

        # 4. Reconstruire leaf_data avec uniquement les feuilles mises à jour
        new_leaf_data_segments = []
        new_leaf_bounds = [0]
        new_leaf_mapping = {}  # mapping ancien_idx -> nouvel_idx

        new_leaf_idx = 0
        for leaf_idx in range(self.n_leaves):
            if leaf_idx in updated_leaves_set:
                # Récupérer les données de cette feuille
                start = self.leaf_bounds[leaf_idx]
                end = self.leaf_bounds[leaf_idx + 1]
                leaf_data = self.leaf_data[start:end]

                # Ajouter au nouveau leaf_data
                new_leaf_data_segments.append(leaf_data)
                new_bound = new_leaf_bounds[-1] + len(leaf_data)
                new_leaf_bounds.append(new_bound)

                # Mettre à jour le mapping
                new_leaf_mapping[leaf_idx] = new_leaf_idx
                new_leaf_idx += 1

        # 5. Mettre à jour leaf_data et leaf_bounds
        pruned_tree.leaf_data = np.concatenate(new_leaf_data_segments) if new_leaf_data_segments else np.array([], dtype=np.int32)
        pruned_tree.leaf_bounds = np.array(new_leaf_bounds, dtype=np.int32)

        # 6. Mettre à jour leaf_offset
        # -1 pour les nœuds internes et les feuilles non mises à jour
        pruned_tree.leaf_offset = np.full(self.n_nodes, -1, dtype=np.int32)

        # Pour chaque nœud, s'il s'agit d'une feuille mise à jour, mettre à jour son offset
        for i in range(self.n_nodes):
            if self.node_is_leaf[i]:
                old_leaf_idx = self.leaf_offset[i]
                if old_leaf_idx in new_leaf_mapping:
                    pruned_tree.leaf_offset[i] = new_leaf_mapping[old_leaf_idx]

        # 7. Mettre à jour les statistiques
        pruned_tree.n_leaves = len(new_leaf_bounds) - 1

        # Calculer les économies réalisées
        original_leaf_data_size = len(self.leaf_data)
        pruned_leaf_data_size = len(pruned_tree.leaf_data)
        reduction_percentage = ((original_leaf_data_size - pruned_leaf_data_size) / original_leaf_data_size) * 100

        print(f"✅ Suppression des feuilles non mises à jour terminée!")
        print(f"  → Taille originale de leaf_data: {original_leaf_data_size}")
        print(f"  → Taille réduite de leaf_data: {pruned_leaf_data_size}")
        print(f"  → Réduction: {reduction_percentage:.1f}% d'espace")

        return pruned_tree