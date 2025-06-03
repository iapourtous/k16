"""
Module de structures d'arbre pour K16.
Définit les différentes classes pour représenter l'arbre K16.
"""

import numpy as np
import time
from collections import defaultdict
from typing import List, Any, Optional, Dict, Union, TYPE_CHECKING

# Imports relatifs pour le nouveau package
if TYPE_CHECKING:
    from k16.io.reader import VectorReader

class TreeNode:
    """
    Noeud de base pour l'arbre K16.
    Représente un nœud dans l'arbre hiérarchique construit pour la recherche de vecteurs similaires.
    """
    
    def __init__(self, centroid: Optional[np.ndarray] = None, level: int = 0):
        """
        Initialise un nœud d'arbre K16.
        
        Args:
            centroid: Vecteur centroïde représentant ce nœud (optionnel)
            level: Niveau du nœud dans l'arbre (0 = racine)
        """
        self.centroid = centroid  # Centroïde du nœud
        self.level = level        # Niveau dans l'arbre
        self.children = []        # Pour les nœuds internes: liste des noeuds enfants
        self.centroids = None     # Pour les nœuds internes: tableau numpy des centroïdes des enfants (aligné avec children)
        self.indices = np.array([], dtype=np.int32)  # Pour les feuilles: tableau numpy des MAX_DATA indices les plus proches
        
        # NOUVEAU : Réduction dimensionnelle pour les ENFANTS
        self.children_top_dims = None  # np.ndarray[max_dims] - indices des dimensions importantes pour les enfants
        self.children_d_head = None    # int - nombre de dimensions conservées pour les enfants
        
    def is_leaf(self) -> bool:
        """
        Vérifie si ce nœud est une feuille.
        
        Returns:
            bool: True si le nœud est une feuille (pas d'enfants), False sinon
        """
        return len(self.children) == 0
    
    def add_child(self, child: 'TreeNode') -> None:
        """
        Ajoute un nœud enfant à ce nœud.
        
        Args:
            child: Nœud enfant à ajouter
        """
        self.children.append(child)
        
        # Mettre à jour le tableau des centroïdes
        if self.centroids is None:
            self.centroids = np.array([child.centroid])
        else:
            self.centroids = np.vstack([self.centroids, child.centroid])
    
    def set_children_centroids(self) -> None:
        """
        Construit le tableau des centroïdes à partir des centroïdes des enfants.
        À appeler après avoir ajouté tous les enfants pour garantir l'alignement.
        """
        if not self.children:
            return
            
        centroids = [child.centroid for child in self.children]
        self.centroids = np.array(centroids)
    
    def set_indices(self, indices: Union[List[int], np.ndarray]) -> None:
        """
        Définit les indices associés à ce nœud (pour les feuilles).
        
        Args:
            indices: Liste ou tableau des indices des vecteurs les plus proches du centroïde
        """
        if isinstance(indices, list):
            self.indices = np.array(indices, dtype=np.int32)
        else:
            self.indices = indices.astype(np.int32)
    
    def get_size(self) -> int:
        """
        Calcule la taille du sous-arbre enraciné à ce nœud.
        
        Returns:
            int: Nombre total de nœuds dans le sous-arbre
        """
        size = 1  # Ce nœud
        for child in self.children:
            size += child.get_size()
        return size
    
    def __str__(self) -> str:
        """Représentation sous forme de chaîne pour le débogage."""
        if self.is_leaf():
            return f"Leaf(level={self.level}, indices={len(self.indices)})"
        else:
            return f"Node(level={self.level}, children={len(self.children)})"

class K16Tree:
    """
    Classe principale pour l'arbre K16.
    Gère un arbre hiérarchique optimisé pour la recherche rapide de vecteurs similaires.
    Peut utiliser une structure plate optimisée pour des performances maximales.
    """

    def __init__(self, root: Optional[TreeNode] = None):
        """
        Initialise un arbre K16.

        Args:
            root: Nœud racine de l'arbre (optionnel)
        """
        self.root = root
        self.stats = {}  # Statistiques sur l'arbre
        self.flat_tree = None  # Version optimisée en structure plate

    def set_root(self, root: TreeNode) -> None:
        """
        Définit le nœud racine de l'arbre.

        Args:
            root: Nœud racine à définir
        """
        self.root = root
        # Réinitialiser l'arbre plat car il n'est plus valide
        self.flat_tree = None
    
    def get_height(self) -> int:
        """
        Calcule la hauteur de l'arbre (profondeur maximale des feuilles).
        
        Returns:
            int: Hauteur de l'arbre
        """
        if not self.root:
            return 0
        
        def get_node_height(node: TreeNode) -> int:
            if node.is_leaf():
                return node.level
            return max(get_node_height(child) for child in node.children)
        
        return get_node_height(self.root)
    
    def get_leaf_count(self) -> int:
        """
        Compte le nombre de feuilles dans l'arbre.
        
        Returns:
            int: Nombre de feuilles
        """
        if not self.root:
            return 0
        
        def count_leaves(node: TreeNode) -> int:
            if node.is_leaf():
                return 1
            return sum(count_leaves(child) for child in node.children)
        
        return count_leaves(self.root)
    
    def get_node_count(self) -> int:
        """
        Compte le nombre total de nœuds dans l'arbre.
        
        Returns:
            int: Nombre total de nœuds
        """
        if not self.root:
            return 0
        return self.root.get_size()
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Calcule diverses statistiques sur l'arbre.
        
        Returns:
            Dict: Dictionnaire de statistiques
        """
        if not self.root:
            return {"error": "Arbre vide"}
        
        stats = {
            "node_count": 0,
            "leaf_count": 0,
            "max_depth": 0,
            "min_leaf_depth": float('inf'),
            "avg_leaf_depth": 0,
            "leaf_depths": [],
            "branching_factors": {},
            "avg_branching_factor": 0,
            "leaf_sizes": [],
            "avg_leaf_size": 0,
            "min_leaf_size": float('inf'),
            "max_leaf_size": 0,
            "total_indices": 0,
            "nodes_with_reduction": 0,
            "avg_dimensions_kept": 0
        }
        
        def traverse(node: TreeNode) -> None:
            nonlocal stats
            
            stats["node_count"] += 1
            
            # Mise à jour de la profondeur maximale
            stats["max_depth"] = max(stats["max_depth"], node.level)
            
            # Statistiques sur la réduction dimensionnelle
            if node.children_top_dims is not None and node.children_d_head is not None:
                stats["nodes_with_reduction"] += 1
            
            if node.is_leaf():
                # C'est une feuille
                stats["leaf_count"] += 1
                stats["min_leaf_depth"] = min(stats["min_leaf_depth"], node.level)
                stats["leaf_depths"].append(node.level)
                
                # Statistiques des indices
                leaf_size = len(node.indices)
                stats["leaf_sizes"].append(leaf_size)
                stats["min_leaf_size"] = min(stats["min_leaf_size"], leaf_size)
                stats["max_leaf_size"] = max(stats["max_leaf_size"], leaf_size)
                stats["total_indices"] += leaf_size
            else:
                # C'est un nœud interne
                branch_count = len(node.children)
                if branch_count in stats["branching_factors"]:
                    stats["branching_factors"][branch_count] += 1
                else:
                    stats["branching_factors"][branch_count] = 1
                
                # Analyser les enfants
                for child in node.children:
                    traverse(child)
        
        # Parcourir l'arbre
        traverse(self.root)
        
        # Calculer les moyennes
        if stats["leaf_count"] > 0:
            stats["avg_leaf_depth"] = sum(stats["leaf_depths"]) / stats["leaf_count"]
            stats["avg_leaf_size"] = sum(stats["leaf_sizes"]) / stats["leaf_count"]
        
        internal_nodes = stats["node_count"] - stats["leaf_count"]
        if internal_nodes > 0:
            total_branches = sum(k * v for k, v in stats["branching_factors"].items())
            stats["avg_branching_factor"] = total_branches / internal_nodes
        
        if stats["nodes_with_reduction"] > 0:
            # Calculer la moyenne des dimensions conservées
            total_dims = 0
            nodes_counted = 0
            
            def count_dims(node: TreeNode) -> None:
                nonlocal total_dims, nodes_counted
                if node.children_d_head is not None:
                    total_dims += node.children_d_head
                    nodes_counted += 1
                for child in node.children:
                    count_dims(child)
            
            count_dims(self.root)
            if nodes_counted > 0:
                stats["avg_dimensions_kept"] = total_dims / nodes_counted
        
        # Conserver les statistiques dans l'instance
        self.stats = stats
        
        return stats
    
    def compute_dimensional_reduction(self, max_dims: int = None, method: str = "variance") -> None:
        """
        Calcule la réduction de dimension locale pour chaque nœud interne.
        Chaque nœud identifie les dimensions qui séparent le mieux ses enfants.

        Args:
            max_dims: Nombre de dimensions à conserver. Si None, utilise le paramètre
                      de configuration max_dims ou une valeur par défaut.
            method: Méthode de réduction de dimension: "variance", "directional"
        """
        if not self.root:
            raise ValueError("Arbre vide - impossible de calculer la réduction de dimension")

        dims = int(self.root.centroid.shape[0])

        # Retrieve config parameters if max_dims not provided
        if max_dims is None:
            try:
                from k16.utils.config import ConfigManager
                cm = ConfigManager()
                max_dims = cm.get("flat_tree", "max_dims", 128)  # Valeur par défaut: 128 dimensions
            except Exception:
                max_dims = 128  # Valeur par défaut si la config n'est pas disponible

        # S'assurer que max_dims est valide
        max_dims = min(max(1, max_dims), dims)  # Entre 1 et dims

        print(f"⏳ Calcul de la réduction de dimension par nœud (max_dims={max_dims}, method={method})...")

        nodes_processed = 0
        start_time = time.time()

        def compute_node_reduction(node: TreeNode) -> None:
            nonlocal nodes_processed

            # Ne pas calculer de réduction pour les feuilles
            if node.is_leaf():
                return

            # S'assurer que le nœud a des enfants avec centroïdes
            if not node.children or len(node.children) < 2:
                # Pas assez d'enfants pour faire une réduction significative
                # Utiliser toutes les dimensions
                node.children_top_dims = np.arange(min(max_dims, dims), dtype=np.int32)
                node.children_d_head = min(max_dims, dims)
                nodes_processed += 1
                for child in node.children:
                    compute_node_reduction(child)
                return

            # Extraire les centroïdes des enfants
            child_centroids = np.vstack([child.centroid for child in node.children])

            if method == "directional":
                # Analyse directionnelle : focus sur les dimensions qui séparent le mieux les clusters
                # Calculer la séparation de chaque dimension
                dim_separation = np.zeros(dims)

                # Pour chaque paire de centroïdes, calculer leur séparation
                n_centroids = child_centroids.shape[0]
                for i in range(n_centroids):
                    for j in range(i+1, n_centroids):  # Uniquement les paires uniques
                        # Calcul de la séparation par dimension
                        dimension_diff = np.abs(child_centroids[i] - child_centroids[j])
                        dim_separation += dimension_diff

                # Trier les dimensions par séparation décroissante
                dims_sorted = np.argsort(-dim_separation)
                node.children_top_dims = np.ascontiguousarray(dims_sorted[:max_dims], dtype=np.int32)
                node.children_d_head = max_dims

            else:
                # Méthode par variance (par défaut)
                var = np.var(child_centroids, axis=0)
                dims_sorted = np.argsort(-var)
                node.children_top_dims = np.ascontiguousarray(dims_sorted[:max_dims], dtype=np.int32)
                node.children_d_head = max_dims

            nodes_processed += 1

            # Afficher la progression pour les gros arbres
            if nodes_processed % 1000 == 0:
                elapsed = time.time() - start_time
                print(f"  → {nodes_processed} nœuds traités ({elapsed:.1f}s)")

            # Appliquer récursivement aux enfants
            for child in node.children:
                compute_node_reduction(child)

        # Commencer le calcul depuis la racine
        compute_node_reduction(self.root)

        elapsed = time.time() - start_time
        method_name = "Analyse directionnelle" if method == "directional" else ("variance")
        print(f"✓ Réduction de dimension par nœud calculée ({method_name})")
        print(f"  → {nodes_processed} nœuds traités en {elapsed:.2f}s")
        print(f"  → {dims} → {max_dims} dimensions par nœud")
    
    def __str__(self) -> str:
        """Représentation sous forme de chaîne pour le débogage."""
        if not self.root and not self.flat_tree:
            return "Empty Tree"

        stats = self.get_statistics()

        if self.flat_tree:
            return (f"K16Tree(nodes={stats['node_count']}, "
                    f"leaves={stats['leaf_count']}, "
                    f"height={stats['max_depth']}, "
                    f"nodes_with_reduction={stats['nodes_with_reduction']}, "
                    f"optimized=True)")
        else:
            return (f"K16Tree(nodes={stats['node_count']}, "
                    f"leaves={stats['leaf_count']}, "
                    f"height={stats['max_depth']}, "
                    f"nodes_with_reduction={stats['nodes_with_reduction']})")
    
    def save_statistics(self, file_path: str) -> None:
        """
        Sauvegarde les statistiques de l'arbre dans un fichier texte.
        
        Args:
            file_path: Chemin du fichier de sortie
        """
        stats = self.get_statistics()
        
        with open(file_path, "w") as f:
            f.write("STATISTIQUES DE L'ARBRE K16\n")
            f.write("===========================\n\n")
            
            f.write("Structure générale\n")
            f.write("-----------------\n")
            f.write(f"Nombre total de nœuds : {stats['node_count']}\n")
            f.write(f"Nombre de feuilles    : {stats['leaf_count']}\n")
            f.write(f"Profondeur maximale   : {stats['max_depth']}\n")
            f.write(f"Profondeur min feuille: {stats['min_leaf_depth']}\n")
            f.write(f"Profondeur moy feuille: {stats['avg_leaf_depth']:.2f}\n\n")
            
            f.write("Facteurs de branchement\n")
            f.write("----------------------\n")
            f.write(f"Facteur de branchement moyen: {stats['avg_branching_factor']:.2f}\n")
            for branches, count in sorted(stats["branching_factors"].items()):
                f.write(f"  {branches} branches: {count} nœuds\n")
            f.write("\n")
            
            f.write("Statistiques des feuilles\n")
            f.write("------------------------\n")
            f.write(f"Taille moyenne: {stats['avg_leaf_size']:.2f} indices\n")
            f.write(f"Taille min    : {stats['min_leaf_size']} indices\n")
            f.write(f"Taille max    : {stats['max_leaf_size']} indices\n")
            f.write(f"Total indices : {stats['total_indices']} indices\n\n")
            
            f.write("Réduction dimensionnelle\n")
            f.write("-----------------------\n")
            f.write(f"Nœuds avec réduction : {stats['nodes_with_reduction']}\n")
            f.write(f"Dimensions moyennes  : {stats['avg_dimensions_kept']:.1f}\n")

    def improve_with_hnsw(self, vectors_reader, max_data: Optional[int] = None) -> 'K16Tree':
        """
        Améliore l'arbre avec HNSW pour optimiser les candidats de chaque feuille.
        Cette fonction est maintenant une façade qui utilise la version dans TreeFlat.
        Supprime automatiquement les feuilles non mises à jour (pruning) pour économiser de l'espace.

        Args:
            vectors_reader: Lecteur de vecteurs
            max_data: Nombre maximum de candidats par feuille (utilise la config si None)

        Returns:
            K16Tree: Nouvelle instance d'arbre amélioré
        """
        if not self.flat_tree:
            raise ValueError("L'arbre doit être converti en structure plate avant l'amélioration HNSW")

        # Utiliser la version dans flat_tree
        print("🔄 Délégation de l'amélioration HNSW à la structure plate...")
        improved_flat_tree = self.flat_tree.improve_with_hnsw(vectors_reader, max_data)

        # Créer une nouvelle instance d'arbre avec l'arbre plat amélioré
        improved_tree = K16Tree(self.root)
        improved_tree.flat_tree = improved_flat_tree

        return improved_tree