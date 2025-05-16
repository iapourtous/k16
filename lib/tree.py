"""
Module de structures d'arbre pour K16.
Définit les différentes classes pour représenter l'arbre K16.
"""

import numpy as np
from typing import List, Any, Optional, Dict, Tuple, Union

class TreeNode:
    """
    Noeud de base pour l'arbre K16.
    Représente un nœud dans l'arbre hiérarchique construit pour la recherche de vecteurs similaires.
    """
    
    def __init__(self, centroid: np.ndarray, level: int = 0):
        """
        Initialise un nœud d'arbre K16.
        
        Args:
            centroid: Vecteur centroïde représentant ce nœud
            level: Niveau du nœud dans l'arbre (0 = racine)
        """
        self.centroid = centroid  # Centroïde du nœud
        self.level = level        # Niveau dans l'arbre
        self.children = []        # Pour les nœuds internes: liste des noeuds enfants
        self.indices = []         # Pour les feuilles: liste pré-calculée des MAX_DATA indices les plus proches
        
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
    
    def set_indices(self, indices: List[int]) -> None:
        """
        Définit les indices associés à ce nœud (pour les feuilles).
        
        Args:
            indices: Liste des indices des vecteurs les plus proches du centroïde
        """
        self.indices = indices
    
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
    """
    
    def __init__(self, root: Optional[TreeNode] = None):
        """
        Initialise un arbre K16.
        
        Args:
            root: Nœud racine de l'arbre (optionnel)
        """
        self.root = root
        self.stats = {}  # Statistiques sur l'arbre
    
    def set_root(self, root: TreeNode) -> None:
        """
        Définit le nœud racine de l'arbre.
        
        Args:
            root: Nœud racine à définir
        """
        self.root = root
    
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
            "total_indices": 0
        }
        
        def traverse(node: TreeNode) -> None:
            nonlocal stats
            
            stats["node_count"] += 1
            
            # Mise à jour de la profondeur maximale
            stats["max_depth"] = max(stats["max_depth"], node.level)
            
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
        
        # Conserver les statistiques dans l'instance
        self.stats = stats
        
        return stats
    
    def __str__(self) -> str:
        """Représentation sous forme de chaîne pour le débogage."""
        if not self.root:
            return "Empty Tree"
        
        stats = self.get_statistics()
        return (f"K16Tree(nodes={stats['node_count']}, "
                f"leaves={stats['leaf_count']}, "
                f"height={stats['max_depth']})")
    
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
            f.write(f"Total indices : {stats['total_indices']} indices\n")