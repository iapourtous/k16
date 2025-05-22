"""
Module de structure d'arbre plat pour K16.
Implémente une structure mémoire plate optimisée pour la recherche rapide.
Structure recommandée par défaut pour les performances optimales.
"""

import numpy as np
from typing import Optional, Dict, Any, TYPE_CHECKING

from .tree import TreeNode, K16Tree

if TYPE_CHECKING:
    from .io import VectorReader

class TreeFlat:
    """
    Structure optimisée pour la représentation en mémoire plate d'un arbre K16.
    Utilise des tableaux numpy contigus pour chaque niveau et élément de l'arbre.
    Cette structure est optimisée pour:
    - Localité mémoire (meilleure utilisation du cache)
    - Opérations vectorisées sur les centroïdes
    - Accès directs sans indirection
    """
    
    def __init__(self, dims: int, max_depth: int):
        """
        Initialise un arbre plat.
        
        Args:
            dims: Dimension des vecteurs
            max_depth: Profondeur maximale de l'arbre
        """
        self.dims = dims              # Dimension des vecteurs
        self.depth = 0                # Profondeur actuelle (sera mise à jour durant la construction)
        self.max_depth = max_depth    # Profondeur maximale possible
        
        # Structure par niveau (pour les nœuds internes)
        self.centroids = []           # Liste de np.ndarray par niveau, chacun de forme (n_nodes, dims)
        self.child_ptr = []           # Liste de np.ndarray par niveau, chacun de forme (n_nodes, n_children)
        
        # Structure des feuilles
        self.leaf_offset = None       # np.ndarray de forme (nb_total_nodes,) avec -1 pour nœuds internes
        self.leaf_data = None         # np.ndarray contenant tous les indices des vecteurs des feuilles
        self.leaf_bounds = None       # np.ndarray de forme (n_leaves+1,) indiquant les offsets dans leaf_data
        
        # Statistiques
        self.n_nodes = 0              # Nombre total de nœuds
        self.n_leaves = 0             # Nombre total de feuilles
        self.n_levels = 0             # Nombre de niveaux
        
        # Pour le mapping des nœuds
        self.nodes_by_level = []      # Liste de liste de nœuds par niveau
    
    @classmethod
    def from_tree(cls, tree: K16Tree) -> 'TreeFlat':
        """
        Convertit un K16Tree traditionnel en structure TreeFlat optimisée.
        
        Args:
            tree: Arbre K16 à convertir
            
        Returns:
            TreeFlat: Structure plate optimisée
        """
        if not tree.root:
            raise ValueError("L'arbre source est vide")
        
        # Déterminer la dimension des vecteurs et la profondeur maximale
        dims = tree.root.centroid.shape[0]
        stats = tree.get_statistics()
        max_depth = stats["max_depth"] + 1  # +1 car les niveaux commencent à 0
        
        # Créer un nouvel arbre plat
        flat_tree = cls(dims, max_depth)
        
        # Compter les nœuds à chaque niveau et compiler les listes de nœuds par niveau
        flat_tree.nodes_by_level = [[] for _ in range(max_depth)]
        
        def collect_nodes_by_level(node: TreeNode) -> None:
            level = node.level
            flat_tree.nodes_by_level[level].append(node)
            for child in node.children:
                collect_nodes_by_level(child)
        
        collect_nodes_by_level(tree.root)
        
        # Total de nœuds et nombre de nœuds par niveau
        total_nodes = sum(len(nodes) for nodes in flat_tree.nodes_by_level)
        level_counts = [len(nodes) for nodes in flat_tree.nodes_by_level]
        
        # Allouer l'espace pour les tableaux de feuilles
        flat_tree.leaf_offset = np.full(total_nodes, -1, dtype=np.int32)
        
        # Identifier les feuilles et calculer le nombre total d'indices
        leaf_nodes = []
        leaf_indices_count = 0
        
        for level, nodes in enumerate(flat_tree.nodes_by_level):
            for node in nodes:
                if node.is_leaf():
                    leaf_nodes.append(node)
                    leaf_indices_count += len(node.indices)
        
        # Allouer l'espace pour les données des feuilles
        flat_tree.leaf_data = np.zeros(leaf_indices_count, dtype=np.int32)
        flat_tree.leaf_bounds = np.zeros(len(leaf_nodes) + 1, dtype=np.int32)
        
        # Initialiser les tableaux pour chaque niveau
        for level, count in enumerate(level_counts):
            if count > 0:
                # Allouer les centroïdes pour ce niveau
                flat_tree.centroids.append(np.zeros((count, dims), dtype=np.float32))
                
                # Calculer le nombre max d'enfants pour ce niveau
                if level < max_depth - 1:  # Pas pour le dernier niveau
                    max_children = max(len(node.children) for node in flat_tree.nodes_by_level[level])
                    if max_children > 0:
                        flat_tree.child_ptr.append(np.full((count, max_children), -1, dtype=np.int32))
                    else:
                        flat_tree.child_ptr.append(np.array([], dtype=np.int32))
                else:
                    flat_tree.child_ptr.append(np.array([], dtype=np.int32))
        
        # Créer un mapping des nœuds vers leurs indices dans l'arbre plat
        node_to_flat_idx = {}
        for level, nodes in enumerate(flat_tree.nodes_by_level):
            for i, node in enumerate(nodes):
                node_to_flat_idx[id(node)] = i
        
        # Remplir les tableaux de centroïdes et connexions entre nœuds
        for level, nodes in enumerate(flat_tree.nodes_by_level):
            for i, node in enumerate(nodes):
                # Remplir le centroïde
                flat_tree.centroids[level][i] = node.centroid
                
                # Si ce n'est pas une feuille, remplir les pointeurs vers les enfants
                if not node.is_leaf() and level < len(flat_tree.child_ptr):
                    for j, child in enumerate(node.children):
                        if j < flat_tree.child_ptr[level].shape[1]:
                            child_idx = node_to_flat_idx[id(child)]
                            flat_tree.child_ptr[level][i, j] = child_idx
        
        # Remplir les données des feuilles
        current_leaf_idx = 0
        current_data_idx = 0
        flat_tree.leaf_bounds[0] = 0
        
        for level, nodes in enumerate(flat_tree.nodes_by_level):
            for i, node in enumerate(nodes):
                if node.is_leaf():
                    # Marquer le nœud comme feuille
                    global_node_idx = sum(level_counts[:level]) + i
                    flat_tree.leaf_offset[global_node_idx] = current_leaf_idx
                    
                    # Copier les indices
                    indices = node.indices
                    n_indices = len(indices)
                    flat_tree.leaf_data[current_data_idx:current_data_idx + n_indices] = indices
                    
                    # Mettre à jour les bornes
                    current_data_idx += n_indices
                    flat_tree.leaf_bounds[current_leaf_idx + 1] = current_data_idx
                    
                    current_leaf_idx += 1
        
        # Finaliser l'arbre plat
        flat_tree.n_nodes = total_nodes
        flat_tree.n_leaves = len(leaf_nodes)
        flat_tree.n_levels = max_depth
        flat_tree.depth = max_depth - 1
        
        # Aucune représentation binaire : la recherche s'effectue uniquement
        # avec les centroïdes float32.
        return flat_tree
    
    
    def search_tree_single(self, query: np.ndarray) -> np.ndarray:
        """
        Recherche dans l'arbre plat avec l'algorithme single path optimisé.
        
        Args:
            query: Vecteur requête normalisé
            
        Returns:
            np.ndarray: Tableau des indices des vecteurs les plus proches
        """
        # Commencer à la racine
        node_idx = 0
        level = 0
        
        # Descendre l'arbre
        while level < self.depth:
            level_centroids = self.centroids[level]
            
            # Vérifier si on a un seul nœud à ce niveau
            if level_centroids.shape[0] <= node_idx:
                break
                
            # Approche classique par similarité cosinus (produit scalaire)
            if self.child_ptr[level].shape[0] <= node_idx:
                break

            children_ptr = self.child_ptr[level][node_idx]
            valid_children = children_ptr[children_ptr >= 0]

            if len(valid_children) == 0:
                break

            # Calculer les similarités avec les centroïdes des enfants valides
            child_centroids = np.array([self.centroids[level + 1][idx] for idx in valid_children])
            similarities = np.dot(child_centroids, query)

            # Trouver le meilleur enfant
            best_idx = np.argmax(similarities)
            node_idx = valid_children[best_idx]
            
            # Passer au niveau suivant
            level += 1
        
        # Arrivé à une feuille ou fin de parcours, vérifier si c'est une feuille
        global_node_idx = sum(len(nodes) for nodes in self.nodes_by_level[:level]) + node_idx
        
        if global_node_idx < len(self.leaf_offset) and self.leaf_offset[global_node_idx] >= 0:
            leaf_idx = self.leaf_offset[global_node_idx]
            start_idx = self.leaf_bounds[leaf_idx]
            end_idx = self.leaf_bounds[leaf_idx + 1]
            return self.leaf_data[start_idx:end_idx]
        
        # Pas de feuille trouvée
        return np.array([], dtype=np.int32)
    
    def search_tree_beam(self, query: np.ndarray, beam_width: int = 3) -> np.ndarray:
        """
        Recherche dans l'arbre plat avec l'algorithme beam search optimisé.

        Args:
            query: Vecteur requête normalisé
            beam_width: Largeur du faisceau (nombre de branches explorées)

        Returns:
            np.ndarray: Tableau des indices des vecteurs les plus proches
        """
        # Cas beam_width = 1 : utiliser la recherche single qui est optimisée
        if beam_width <= 1:
            return self.search_tree_single(query)

        # Commencer avec la racine (niveau 0, index 0)
        beam = [(0, 0, 1.0)]  # (level, node_idx, score)
        leaves = []  # Feuilles trouvées [(leaf_idx, score), ...]

        # Descendre l'arbre en explorant plusieurs chemins
        while beam:
            # Extraire le nœud le plus prometteur
            level, node_idx, score = beam.pop(0)

            # Si on a atteint le niveau maximum
            if level >= self.depth:
                continue

            # Vérifier si ce nœud est une feuille
            if node_idx < len(self.nodes_by_level[level]):
                global_idx = sum(len(nodes) for nodes in self.nodes_by_level[:level]) + node_idx

                if global_idx < len(self.leaf_offset) and self.leaf_offset[global_idx] >= 0:
                    # C'est une feuille, l'ajouter aux résultats
                    leaf_idx = self.leaf_offset[global_idx]
                    leaves.append((leaf_idx, score))
                    continue

            # Ce n'est pas une feuille, explorer les enfants
            if level < len(self.child_ptr) and node_idx < len(self.child_ptr[level]):
                children_ptr = self.child_ptr[level][node_idx]
                valid_children = children_ptr[children_ptr >= 0]

                if len(valid_children) > 0:
                    # Calculer les similarités avec les centroïdes des enfants valides
                    child_centroids = np.array([self.centroids[level + 1][idx] for idx in valid_children])
                    similarities = np.dot(child_centroids, query)

                    # Trier par similarité décroissante
                    sorted_indices = np.argsort(-similarities)

                    # Ajouter les meilleurs enfants au faisceau
                    for i in sorted_indices[:beam_width]:
                        child_idx = valid_children[i]
                        child_score = similarities[i]
                        beam.append((level + 1, child_idx, child_score))

            # Trier le faisceau par score décroissant et garder les beam_width meilleurs
            beam.sort(key=lambda x: x[2], reverse=True)
            beam = beam[:beam_width]

        # Collecter les indices de toutes les feuilles trouvées
        all_indices = []

        # Trier les feuilles par score décroissant (meilleure qualité d'abord)
        leaves.sort(key=lambda x: x[1], reverse=True)

        # Collecter les indices des beam_width meilleures feuilles
        for leaf_idx, _ in leaves[:beam_width]:
            start_idx = self.leaf_bounds[leaf_idx]
            end_idx = self.leaf_bounds[leaf_idx + 1]
            all_indices.extend(self.leaf_data[start_idx:end_idx])

        # S'il n'y a pas assez de données, continuer à prendre des feuilles
        if len(all_indices) < 200 and len(leaves) > beam_width:
            for leaf_idx, _ in leaves[beam_width:]:
                start_idx = self.leaf_bounds[leaf_idx]
                end_idx = self.leaf_bounds[leaf_idx + 1]
                all_indices.extend(self.leaf_data[start_idx:end_idx])
                if len(all_indices) >= 200:
                    break

        # Éliminer les doublons et retourner
        return np.array(list(set(all_indices)), dtype=np.int32)
    
    def search_tree(
        self,
        query: np.ndarray,
        beam_width: int = 1,
        vectors_reader: Optional['VectorReader'] = None,
        k: Optional[int] = None,
    ) -> np.ndarray:
        """
        Recherche dans l'arbre plat et, si un ``VectorReader`` est fourni, trie
        automatiquement les candidats par similarité (produit scalaire) et
        retourne éventuellement les ``k`` meilleurs.

        Cette signature reste rétro-compatible : les appels existants qui ne
        fournissent que ``query`` et ``beam_width`` conservent l'ancien
        comportement (ensemble non trié).

        Args:
            query: Vecteur requête normalisé (float32, norme ≈ 1).
            beam_width: Largeur du faisceau (``1`` = single path).
            vectors_reader: Instance de ``VectorReader`` permettant de récupérer
                les vecteurs bruts pour calculer les similarités.  Si ``None``
                (par défaut), aucun tri n'est effectué — on renvoie simplement
                l'ensemble de candidats comme auparavant.
            k: Nombre maximum de résultats à retourner après tri.  Ignoré si
                ``vectors_reader`` est ``None``.

        Returns
        -------
        np.ndarray
            Tableau d'indices, éventuellement trié et tronqué à ``k`` éléments.
        """

        # 1) Récupération de l'ensemble de candidats non trié (comportement
        #    historique).
        if beam_width <= 1:
            candidates = self.search_tree_single(query)
        else:
            candidates = self.search_tree_beam(query, beam_width)

        # 2) Tri optionnel si l'on dispose des vecteurs.
        if vectors_reader is not None and len(candidates) > 0:
            # Calcul des similarités cosinus via produit scalaire (les vecteurs
            # d'entrée sont supposés normalisés).
            similarities = vectors_reader.dot(candidates, query)

            # Si k est fourni et plus petit que le nombre total de candidats,
            # utiliser argpartition (O(n)) plutôt que argsort (O(n log n)) pour
            # extraire rapidement les k meilleurs, puis trier uniquement ces k.
            if k is not None and 0 < k < len(candidates):
                # Obtenir les k indices locaux (pas encore triés) des plus grandes valeurs
                top_k_local = np.argpartition(-similarities, k - 1)[:k]

                # Trier ces k indices selon la similarité décroissante
                top_k_sorted = top_k_local[np.argsort(-similarities[top_k_local])]
                candidates = np.array([candidates[i] for i in top_k_sorted], dtype=np.int32)
            else:
                # k absent ou supérieur : trier entièrement (petit coût si n modéré)
                sorted_local_idx = np.argsort(-similarities)
                candidates = np.array([candidates[i] for i in sorted_local_idx], dtype=np.int32)

        return candidates
    
    def save(self, file_path: str) -> None:
        """
        Sauvegarde l'arbre plat dans un fichier.
        
        Args:
            file_path: Chemin du fichier de sortie
        """
        # Préparer les données à sauvegarder
        data = {
            'dims': self.dims,
            'depth': self.depth,
            'max_depth': self.max_depth,
            'n_nodes': self.n_nodes,
            'n_leaves': self.n_leaves,
            'n_levels': self.n_levels,
            'centroids': self.centroids,
            'child_ptr': self.child_ptr,
            'leaf_offset': self.leaf_offset,
            'leaf_data': self.leaf_data,
            'leaf_bounds': self.leaf_bounds
        }
        
        # Sauvegarder avec numpy
        np.save(file_path, data, allow_pickle=True)
    
    @classmethod
    def load(cls, file_path: str) -> 'TreeFlat':
        """
        Charge un arbre plat depuis un fichier.
        
        Args:
            file_path: Chemin du fichier d'entrée
            
        Returns:
            TreeFlat: Structure plate optimisée
        """
        # Charger les données
        data = np.load(file_path, allow_pickle=True).item()
        
        # Créer un nouvel arbre plat
        tree = cls(data['dims'], data['max_depth'])
        
        # Remplir les attributs
        tree.depth = data['depth']
        tree.n_nodes = data['n_nodes']
        tree.n_leaves = data['n_leaves']
        tree.n_levels = data['n_levels']
        tree.centroids = data['centroids']
        tree.child_ptr = data['child_ptr']
        tree.leaf_offset = data['leaf_offset']
        tree.leaf_data = data['leaf_data']
        tree.leaf_bounds = data['leaf_bounds']
        
        return tree
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Retourne des statistiques sur l'arbre plat.
        
        Returns:
            Dict[str, Any]: Dictionnaire des statistiques
        """
        stats = {
            "n_nodes": self.n_nodes,
            "n_leaves": self.n_leaves,
            "max_depth": self.depth,
            "n_levels": self.n_levels,
            "nodes_per_level": [len(nodes) for nodes in self.nodes_by_level]
        }
        
        # Si nous avons des feuilles, calculer la taille moyenne et totale
        if self.n_leaves > 0:
            leaf_sizes = [self.leaf_bounds[i+1] - self.leaf_bounds[i] for i in range(self.n_leaves)]
            stats["avg_leaf_size"] = sum(leaf_sizes) / self.n_leaves
            stats["min_leaf_size"] = min(leaf_sizes)
            stats["max_leaf_size"] = max(leaf_sizes)
            stats["total_indices"] = len(self.leaf_data)
        
        return stats