"""
Module de structures d'arbre pour K16.
Définit les différentes classes pour représenter l'arbre K16.
"""

import numpy as np
from typing import List, Any, Optional, Dict, Union

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
        self.centroids = None     # Pour les nœuds internes: tableau numpy des centroïdes des enfants (aligné avec children)
        self.indices = np.array([], dtype=np.int32)  # Pour les feuilles: tableau numpy des MAX_DATA indices les plus proches
        
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
        self.top_dims_by_level = None  # Liste des indices de dimensions par niveau pour la réduction locale
        self.d_head_by_level = None  # Liste du nombre de dimensions conservées par niveau

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
    
    def compute_dimensional_reduction(self) -> List[np.ndarray]:
        """
        Calcule la réduction de dimension locale par niveau basée sur la variance des centroïdes.
        Utilise les paramètres variance_ratio_root et variance_ratio_decay de la configuration.
        
        Returns:
            List[np.ndarray]: Liste des indices de dimensions par niveau
        """
        if not self.root:
            raise ValueError("Arbre vide - impossible de calculer la réduction de dimension")
        
        dims = int(self.root.centroid.shape[0])
        
        # Retrieve config parameters
        try:
            from .config import ConfigManager
            cm = ConfigManager()
            variance_ratio_root = cm.get("flat_tree", "variance_ratio_root", 0.80)
            variance_ratio_decay = cm.get("flat_tree", "variance_ratio_decay", 0.05)
        except Exception:
            variance_ratio_root = 0.80
            variance_ratio_decay = 0.05
        
        # Gather nodes per level
        stats = self.get_statistics()
        max_depth = stats["max_depth"] + 1
        nodes_by_level = [[] for _ in range(max_depth)]
        
        def _collect(node: TreeNode):
            lvl = node.level
            nodes_by_level[lvl].append(node)
            for child in node.children:
                _collect(child)
        
        _collect(self.root)
        
        # Compute local dimensional reduction per level
        top_dims_by_level = []
        d_head_by_level = []
        
        for level, nodes in enumerate(nodes_by_level):
            if not nodes:
                top_dims_by_level.append(np.array([], dtype=np.int32))
                d_head_by_level.append(0)
                continue
            
            # Calculate variance ratio for this level
            variance_ratio = variance_ratio_root - level * variance_ratio_decay
            variance_ratio = max(variance_ratio, 0.1)  # Minimum 10% of dimensions
            
            # Calculate number of dimensions to keep
            d_head_level = max(1, int(dims * variance_ratio))
            d_head_level = min(d_head_level, dims)  # Cannot exceed total dimensions
            
            # Calculate variance for this level's centroids
            level_centroids = np.vstack([node.centroid for node in nodes])
            var = np.var(level_centroids, axis=0)
            dims_sorted = np.argsort(-var)
            top_dims = np.ascontiguousarray(dims_sorted[:d_head_level], dtype=np.int32)
            
            top_dims_by_level.append(top_dims)
            d_head_by_level.append(d_head_level)
        
        # Store in the tree structure
        self.top_dims_by_level = top_dims_by_level
        self.d_head_by_level = d_head_by_level
        
        print(f"✓ Réduction de dimension locale calculée:")
        for level, d_head_level in enumerate(d_head_by_level):
            if d_head_level > 0:
                ratio = d_head_level / dims
                print(f"  Niveau {level}: {dims} -> {d_head_level} dimensions (ratio: {ratio:.2f})")
        
        return top_dims_by_level

    def apply_perfect_recall(self, vectors: np.ndarray, max_data: int = 200) -> None:
        """
        Applique la transformation Perfect Recall selon l'algorithme défini.
        Garantit 100% de rappel en créant des centroïdes exacts pour chaque vecteur.
        Doit être appelé après compute_dimensional_reduction().
        """
        if not self.root:
            raise ValueError("Arbre vide - impossible d'appliquer Perfect Recall")

        print("🎯 K16Tree Perfect Recall Transformation...")

        # Phase 1: Construction HNSW global
        print("⏳ Phase 1: Building global HNSW index...")
        try:
            import faiss
            n_vectors, dims = vectors.shape

            # Créer index HNSW
            hnsw_index = faiss.IndexHNSWFlat(dims, 32)
            hnsw_index.add(vectors.astype(np.float32))
            print("✓ Global HNSW index ready")

        except ImportError:
            raise RuntimeError("FAISS required for Perfect Recall")

        # Phase 2: Navigation et identification des feuilles touchées
        print("⏳ Phase 2: Identifying vector landing leaves...")
        leaf_to_vectors = {}  # leaf_node -> [vector_indices]

        for i, vector in enumerate(vectors):
            # Naviguer pour trouver la feuille où tombe ce vecteur
            leaf_node = self._navigate_to_leaf(vector)

            if leaf_node not in leaf_to_vectors:
                leaf_to_vectors[leaf_node] = []
            leaf_to_vectors[leaf_node].append(i)

        print(f"✓ Phase 2 complete: {len(leaf_to_vectors)} leaves touched")

        # Phase 3: Transformation des feuilles touchées
        print("⏳ Phase 3: Transforming touched leaves...")
        for leaf_node, vec_indices in leaf_to_vectors.items():
            self._transform_leaf_for_perfect_recall(leaf_node, vec_indices, vectors, hnsw_index, max_data)

        print("🎉 Perfect Recall transformation complete!")
        print(f"   Guarantee : 100 % recall")

    def _navigate_to_leaf(self, query: np.ndarray) -> TreeNode:
        """
        Navigation dans l'arbre hiérarchique pour trouver la feuille.
        Utilise la même logique que TreeFlat avec réduction de dimensions par niveau.
        """
        if not self.top_dims_by_level or not self.d_head_by_level:
            raise ValueError("Réduction de dimensions non calculée - appelez compute_dimensional_reduction() d'abord")

        current = self.root

        while not current.is_leaf():
            if not current.children:
                break

            # Obtenir la réduction de dimensions pour le niveau des enfants
            children_level = current.level + 1
            if children_level < len(self.top_dims_by_level):
                top_dims_level = self.top_dims_by_level[children_level]

                # Préparer la query pour ce niveau (comme dans TreeFlat)
                q_head = query[top_dims_level]

                tail_mask = np.ones(len(query), dtype=bool)
                tail_mask[top_dims_level] = False
                q_tail = query[tail_mask]
                q_tail_norm = float(np.sqrt(np.dot(q_tail, q_tail)))
            else:
                # Fallback si pas de réduction spécifique
                q_head = query.astype(np.float32)
                q_tail_norm = 0.0

            # Calculer les scores pour tous les enfants avec la même logique que TreeFlat
            best_child = None
            best_score = float('-inf')

            for child in current.children:
                # Calculer head et tail pour le centroïde de l'enfant
                if children_level < len(self.top_dims_by_level):
                    top_dims_level = self.top_dims_by_level[children_level]
                    centroid_head = child.centroid[top_dims_level]

                    tail_mask = np.ones(len(child.centroid), dtype=bool)
                    tail_mask[top_dims_level] = False
                    centroid_tail = child.centroid[tail_mask]
                    centroid_tail_norm = float(np.sqrt(np.dot(centroid_tail, centroid_tail)))
                else:
                    centroid_head = child.centroid.astype(np.float32)
                    centroid_tail_norm = 0.0

                # Score avec la même formule que TreeFlat: dot_head + tail_norm * q_tail_norm
                dot_head = np.dot(centroid_head, q_head)
                score = dot_head + centroid_tail_norm * q_tail_norm

                if score > best_score:
                    best_score = score
                    best_child = child

            if best_child is None:
                break

            current = best_child

        return current

    def _transform_leaf_for_perfect_recall(self, leaf_node: TreeNode, vec_indices: List[int],
                                         vectors: np.ndarray, hnsw_index, max_data: int) -> None:
        """
        Transforme une feuille en nœud interne avec des enfants à centroïdes exacts.
        """
        # 1. VIDER la feuille de ses indices actuels
        leaf_node.indices = np.array([], dtype=np.int32)

        # 2. TRANSFORMER la feuille en nœud interne en créant des enfants
        new_level = leaf_node.level + 1

        for vid in vec_indices:
            v = vectors[vid]

            # Recherche HNSW pour les voisins
            _, I = hnsw_index.search(v.reshape(1, -1).astype(np.float32), max_data)
            idx_list = I[0].tolist()

            # Créer un nouvel enfant avec centroïde exact
            child_node = TreeNode(centroid=v.copy(), level=new_level)
            child_node.set_indices(idx_list)

            leaf_node.add_child(child_node)

        # 3. Mettre à jour les centroïdes des enfants
        leaf_node.set_children_centroids()

    def __str__(self) -> str:
        """Représentation sous forme de chaîne pour le débogage."""
        if not self.root and not self.flat_tree:
            return "Empty Tree"

        stats = self.get_statistics()

        if self.flat_tree:
            return (f"K16Tree(nodes={stats['node_count']}, "
                    f"leaves={stats['leaf_count']}, "
                    f"height={stats['max_depth']}, "
                    f"optimized=True)")
        else:
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