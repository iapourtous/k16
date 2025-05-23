"""
Module de recherche pour K16.
Implémente les algorithmes de recherche de vecteurs similaires dans l'arbre K16.
"""

import numpy as np
import time
from typing import List, Dict, Any, Optional, Union, Tuple
import faiss

from .tree import TreeNode, K16Tree
from .io import VectorReader

class Searcher:
    """
    Classe principale pour la recherche dans l'arbre K16.
    Implémente différentes stratégies de recherche pour trouver les vecteurs similaires.
    Supporte la structure d'arbre traditionnelle et la structure plate optimisée.
    """

    def __init__(self, tree: Union[TreeNode, K16Tree], vectors_reader: VectorReader, use_faiss: bool = True,
                 search_type: str = "single", beam_width: int = 3, max_data: int = 4000):
        """
        Initialise le chercheur avec un arbre et un lecteur de vecteurs.

        Args:
            tree: Racine de l'arbre K16 ou instance de K16Tree
            vectors_reader: Lecteur de vecteurs
            use_faiss: Utiliser FAISS pour accélérer la recherche
            search_type: Type de recherche - "single" ou "beam"
            beam_width: Nombre de branches à explorer en recherche par faisceau
            max_data: Nombre de vecteurs à utiliser pour le remplissage dans beam_search_tree
            use_numba: Utiliser les optimisations Numba si disponibles
        """
        # Gérer les différents types d'arbre
        if isinstance(tree, K16Tree):
            self.k16tree = tree
            self.tree = tree.root
            self.flat_tree = tree.flat_tree
        else:
            self.k16tree = None
            self.tree = tree
            self.flat_tree = None

        self.vectors_reader = vectors_reader
        self.use_faiss = use_faiss
        self.search_type = search_type
        self.beam_width = beam_width
        self.max_data = max_data
        self.faiss_available = True

        # Si nous avons une structure plate, indiquer qu'elle sera utilisée
        if self.flat_tree:
            print("✓ Structure plate optimisée détectée et activée")

        # Vérifier si FAISS est disponible
        try:
            import faiss
            self.faiss_available = True
        except ImportError:
            self.faiss_available = False
            if use_faiss:
                print("⚠️ FAISS n'est pas disponible. Utilisation de numpy à la place.")
                self.use_faiss = False
    
    def find_nearest_centroid(self, centroids: np.ndarray, query: np.ndarray) -> int:
        """
        Trouve l'indice du centroïde le plus proche du vecteur query.
        Utilise le batch processing pour le dot product.

        Args:
            centroids: Tableau des centroïdes
            query: Vecteur de requête

        Returns:
            int: Indice du centroïde le plus proche
        """
        # Utiliser le batch processing pour le calcul du produit scalaire
        similarities = np.dot(centroids, query)
        # Retourner l'indice du centroïde le plus similaire
        return np.argmax(similarities)
    
    def find_top_k_centroids(self, centroids: np.ndarray, query: np.ndarray, k: int) -> List[int]:
        """
        Trouve les k indices des centroïdes les plus proches du vecteur query.
        Utilise le batch processing pour le dot product.

        Args:
            centroids: Tableau des centroïdes
            query: Vecteur de requête
            k: Nombre de centroïdes à retourner

        Returns:
            List[int]: Liste des indices des k centroïdes les plus proches
        """
        # Pour les embeddings normalisés, utiliser le produit scalaire (similarité cosinus)
        # Batch processing du produit scalaire
        similarities = np.dot(centroids, query)
        
        # Retourner les indices des k centroïdes les plus similaires
        if k >= len(centroids):
            return list(range(len(centroids)))
        return np.argsort(similarities)[-k:][::-1].tolist()

    def search_tree(self, query: np.ndarray, k: int = 10) -> List[int]:
        """
        Recherche les k voisins les plus proches du vecteur query en descendant l'arbre.
        Utilise la stratégie configurée (single ou beam).
        Si une structure plate optimisée est disponible, l'utilise en priorité.

        Args:
            query: Vecteur de requête
            k: Nombre de voisins à retourner

        Returns:
            List[int]: Liste des indices des k voisins les plus proches
        """
        # Optimisation: Si nous avons un arbre plat, l'utiliser directement (chemin rapide)
        if self.flat_tree is not None:
            # Appel direct à la méthode search_tree avec bypass complet du pipeline traditionnel
            return self.flat_tree.search_tree(query, self.beam_width)

        # Utiliser l'implémentation standard uniquement si pas d'arbre plat
        if self.search_type == "beam":
            return self.beam_search_tree(query, k)
        else:
            return self.single_search_tree(query, k)

    def single_search_tree(self, query: np.ndarray, k: int = 10) -> List[int]:
        """
        Recherche simple : descend l'arbre en suivant une seule branche.
        Utilise les centroïdes stockés dans les nœuds sous forme de tableau aligné avec children.

        Args:
            query: Vecteur de requête
            k: Nombre de voisins à retourner

        Returns:
            List[int]: Liste des indices des k voisins les plus proches
        """
        # Descendre l'arbre jusqu'à une feuille
        node = self.tree
        path = []

        while node.children:
            # Utiliser directement le tableau de centroïdes aligné avec children
            if node.centroids is None:
                # Si centroids n'est pas initialisé, l'initialiser
                centroids = np.array([child.centroid for child in node.children])
                node.centroids = centroids
            else:
                centroids = node.centroids
                
            # Batch processing du produit scalaire
            # Calculer la similarité avec tous les centroïdes en une seule opération
            similarities = np.dot(centroids, query)
            # Trouver le centroïde avec la similarité maximale
            idx = np.argmax(similarities)
            
            # Descendre vers cet enfant
            node = node.children[idx]
            path.append(idx)

        # Nous sommes maintenant dans une feuille
        # Récupérer directement les indices pré-calculés (maintenant en numpy array)
        candidate_indices = node.indices.tolist()

        # Si aucun candidat n'est trouvé, retourner une liste vide
        if not candidate_indices:
            return []

        return candidate_indices

    def beam_search_tree(self, query: np.ndarray, k: int = 10) -> List[int]:
        """
        Recherche par faisceau : explore plusieurs branches prometteuses simultanément.
        Utilise une stratégie de remplissage pour garantir MAX_DATA candidats.
        Utilise le batch processing pour le dot product.

        Args:
            query: Vecteur de requête
            k: Nombre de voisins à retourner

        Returns:
            List[int]: Liste des indices des k voisins les plus proches
        """
        # Commencer avec la racine
        beam = [(self.tree, 1.0)]  # (node, score)
        leaves_data = []  # [(node, score, current_index)]

        # Phase 1: Descendre l'arbre et collecter toutes les feuilles
        while beam and not all(node.is_leaf() for node, _ in beam):
            next_beam = []

            for node, score in beam:
                if node.is_leaf():
                    # Stocker les feuilles avec leurs scores
                    if len(node.indices) > 0:
                        leaves_data.append((node, score, 0))  # index 0 pour commencer
                else:
                    # Explorer les k meilleures branches
                    # Utiliser le tableau de centroïdes pré-calculé si disponible
                    if node.centroids is None:
                        # Si centroids n'est pas initialisé, l'initialiser
                        centroids = np.array([child.centroid for child in node.children])
                        node.centroids = centroids
                    else:
                        centroids = node.centroids
                    
                    # Batch processing pour calculer les similarités avec tous les centroïdes
                    similarities = np.dot(centroids, query)
                    
                    # Prendre les beam_width meilleurs centroïdes
                    if self.beam_width >= len(centroids):
                        top_k_indices = np.arange(len(centroids))
                    else:
                        top_k_indices = np.argsort(-similarities)[:self.beam_width]

                    for idx in top_k_indices:
                        child = node.children[idx]
                        child_score = similarities[idx]
                        next_beam.append((child, child_score))

            # Garder les meilleures branches pour la prochaine itération
            if next_beam:
                next_beam.sort(key=lambda x: x[1], reverse=True)
                beam = next_beam[:self.beam_width]

        # Ajouter les feuilles finales
        for node, score in beam:
            if node.is_leaf() and len(node.indices) > 0:
                leaves_data.append((node, score, 0))

        # Phase 2: Stratégie de remplissage avec récursion
        all_candidates = set()

        # Utiliser MAX_DATA depuis la configuration
        max_data = self.max_data

        # Première passe : répartition égale
        if leaves_data:
            base_per_leaf = max(1, max_data // len(leaves_data))

            for i, (leaf, score, _) in enumerate(leaves_data):
                indices = leaf.indices
                n_to_take = min(base_per_leaf, len(indices))
                candidates = indices[:n_to_take].tolist()
                all_candidates.update(candidates)
                # Mettre à jour l'index courant pour cette feuille
                leaves_data[i] = (leaf, score, n_to_take)

        # Deuxième passe : remplissage récursif jusqu'à MAX_DATA
        self._fill_candidates_recursive(all_candidates, leaves_data, max_data)

        return list(all_candidates)

    def _fill_candidates_recursive(self, candidates: set, leaves_data: list, max_data: int) -> None:
        """
        Remplit récursivement les candidats jusqu'à atteindre max_data.

        Args:
            candidates: Ensemble des candidats actuels
            leaves_data: Liste des (node, score, current_index) pour chaque feuille
            max_data: Nombre cible de candidats
        """
        # Condition d'arrêt : on a assez de candidats
        if len(candidates) >= max_data:
            return

        # Condition d'arrêt : plus de candidats disponibles
        all_exhausted = True
        for leaf, _, idx in leaves_data:
            if idx < len(leaf.indices):
                all_exhausted = False
                break

        if all_exhausted:
            return

        # Trier les feuilles par score décroissant
        leaves_data.sort(key=lambda x: x[1], reverse=True)

        # Essayer d'ajouter un candidat de chaque feuille (round-robin par score)
        candidates_added = False

        for i, (leaf, score, current_idx) in enumerate(leaves_data):
            if len(candidates) >= max_data:
                break

            indices = leaf.indices.tolist()
            if current_idx < len(indices):
                # Essayer d'ajouter le prochain candidat non-dupliqué
                while current_idx < len(indices):
                    candidate = indices[current_idx]
                    if candidate not in candidates:
                        candidates.add(candidate)
                        candidates_added = True
                        leaves_data[i] = (leaf, score, current_idx + 1)
                        break
                    current_idx += 1
                    leaves_data[i] = (leaf, score, current_idx)

        # Appel récursif si on a ajouté des candidats et qu'on n'a pas atteint max_data
        if candidates_added and len(candidates) < max_data:
            self._fill_candidates_recursive(candidates, leaves_data, max_data)

    def filter_candidates(self, candidates: List[int], query: np.ndarray, k: int) -> List[int]:
        """
        Filtre les candidats pour ne garder que les k plus proches.
        Utilise le batch processing pour le dot product.
        
        Args:
            candidates: Liste des indices candidats
            query: Vecteur de requête
            k: Nombre de voisins à retourner
            
        Returns:
            List[int]: Liste des indices des k voisins les plus proches
        """
        if len(candidates) <= k:
            return candidates
        
        if self.use_faiss and self.faiss_available:
            # Filtrage avec FAISS (déjà optimisé pour le batch processing)
            # Récupérer les vecteurs candidats
            candidate_vectors = self.vectors_reader[candidates]
            
            # Créer un index FAISS pour la recherche rapide
            dimension = query.shape[0]
            index = faiss.IndexFlatIP(dimension)  # Indice de produit interne (pour vecteurs normalisés)
            index.add(candidate_vectors)
            
            # Rechercher les k plus proches dans les candidats
            D, I = index.search(query.reshape(1, -1), k)
            
            # Convertir les indices locaux en indices globaux
            results = [candidates[idx] for idx in I[0]]
        else:
            # Utiliser directement la méthode dot du VectorReader qui est déjà optimisée
            # pour le batch processing
            similarities = self.vectors_reader.dot(candidates, query)
            top_k_indices = np.argsort(-similarities)[:k]
            results = [candidates[idx] for idx in top_k_indices]
        
        return results
    
    def search_k_nearest(self, query: np.ndarray, k: int = 10) -> List[int]:
        """
        Recherche complète des k voisins les plus proches.
        Descend l'arbre, puis filtre les candidats.

        Args:
            query: Vecteur de requête
            k: Nombre de voisins à retourner

        Returns:
            List[int]: Liste des indices des k voisins les plus proches
        """
        # Chemin rapide: Si nous avons un arbre plat et beam_width > 1, utiliser directement l'arbre plat
        # sans passer par le filtrage (comme dans experiment/test_flat.py)
        if self.flat_tree is not None and self.beam_width > 1:
            return self.flat_tree.search_tree(query, self.beam_width)

        # Obtenir les candidats en descendant l'arbre
        candidates = self.search_tree(query, k)

        # Si moins de candidats que demandé, retourner tous les candidats
        if len(candidates) <= k:
            return candidates

        # Filtrer les candidats pour obtenir les k plus proches
        return self.filter_candidates(candidates, query, k)
    
    def brute_force_search(self, query: np.ndarray, k: int = 10) -> List[int]:
        """
        Recherche naïve des k voisins les plus proches avec FAISS ou numpy.
        
        Args:
            query: Vecteur de requête
            k: Nombre de voisins à retourner
            
        Returns:
            List[int]: Liste des indices des k voisins les plus proches
        """
        if self.use_faiss and self.faiss_available:
            # Utiliser un FlatIndex FAISS pour la recherche par similarité cosinus
            dimension = query.shape[0]
            
            if self.vectors_reader.mode == "ram":
                # En mode RAM, on peut facilement utiliser tous les vecteurs
                index = faiss.IndexFlatIP(dimension)  # Indice de produit interne pour similarité cosinus
                index.add(self.vectors_reader.vectors)  # Ajouter tous les vecteurs
                
                # Rechercher les k plus proches voisins
                D, I = index.search(query.reshape(1, -1), k)
                return I[0].tolist()
            else:
                # En mode mmap, faire la recherche par lots pour éviter de charger tous les vecteurs
                batch_size = 100000  # Taille de lot plus grande, FAISS étant plus efficace
                n_batches = (len(self.vectors_reader) + batch_size - 1) // batch_size
                
                # Garder les k indices les plus similaires
                faiss_index = faiss.IndexFlatIP(dimension)
                final_indices = []
                final_distances = []
                
                for i in range(n_batches):
                    start_idx = i * batch_size
                    end_idx = min((i + 1) * batch_size, len(self.vectors_reader))
                    batch_indices = list(range(start_idx, end_idx))
                    
                    # Récupérer les vecteurs du lot
                    batch_vectors = self.vectors_reader[batch_indices]
                    
                    # Créer un index FAISS temporaire pour ce lot
                    index_batch = faiss.IndexFlatIP(dimension)
                    index_batch.add(batch_vectors)
                    
                    # Rechercher les k plus proches voisins dans ce lot
                    D, I = index_batch.search(query.reshape(1, -1), min(k, len(batch_indices)))
                    
                    # Convertir les indices locaux en indices globaux
                    global_indices = [batch_indices[idx] for idx in I[0]]
                    
                    # Ajouter ces vecteurs à l'index global
                    for idx, dist in zip(global_indices, D[0]):
                        final_indices.append(idx)
                        final_distances.append(dist)
                    
                    # Si nous avons plus de k résultats, garder uniquement les k meilleurs
                    if len(final_indices) >= 2*k:
                        # Trier par distance décroissante (similarité cosinus)
                        sorted_pairs = sorted(zip(final_indices, final_distances), key=lambda x: x[1], reverse=True)
                        # Garder les k meilleurs
                        final_indices = [idx for idx, _ in sorted_pairs[:k]]
                        final_distances = [dist for _, dist in sorted_pairs[:k]]
                
                # Faire un tri final pour avoir les k plus proches
                if len(final_indices) > k:
                    sorted_pairs = sorted(zip(final_indices, final_distances), key=lambda x: x[1], reverse=True)
                    final_indices = [idx for idx, _ in sorted_pairs[:k]]
                
                return final_indices
        else:
            # Recherche naïve avec numpy - utiliser des opérations vectorisées
            if self.vectors_reader.mode == "ram":
                # Calculer la similarité cosinus avec tous les vecteurs en une seule opération
                similarities = self.vectors_reader.dot(list(range(len(self.vectors_reader))), query)
                
                # Retourner les indices des k vecteurs les plus similaires
                return np.argsort(-similarities)[:k].tolist()
            else:
                # En mmap, faire le calcul par lots mais utiliser des opérations vectorisées
                batch_size = 10000
                n_batches = (len(self.vectors_reader) + batch_size - 1) // batch_size
                
                # Garder les k indices les plus similaires
                top_k_indices = np.array([], dtype=int)
                top_k_similarities = np.array([], dtype=np.float32)
                
                for i in range(n_batches):
                    start_idx = i * batch_size
                    end_idx = min((i + 1) * batch_size, len(self.vectors_reader))
                    batch_indices = list(range(start_idx, end_idx))
                    
                    # Calculer la similarité pour ce lot - dot est déjà optimisé pour le batch processing
                    batch_similarities = self.vectors_reader.dot(batch_indices, query)
                    
                    # Trouver les indices locaux des k plus similaires dans ce lot
                    local_top_indices = np.argsort(-batch_similarities)[:k]
                    local_top_similarities = batch_similarities[local_top_indices]
                    
                    # Convertir les indices locaux en indices globaux
                    global_top_indices = np.array([batch_indices[idx] for idx in local_top_indices])
                    
                    # Mettre à jour les top k
                    if len(top_k_indices) > 0:
                        combined_indices = np.concatenate([top_k_indices, global_top_indices])
                        combined_similarities = np.concatenate([top_k_similarities, local_top_similarities])
                    else:
                        combined_indices = global_top_indices
                        combined_similarities = local_top_similarities
                    
                    # Trier et garder les k meilleurs
                    top_k_positions = np.argsort(-combined_similarities)[:k]
                    top_k_indices = np.array([combined_indices[i] for i in top_k_positions])
                    top_k_similarities = combined_similarities[top_k_positions]
                
                return top_k_indices.tolist()
    
    def evaluate_search(self, queries: np.ndarray, k: int = 10) -> Dict[str, Any]:
        """
        Évalue les performances de la recherche dans l'arbre par rapport à la recherche naïve.

        Args:
            queries: Vecteurs requêtes
            k: Nombre de voisins à retourner

        Returns:
            Dict[str, Any]: Dictionnaire de métriques de performance
        """
        search_type_desc = f"{self.search_type} (beam_width={self.beam_width})" if self.search_type == "beam" else self.search_type
        flat_tree_desc = " avec arbre plat" if self.flat_tree is not None else ""
        print(f"\n⏳ Évaluation avec {len(queries)} requêtes, k={k}, type de recherche: {search_type_desc}{flat_tree_desc}...")

        tree_search_time = 0
        tree_filter_time = 0
        naive_search_time = 0
        recall_sum = 0
        candidates_count = []

        from tqdm.auto import tqdm

        # Optimisation: mode direct pour arbre plat
        use_direct_mode = False

        for i, query in enumerate(tqdm(queries, desc="Évaluation")):
            # Recherche optimisée avec l'arbre
            if use_direct_mode:
                # Chemin ultra-rapide: appel direct au flat_tree
                start_time = time.time()
                tree_results = self.flat_tree.search_tree(query, self.beam_width)
                total_search_time = time.time() - start_time
                tree_search_time += total_search_time
                tree_filter_time += 0  # Pas de filtrage
                candidates_count.append(len(tree_results))
            else:
                # Chemin standard avec tree_search puis filter
                start_time = time.time()
                tree_candidates = self.search_tree(query)
                tree_search_time += time.time() - start_time

                # Stocker le nombre de candidats pour les statistiques
                candidates_count.append(len(tree_candidates))

                # Filtrer les candidats pour ne garder que les k plus proches
                filter_start_time = time.time()
                tree_results = self.filter_candidates(tree_candidates, query, k)
                tree_filter_time += time.time() - filter_start_time

            # Recherche naïve
            start_time = time.time()
            naive_results = self.brute_force_search(query, k)
            naive_search_time += time.time() - start_time

            # Calcul du recall (combien de vrais voisins sont trouvés)
            intersection = set(tree_results).intersection(set(naive_results))
            recall = len(intersection) / k if k > 0 else 0
            recall_sum += recall

            if (i + 1) % 10 == 0 or (i + 1) == len(queries):
                print(f"  → Requête {i+1}/{len(queries)}: Recall = {recall:.4f}, Candidats = {len(tree_results)}")

        # Moyennes
        avg_tree_time = tree_search_time / len(queries)
        avg_filter_time = tree_filter_time / len(queries)
        avg_total_time = avg_tree_time + avg_filter_time
        avg_naive_time = naive_search_time / len(queries)
        avg_recall = recall_sum / len(queries)
        speedup = avg_naive_time / avg_total_time if avg_total_time > 0 else 0

        avg_candidates = sum(candidates_count) / len(candidates_count) if candidates_count else 0
        min_candidates = min(candidates_count) if candidates_count else 0
        max_candidates = max(candidates_count) if candidates_count else 0

        print("\n✓ Résultats de l'évaluation:")
        print(f"  - Nombre de requêtes     : {len(queries)}")
        print(f"  - k (voisins demandés)   : {k}")
        print(f"  - Mode                   : {self.vectors_reader.mode.upper()}")
        print(f"  - Type de recherche      : {search_type_desc}")
        print(f"  - Statistiques candidats:")
        print(f"    • Moyenne             : {avg_candidates:.1f}")
        print(f"    • Minimum             : {min_candidates}")
        print(f"    • Maximum             : {max_candidates}")
        print(f"  - Temps moyen (arbre)    : {avg_tree_time*1000:.2f} ms")
        print(f"  - Temps moyen (filtre)   : {avg_filter_time*1000:.2f} ms")
        print(f"  - Temps moyen (total)    : {avg_total_time*1000:.2f} ms")
        print(f"  - Temps moyen (naïf)     : {avg_naive_time*1000:.2f} ms")
        print(f"  - Accélération           : {speedup:.2f}x")
        print(f"  - Recall moyen           : {avg_recall:.4f} ({avg_recall*100:.2f}%)")

        return {
            "search_type": self.search_type,
            "beam_width": self.beam_width if self.search_type == "beam" else None,
            "avg_tree_time": avg_tree_time,
            "avg_filter_time": avg_filter_time,
            "avg_total_time": avg_total_time,
            "avg_naive_time": avg_naive_time,
            "speedup": speedup,
            "avg_recall": avg_recall,
            "avg_candidates": avg_candidates,
            "min_candidates": min_candidates,
            "max_candidates": max_candidates
        }