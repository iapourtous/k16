"""
Module de recherche pour K16.
Interface simplifiée pour TreeFlat compressé avec Numba JIT.
"""

import numpy as np
import time
from typing import List, Dict, Any
import faiss

from .tree import K16Tree
from .io import VectorReader

class Searcher:
    """
    Classe principale pour la recherche TreeFlat compressée.
    Utilise uniquement la structure plate optimisée avec compression et Numba JIT.
    """

    def __init__(self, k16tree: K16Tree, vectors_reader: VectorReader, use_faiss: bool = True,
                 search_type: str = "single", beam_width: int = 3, max_data: int = 4000):
        """
        Initialise le chercheur avec TreeFlat uniquement.

        Args:
            k16tree: Instance de K16Tree avec flat_tree
            vectors_reader: Lecteur de vecteurs
            use_faiss: Utiliser FAISS pour accélérer la recherche
            search_type: Type de recherche - "single" ou "beam"
            beam_width: Nombre de branches à explorer en recherche par faisceau
            max_data: Nombre de vecteurs à utiliser pour le remplissage
        """
        # TreeFlat uniquement
        self.k16tree = k16tree
        self.flat_tree = k16tree.flat_tree

        if self.flat_tree is None:
            raise ValueError("TreeFlat requis - structure non-plate non supportée")

        self.vectors_reader = vectors_reader
        self.use_faiss = use_faiss
        self.search_type = search_type
        self.beam_width = beam_width
        self.max_data = max_data
        self.faiss_available = True

        print("✓ Structure TreeFlat compressée activée")

        # Vérifier si FAISS est disponible
        try:
            import faiss
            self.faiss_available = True
        except ImportError:
            self.faiss_available = False
            if use_faiss:
                print("⚠️ FAISS n'est pas disponible. Utilisation de numpy à la place.")
                self.use_faiss = False

    def search_tree(self, query: np.ndarray) -> List[int]:
        """
        Recherche TreeFlat - retourne les candidats bruts.

        Args:
            query: Vecteur de requête

        Returns:
            List[int]: Liste des indices candidats
        """
        if self.search_type == "beam":
            return self.flat_tree.search_tree_beam(query, self.beam_width)
        else:
            return self.flat_tree.search_tree_single(query)

    def filter_candidates(self, candidates: List[int], query: np.ndarray, k: int) -> List[int]:
        """
        Filtre les candidats pour ne garder que les k plus proches.
        
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
            # Filtrage avec FAISS
            candidate_vectors = self.vectors_reader[candidates]
            dimension = query.shape[0]
            index = faiss.IndexFlatIP(dimension)
            index.add(candidate_vectors)
            
            D, I = index.search(query.reshape(1, -1), k)
            results = [candidates[idx] for idx in I[0]]
        else:
            # Utiliser VectorReader optimisé
            similarities = self.vectors_reader.dot(candidates, query)
            top_k_indices = np.argsort(-similarities)[:k]
            results = [candidates[idx] for idx in top_k_indices]
        
        return results
    
    def brute_force_search(self, query: np.ndarray, k: int = 10) -> List[int]:
        """
        Recherche naïve des k voisins les plus proches.
        
        Args:
            query: Vecteur de requête
            k: Nombre de voisins à retourner
            
        Returns:
            List[int]: Liste des indices des k voisins les plus proches
        """
        if self.use_faiss and self.faiss_available:
            dimension = query.shape[0]
            
            if self.vectors_reader.mode == "ram":
                index = faiss.IndexFlatIP(dimension)
                index.add(self.vectors_reader.vectors)
                D, I = index.search(query.reshape(1, -1), k)
                return I[0].tolist()
            else:
                # Mode mmap par lots
                batch_size = 100000
                n_batches = (len(self.vectors_reader) + batch_size - 1) // batch_size
                
                final_indices = []
                final_distances = []
                
                for i in range(n_batches):
                    start_idx = i * batch_size
                    end_idx = min((i + 1) * batch_size, len(self.vectors_reader))
                    batch_indices = list(range(start_idx, end_idx))
                    
                    batch_vectors = self.vectors_reader[batch_indices]
                    index_batch = faiss.IndexFlatIP(dimension)
                    index_batch.add(batch_vectors)
                    
                    D, I = index_batch.search(query.reshape(1, -1), min(k, len(batch_indices)))
                    
                    global_indices = [batch_indices[idx] for idx in I[0]]
                    
                    for idx, dist in zip(global_indices, D[0]):
                        final_indices.append(idx)
                        final_distances.append(dist)
                    
                    if len(final_indices) >= 2*k:
                        sorted_pairs = sorted(zip(final_indices, final_distances), key=lambda x: x[1], reverse=True)
                        final_indices = [idx for idx, _ in sorted_pairs[:k]]
                        final_distances = [dist for _, dist in sorted_pairs[:k]]
                
                if len(final_indices) > k:
                    sorted_pairs = sorted(zip(final_indices, final_distances), key=lambda x: x[1], reverse=True)
                    final_indices = [idx for idx, _ in sorted_pairs[:k]]
                
                return final_indices
        else:
            # Recherche numpy
            if self.vectors_reader.mode == "ram":
                similarities = self.vectors_reader.dot(list(range(len(self.vectors_reader))), query)
                return np.argsort(-similarities)[:k].tolist()
            else:
                # Mode mmap par lots
                batch_size = 10000
                n_batches = (len(self.vectors_reader) + batch_size - 1) // batch_size
                
                top_k_indices = np.array([], dtype=int)
                top_k_similarities = np.array([], dtype=np.float32)
                
                for i in range(n_batches):
                    start_idx = i * batch_size
                    end_idx = min((i + 1) * batch_size, len(self.vectors_reader))
                    batch_indices = list(range(start_idx, end_idx))
                    
                    batch_similarities = self.vectors_reader.dot(batch_indices, query)
                    local_top_indices = np.argsort(-batch_similarities)[:k]
                    local_top_similarities = batch_similarities[local_top_indices]
                    
                    global_top_indices = np.array([batch_indices[idx] for idx in local_top_indices])
                    
                    if len(top_k_indices) > 0:
                        combined_indices = np.concatenate([top_k_indices, global_top_indices])
                        combined_similarities = np.concatenate([top_k_similarities, local_top_similarities])
                    else:
                        combined_indices = global_top_indices
                        combined_similarities = local_top_similarities
                    
                    top_k_positions = np.argsort(-combined_similarities)[:k]
                    top_k_indices = np.array([combined_indices[i] for i in top_k_positions])
                    top_k_similarities = combined_similarities[top_k_positions]
                
                return top_k_indices.tolist()
    
    def evaluate_search(self, queries: np.ndarray, k: int = 10) -> Dict[str, Any]:
        """
        Évalue les performances TreeFlat vs recherche naïve.

        Args:
            queries: Vecteurs requêtes
            k: Nombre de voisins à retourner

        Returns:
            Dict[str, Any]: Dictionnaire de métriques de performance
        """
        search_type_desc = f"{self.search_type} (beam_width={self.beam_width})" if self.search_type == "beam" else self.search_type
        print(f"\n⏳ Évaluation avec {len(queries)} requêtes, k={k}, type de recherche: {search_type_desc} avec TreeFlat...")

        tree_search_time = 0
        tree_filter_time = 0
        naive_search_time = 0
        recall_sum = 0
        candidates_count = []

        from tqdm.auto import tqdm

        # Warmup Numba JIT
        print("🔥 Warmup Numba JIT compilation...")
        warmup_query = queries[0]
        for _ in range(3):
            _ = self.flat_tree.search_tree_single(warmup_query)

        for i, query in enumerate(tqdm(queries, desc="Évaluation")):
            # Recherche TreeFlat
            start_time = time.time()
            tree_candidates = self.search_tree(query)
            tree_search_time += time.time() - start_time

            candidates_count.append(len(tree_candidates))

            # Filtrage
            filter_start_time = time.time()
            tree_results = self.filter_candidates(tree_candidates, query, k)
            tree_filter_time += time.time() - filter_start_time

            # Recherche naïve
            start_time = time.time()
            naive_results = self.brute_force_search(query, k)
            naive_search_time += time.time() - start_time

            # Calcul du recall
            intersection = set(tree_results).intersection(set(naive_results))
            recall = len(intersection) / k if k > 0 else 0
            recall_sum += recall

            if (i + 1) % 10 == 0 or (i + 1) == len(queries):
                print(f"  → Requête {i+1}/{len(queries)}: Recall = {recall:.4f}, Candidats = {len(tree_candidates)}")

        # Calcul des moyennes
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