### k16/cli.py

```python
"""
Interface en ligne de commande pour K16.
Fournit des commandes pour télécharger des données, construire des arbres,
tester les performances et effectuer des recherches interactives.
"""

import os
import sys
import argparse

from k16.utils.config import ConfigManager
from k16.utils.cli_build import build_command
from k16.utils.cli_test import test_command
from k16.utils.cli_get_data import get_data_command
from k16.utils.cli_search import search_command
from k16.utils.cli_api import api_command

def main() -> int:
    """
    Point d'entrée principal pour l'interface en ligne de commande.
    
    Returns:
        int: Code de retour (0 pour succès, autre pour erreur)
    """
    # Initialisation du gestionnaire de configuration par défaut
    config_manager = ConfigManager()

    # Récupération des paramètres pour la configuration par défaut
    build_config = config_manager.get_section("build_tree")
    search_config = config_manager.get_section("search")
    files_config = config_manager.get_section("files")
    flat_tree_config = config_manager.get_section("flat_tree")
    prepare_data_config = config_manager.get_section("prepare_data")

    # Définir les chemins par défaut
    default_vectors_path = os.path.join(files_config["vectors_dir"], files_config["default_vectors"])
    default_tree_path = os.path.join(files_config["trees_dir"], files_config["default_tree"])
    default_qa_path = os.path.join(files_config["vectors_dir"], files_config.get("default_qa", "qa.txt"))

    # Parseur principal
    parser = argparse.ArgumentParser(
        description="K16 - Bibliothèque pour la recherche rapide de vecteurs d'embedding similaires",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--config", default=config_manager.config_path,
                        help=f"Chemin vers le fichier de configuration")
    parser.add_argument("--version", action="version", version="K16 v1.0.0")

    # Sous-parseurs pour les différentes commandes
    subparsers = parser.add_subparsers(dest="command", help="Commandes disponibles")

    # Commande API
    api_parser = subparsers.add_parser("api", help="Démarrer l'API de recherche")
    api_parser.add_argument("--host", default=None,
                        help="Adresse d'hôte pour l'API (par défaut: 127.0.0.1)")
    api_parser.add_argument("--port", type=int, default=None,
                        help="Port pour l'API (par défaut: 8000)")
    api_parser.add_argument("--reload", action="store_true", default=None,
                        help="Activer le rechargement automatique en cas de modification du code")
    api_parser.set_defaults(func=api_command)

    # Commande build
    build_parser = subparsers.add_parser("build", help="Construire un arbre optimisé")
    build_parser.add_argument("vectors_file", nargs="?", default=default_vectors_path,
                       help=f"Fichier binaire contenant les vecteurs embeddings")
    build_parser.add_argument("tree_file", nargs="?", default=default_tree_path,
                       help=f"Fichier de sortie pour l'arbre")
    build_parser.add_argument("--max_depth", type=int, default=build_config["max_depth"],
                       help=f"Profondeur maximale de l'arbre")
    build_parser.add_argument("--k", type=int, default=build_config["k"],
                       help=f"Nombre de branches par nœud")
    build_parser.add_argument("--k_adaptive", action="store_true", default=False,
                       help="Utiliser la méthode du coude pour déterminer k automatiquement")
    build_parser.add_argument("--max_leaf_size", type=int, default=build_config["max_leaf_size"],
                       help=f"Taille maximale d'une feuille pour l'arrêt de la subdivision")
    build_parser.add_argument("--max_data", type=int, default=build_config["max_data"],
                       help=f"MAX_DATA: Nombre de vecteurs à stocker dans chaque feuille")
    build_parser.add_argument("--max_dims", type=int, default=flat_tree_config.get("max_dims", 128),
                       help=f"Nombre de dimensions à conserver pour la réduction dimensionnelle")
    build_parser.add_argument("--hnsw", action="store_true", default=build_config.get("use_hnsw_improvement", True),
                       help="Activer l'amélioration des candidats par HNSW après construction")
    build_parser.set_defaults(func=build_command)

    # Commande getData
    data_parser = subparsers.add_parser("getData", help="Télécharger et préparer les données")
    data_parser.add_argument("out_text", nargs="?", default=default_qa_path,
                      help=f"Fichier texte QA (par défaut: {default_qa_path})")
    data_parser.add_argument("out_vec", nargs="?", default=default_vectors_path,
                      help=f"Fichier binaire embeddings (par défaut: {default_vectors_path})")
    data_parser.add_argument("--model", default=prepare_data_config.get("model", "intfloat/multilingual-e5-large"),
                      help=f"Modèle d'embedding à utiliser (par défaut: {prepare_data_config.get('model', 'intfloat/multilingual-e5-large')})")
    data_parser.add_argument("--batch-size", type=int, default=prepare_data_config.get("batch_size", 128),
                      help=f"Taille des lots pour l'encodage (par défaut: {prepare_data_config.get('batch_size', 128)})")
    data_parser.add_argument("--force", action="store_true", default=False,
                      help="Forcer le recalcul des embeddings même si le fichier existe déjà")
    data_parser.set_defaults(func=get_data_command)

    # Commande test
    test_parser = subparsers.add_parser("test", help="Tester la performance de la recherche")
    test_parser.add_argument("vectors_file", nargs="?", default=default_vectors_path,
                       help=f"Fichier binaire contenant les vecteurs embeddings")
    test_parser.add_argument("tree_file", nargs="?", default=default_tree_path,
                       help=f"Fichier de l'arbre à utiliser pour la recherche")
    test_parser.add_argument("--k", type=int, default=search_config["k"],
                       help=f"Nombre de voisins à retourner")
    test_parser.add_argument("--mode", choices=["ram", "mmap"], default=search_config["mode"],
                       help=f"Mode de chargement des vecteurs")
    test_parser.add_argument("--cache_size", type=int, default=search_config["cache_size_mb"],
                       help=f"Taille du cache en MB pour le mode mmap")
    test_parser.add_argument("--queries", type=int, default=search_config["queries"],
                       help=f"Nombre de requêtes aléatoires à effectuer")
    test_parser.add_argument("--search_type", choices=["single", "beam"], default=search_config["search_type"],
                       help=f"Type de recherche dans l'arbre")
    test_parser.add_argument("--beam_width", type=int, default=search_config.get("beam_width", 3),
                       help=f"Largeur du faisceau pour la recherche beam")
    test_parser.add_argument("--use_faiss", action="store_true", default=search_config["use_faiss"],
                       help=f"Utiliser FAISS pour accélérer le filtrage final")
    test_parser.add_argument("--evaluate", action="store_true", default=True,
                       help=f"Évaluer les performances (recall et accélération)")
    test_parser.add_argument("--no-evaluate", dest="evaluate", action="store_false",
                       help=f"Désactiver l'évaluation des performances")
    test_parser.set_defaults(func=test_command)
    
    # Commande search
    search_parser = subparsers.add_parser("search", help="Recherche interactive en ligne de commande")
    search_parser.add_argument("vectors_file", nargs="?", default=default_vectors_path,
                         help=f"Fichier binaire contenant les vecteurs embeddings")
    search_parser.add_argument("tree_file", nargs="?", default=default_tree_path,
                         help=f"Fichier de l'arbre à utiliser pour la recherche")
    search_parser.add_argument("qa_file", nargs="?", default=default_qa_path,
                         help=f"Fichier contenant les questions et réponses")
    search_parser.add_argument("--k", type=int, default=search_config["k"],
                         help=f"Nombre de résultats à afficher")
    search_parser.add_argument("--mode", choices=["ram", "mmap"], default=search_config["mode"],
                         help=f"Mode de chargement des vecteurs")
    search_parser.add_argument("--cache_size", type=int, default=search_config["cache_size_mb"],
                         help=f"Taille du cache en MB pour le mode mmap")
    search_parser.add_argument("--search_type", choices=["single", "beam"], default=search_config["search_type"],
                         help=f"Type de recherche dans l'arbre")
    search_parser.add_argument("--beam_width", type=int, default=search_config.get("beam_width", 3),
                         help=f"Largeur du faisceau pour la recherche beam")
    search_parser.add_argument("--use_faiss", action="store_true", default=search_config["use_faiss"],
                         help=f"Utiliser FAISS pour accélérer le filtrage final")
    search_parser.add_argument("--model", default=prepare_data_config.get("model", "intfloat/multilingual-e5-large"),
                         help=f"Modèle d'embedding à utiliser")
    search_parser.set_defaults(func=search_command)

    # Traitement des arguments
    args = parser.parse_args()

    # Exécution de la commande spécifiée
    if hasattr(args, "func"):
        return args.func(args)
    else:
        parser.print_help()
        return 0

if __name__ == "__main__":
    sys.exit(main())
```

### k16/search/searcher.py

```python
"""
Module de recherche pour K16.
Interface simplifiée pour TreeFlat compressé avec Numba JIT.
"""

import numpy as np
import time
from typing import List, Dict, Any, Tuple, Optional
import faiss

from k16.core.tree import K16Tree
from k16.io.reader import VectorReader

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

        try:
            from tqdm.auto import tqdm
            use_tqdm = True
        except ImportError:
            use_tqdm = False
            print("Info: tqdm not available, progress will not be shown")

        # Warmup Numba JIT
        print("🔥 Warmup Numba JIT compilation...")
        warmup_query = queries[0]
        for _ in range(3):
            _ = self.flat_tree.search_tree_single(warmup_query)

        if use_tqdm:
            queries_iter = tqdm(queries, desc="Évaluation")
        else:
            queries_iter = queries
            print(f"⏳ Traitement de {len(queries)} requêtes...")

        for i, query in enumerate(queries_iter):
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

            if not use_tqdm and ((i + 1) % 10 == 0 or (i + 1) == len(queries)):
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

    def search(self, query: np.ndarray, k: int = 10) -> Tuple[List[int], List[float]]:
        """
        Effectue une recherche des k plus proches voisins.

        Args:
            query: Vecteur de requête
            k: Nombre de voisins à retourner

        Returns:
            Tuple[List[int], List[float]]: Tuple contenant (indices, scores)
        """
        # Normaliser la requête pour des résultats cohérents
        query_norm = np.linalg.norm(query)
        if query_norm > 0:
            query = query / query_norm

        # Trouver les candidats avec l'arbre
        candidates = self.search_tree(query)

        # Filtrer les candidats pour ne garder que les k plus proches
        top_indices = self.filter_candidates(candidates, query, k)

        # Calculer les scores de similarité
        top_vectors = self.vectors_reader[top_indices]
        scores = [np.dot(query, top_vectors[i]) for i in range(len(top_indices))]

        # Trier par score décroissant
        sorted_pairs = sorted(zip(top_indices, scores), key=lambda x: x[1], reverse=True)
        indices = [idx for idx, _ in sorted_pairs]
        scores = [score for _, score in sorted_pairs]

        return indices, scores

def search(tree: K16Tree, vectors_reader: VectorReader, query: np.ndarray, k: int = 10, 
           use_faiss: bool = True, search_type: str = "beam", beam_width: int = 3) -> Tuple[List[int], List[float]]:
    """
    Fonction utilitaire pour effectuer une recherche avec K16.

    Args:
        tree: Instance de K16Tree
        vectors_reader: Lecteur de vecteurs
        query: Vecteur de requête
        k: Nombre de voisins à retourner
        use_faiss: Utiliser FAISS pour accélérer la recherche
        search_type: Type de recherche - "single" ou "beam"
        beam_width: Nombre de branches à explorer en recherche par faisceau

    Returns:
        Tuple[List[int], List[float]]: Tuple contenant (indices, scores)
    """
    searcher = Searcher(
        k16tree=tree,
        vectors_reader=vectors_reader,
        use_faiss=use_faiss,
        search_type=search_type,
        beam_width=beam_width
    )
    return searcher.search(query, k)
```

### k16/io/writer.py

```python
"""
Module d'écriture de vecteurs et d'arbres pour K16.
Fournit des classes optimisées pour sauvegarder des vecteurs et des arbres.
"""

import os
import struct
import time
import numpy as np
from typing import Optional

from k16.core.tree import K16Tree

class VectorWriter:
    """Classe pour écrire des vecteurs dans un fichier binaire."""
    
    @staticmethod
    def write_bin(vectors: np.ndarray, file_path: str) -> None:
        """
        Écrit des vecteurs dans un fichier binaire.
        Format: header (n, d: uint64) suivi des données en float32.
        
        Args:
            vectors: Tableau numpy contenant les vecteurs (shape: [n, d])
            file_path: Chemin du fichier de sortie
        """
        n, d = vectors.shape
        start_time = time.time()
        print(f"⏳ Écriture de {n:,} vecteurs (dim {d}) vers {file_path}...")
        
        # Créer le répertoire si nécessaire
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        
        # Convertir en float32 si ce n'est pas déjà le cas
        vectors_float32 = vectors.astype(np.float32)
        
        with open(file_path, "wb") as f:
            # Écrire l'entête: nombre de vecteurs (n) et dimension (d)
            f.write(struct.pack("<QQ", n, d))
            # Écrire les données
            f.write(vectors_float32.tobytes())
        
        elapsed = time.time() - start_time
        print(f"✓ {n:,} vecteurs (dim {d}) écrits dans {file_path} [terminé en {elapsed:.2f}s]")

def write_vectors(vectors: np.ndarray, file_path: str) -> None:
    """
    Fonction utilitaire pour écrire des vecteurs dans un fichier.
    
    Args:
        vectors: Tableau numpy contenant les vecteurs (shape: [n, d])
        file_path: Chemin du fichier de sortie
    """
    VectorWriter.write_bin(vectors, file_path)

def write_tree(tree: K16Tree, file_path: str, mmap_dir: bool = False, minimal: bool = True) -> None:
    """
    Sauvegarde un arbre K16 au format plat optimisé.
    
    Args:
        tree: L'arbre K16 à sauvegarder
        file_path: Chemin du fichier de sortie
        mmap_dir: Si True, crée un répertoire de fichiers numpy pour le memory-mapping
        minimal: Si True (par défaut), sauvegarde uniquement les structures essentielles
    """
    # Vérifier que l'arbre a une structure plate
    if tree.flat_tree is None:
        raise ValueError("L'arbre doit avoir une structure plate (flat_tree) pour être sauvegardé")
    
    # Déterminer le chemin de sauvegarde
    if not file_path.endswith('.flat.npy'):
        flat_path = os.path.splitext(file_path)[0] + '.flat.npy'
    else:
        flat_path = file_path
    
    print(f"⏳ Sauvegarde de la structure plate vers {flat_path}...")
    
    # Créer le répertoire si nécessaire
    os.makedirs(os.path.dirname(os.path.abspath(flat_path)), exist_ok=True)
    
    # Sauvegarder la structure plate
    tree.flat_tree.save(flat_path, mmap_dir=mmap_dir, minimal=minimal)
    
    print(f"✓ Structure plate sauvegardée vers {flat_path}")
    
    return flat_path
```

### k16/io/reader.py

```python
"""
Module de lecture de vecteurs et d'arbres pour K16.
Fournit des classes optimisées pour charger des vecteurs et des arbres.
"""

import os
import struct
import time
import numpy as np
import mmap
import sys
from typing import Tuple, List, Dict, Any, Optional, Union
from collections import OrderedDict

from k16.core.tree import K16Tree
from k16.utils.config import ConfigManager
from k16.core.flat_tree import TreeFlat

class VectorReader:
    """
    Classe pour lire des vecteurs depuis un fichier binaire.
    Supporte deux modes: RAM et mmap.
    """
    
    def __init__(self, file_path: str, mode: str = "ram", cache_size_mb: Optional[int] = None):
        """
        Initialise le lecteur de vecteurs.
        
        Args:
            file_path: Chemin vers le fichier binaire contenant les vecteurs
            mode: Mode de lecture - "ram" (défaut) ou "mmap"
            cache_size_mb: Taille du cache en mégaoctets pour le mode mmap (si None, utilise la valeur de ConfigManager)
        """
        self.file_path = file_path
        self.mode = mode.lower()
        
        if self.mode not in ["ram", "mmap"]:
            raise ValueError("Mode doit être 'ram', 'mmap'")
        
        # Paramètres des vecteurs
        self.n = 0  # Nombre de vecteurs
        self.d = 0  # Dimension des vecteurs
        self.header_size = 16  # 2 entiers 64 bits
        self.vector_size = 0  # Taille d'un vecteur en octets
        
        # Stockage des vecteurs
        self.vectors = None  # Pour le mode RAM
        self.mmap_file = None  # Pour le mode mmap
        self.mmap_obj = None  # Pour le mode mmap
        
        # Cache LRU pour le mode mmap
        self.cache_enabled = False
        self.cache_size_mb = cache_size_mb
        if self.cache_size_mb is None:
            # Obtenir la valeur par défaut de la configuration
            config_manager = ConfigManager()
            self.cache_size_mb = config_manager.get("search", "cache_size_mb", 500)
        
        self.cache = None
        self.cache_hits = 0
        self.cache_misses = 0
        self.max_cache_size = 0
        self.cache_capacity = 0
        
        # Charger les vecteurs
        self._load_vectors()
    
    def _load_vectors(self) -> None:
        """Charge les vecteurs selon le mode choisi."""
        start_time = time.time()
        print(f"⏳ Chargement des vecteurs depuis {self.file_path} en mode {self.mode.upper()}...")
        
        with open(self.file_path, "rb") as f:
            # Lire l'en-tête (nombre et dimension des vecteurs)
            self.n, self.d = struct.unpack("<QQ", f.read(self.header_size))
            print(f"  → Format détecté: {self.n:,} vecteurs de dimension {self.d}")
            
            # Calculer la taille d'un vecteur en octets
            self.vector_size = self.d * 4  # float32 = 4 octets
            
            if self.mode == "ram":
                # Mode RAM: charger tous les vecteurs en mémoire
                buffer = f.read(self.n * self.vector_size)
                self.vectors = np.frombuffer(buffer, dtype=np.float32).reshape(self.n, self.d)
            else:
                # Mode mmap: mapper le fichier en mémoire
                self.mmap_file = open(self.file_path, "rb")
                self.mmap_obj = mmap.mmap(self.mmap_file.fileno(), 0, access=mmap.ACCESS_READ)
                
                # Initialiser le cache LRU
                self.max_cache_size = self.cache_size_mb * 1024 * 1024  # Convertir MB en octets
                self.cache = OrderedDict()  # Cache LRU
                self.cache_enabled = True
                
                # Limiter la taille du cache si trop grande
                max_possible_vectors = self.n
                max_possible_size = max_possible_vectors * self.vector_size
                if self.max_cache_size > max_possible_size:
                    adjusted_size = max_possible_size
                    adjusted_mb = adjusted_size / (1024 * 1024)
                    print(f"  → Cache limité à {adjusted_mb:.1f} MB (taille totale des données)")
                    self.max_cache_size = adjusted_size
                
                # Calculer combien de vecteurs peuvent tenir dans le cache
                self.cache_capacity = self.max_cache_size // self.vector_size
                print(f"  → Cache LRU initialisé: {self.cache_size_mb} MB ({self.cache_capacity:,} vecteurs)")
        
        elapsed = time.time() - start_time
        
        if self.mode == "ram":
            memory_usage = f"{self.vectors.nbytes / (1024**2):.1f} MB"
        else:
            memory_usage = f"Mmap + Cache {self.cache_size_mb} MB"
            
        print(f"✓ {self.n:,} vecteurs (dim {self.d}) prêts en mode {self.mode.upper()} [terminé en {elapsed:.2f}s]")
        print(f"  → Mémoire utilisée: {memory_usage}")
    
    def _update_cache_stats(self, force: bool = False) -> bool:
        """
        Affiche les statistiques du cache si assez d'accès ont été effectués.
        
        Args:
            force: Si True, affiche les statistiques même si le seuil n'est pas atteint
            
        Returns:
            bool: True si les statistiques ont été affichées, False sinon
        """
        if not self.cache_enabled:
            return False
            
        total = self.cache_hits + self.cache_misses
        
        # Afficher les stats tous les 100000 accès ou si forcé
        if force or (total > 0 and total % 100000 == 0):
            hit_rate = self.cache_hits / total * 100 if total > 0 else 0
            cache_usage = len(self.cache) / self.cache_capacity * 100 if self.cache_capacity > 0 else 0
            print(f"  → Cache stats: {hit_rate:.1f}% hits, {cache_usage:.1f}% rempli "
                  f"({len(self.cache):,}/{self.cache_capacity:,} vecteurs)")
            return True
        return False
    
    def __getitem__(self, index):
        """
        Récupère un ou plusieurs vecteurs par leur indice, avec cache LRU en mode mmap.
        
        Args:
            index: Un entier, une liste d'entiers ou un slice
            
        Returns:
            Un vecteur numpy ou un tableau de vecteurs
        """
        if self.mode == "ram":
            # En mode RAM, juste indexer le tableau numpy
            return self.vectors[index]
        else:
            # En mode mmap, utiliser le cache si disponible, sinon lire depuis le fichier
            if isinstance(index, (int, np.integer)):
                # Cas d'un seul indice - vérifier le cache d'abord
                if self.cache_enabled and index in self.cache:
                    # Cache hit
                    self.cache_hits += 1
                    # Déplacer l'élément à la fin (MRU)
                    vector = self.cache.pop(index)
                    self.cache[index] = vector
                    self._update_cache_stats()
                    return vector
                else:
                    # Cache miss - lire depuis le fichier
                    if self.cache_enabled:
                        self.cache_misses += 1
                    
                    # Lire le vecteur depuis mmap
                    offset = self.header_size + index * self.vector_size
                    self.mmap_obj.seek(offset)
                    vector_bytes = self.mmap_obj.read(self.vector_size)
                    vector = np.frombuffer(vector_bytes, dtype=np.float32)
                    
                    # Mettre en cache si activé
                    if self.cache_enabled:
                        # Si le cache est plein, supprimer l'élément le moins récemment utilisé (LRU)
                        if len(self.cache) >= self.cache_capacity and self.cache_capacity > 0:
                            self.cache.popitem(last=False)  # Supprimer le premier élément (LRU)
                        
                        # Ajouter au cache
                        self.cache[index] = vector
                        
                        self._update_cache_stats()
                    
                    return vector
            elif isinstance(index, slice):
                # Cas d'un slice
                start = index.start or 0
                stop = index.stop or self.n
                step = index.step or 1
                
                if step == 1:
                    # Optimisation: lecture en bloc pour les slices consécutifs
                    size = stop - start
                    
                    # Vérifier si tous les éléments sont dans le cache
                    if self.cache_enabled and size <= 1000:  # Pour éviter de trop grandes vérifications
                        cache_indices = [i for i in range(start, stop) if i in self.cache]
                        # Si plus de 80% des indices sont dans le cache, utiliser le cache
                        if len(cache_indices) > 0.8 * size:
                            return np.array([self[i] for i in range(start, stop)])
                    
                    # Lecture directe depuis le fichier
                    offset = self.header_size + start * self.vector_size
                    self.mmap_obj.seek(offset)
                    buffer = self.mmap_obj.read(size * self.vector_size)
                    vectors = np.frombuffer(buffer, dtype=np.float32).reshape(size, self.d)
                    
                    # Mettre en cache les vecteurs fréquemment utilisés
                    if self.cache_enabled and size <= 50:  # Limiter aux petites plages
                        for i, idx in enumerate(range(start, stop)):
                            # Si le cache est plein, supprimer l'élément LRU
                            if len(self.cache) >= self.cache_capacity:
                                self.cache.popitem(last=False)
                            self.cache[idx] = vectors[i]
                    
                    return vectors
                else:
                    # Cas non optimisé pour les slices avec step > 1
                    indices = range(start, stop, step)
                    return np.array([self[i] for i in indices])
            else:
                # Cas d'une liste d'indices
                if len(index) > 200:
                    # Optimisation 1: Vérifier le cache pour tous les indices d'abord
                    if self.cache_enabled:
                        # Séparer les indices présents dans le cache et ceux absents
                        cached_indices = []
                        missing_indices = []
                        
                        for idx in index:
                            if idx in self.cache:
                                cached_indices.append(idx)
                            else:
                                missing_indices.append(idx)
                        
                        # Si plus de 80% sont dans le cache, accéder individuellement
                        if len(cached_indices) > 0.8 * len(index):
                            return np.array([self[i] for i in index])
                    
                    # Optimisation 2: regrouper les indices consécutifs pour les grandes listes
                    sorted_indices = np.sort(index)
                    groups = []
                    current_group = [sorted_indices[0]]
                    
                    # Identifier les groupes d'indices consécutifs
                    for i in range(1, len(sorted_indices)):
                        if sorted_indices[i] == sorted_indices[i-1] + 1:
                            current_group.append(sorted_indices[i])
                        else:
                            groups.append(current_group)
                            current_group = [sorted_indices[i]]
                    groups.append(current_group)
                    
                    # Lire chaque groupe en bloc
                    vectors = []
                    for group in groups:
                        if len(group) == 1:
                            # Un seul indice
                            vectors.append(self[group[0]])
                        else:
                            # Bloc d'indices consécutifs
                            start, end = group[0], group[-1] + 1
                            vectors.append(self[start:end])
                    
                    # Réorganiser les vecteurs selon l'ordre original des indices
                    result = np.vstack([v if len(v.shape) > 1 else v.reshape(1, -1) for v in vectors])
                    index_map = {idx: i for i, idx in enumerate(sorted_indices)}
                    return result[[index_map[idx] for idx in index]]
                else:
                    # Pour les petites listes, utiliser le cache pour chaque vecteur
                    return np.array([self[i] for i in index])
    
    def __len__(self) -> int:
        """Retourne le nombre de vecteurs."""
        return self.n
    
    def dot(self, vectors, query):
        """
        Calcule le produit scalaire entre les vecteurs et une requête.
        Optimisé pour SIMD avec des tailles adaptées (multiples de 16/32/64).

        Args:
            vectors: Les vecteurs (indice ou liste d'indices)
            query: Le vecteur requête

        Returns:
            Un tableau numpy avec les scores de similarité
        """
        # S'assurer que la requête est en float32 et contiguë (optimal pour SIMD)
        query = np.ascontiguousarray(query, dtype=np.float32)

        if self.mode == "ram":
            # En mode RAM, utiliser un dot product vectorisé pour SIMD
            vecs = self[vectors]
            # Assurer contiguïté mémoire pour optimisations SIMD
            if not vecs.flags.c_contiguous:
                vecs = np.ascontiguousarray(vecs, dtype=np.float32)
            return np.dot(vecs, query)
        else:
            # En mode mmap, stratégie optimisée pour différentes tailles
            if isinstance(vectors, (int, np.integer)):
                # Cas d'un seul vecteur
                return np.dot(self[vectors], query)
            else:
                # Adapter la taille des lots pour maximiser SIMD
                # Choisir une taille de lot qui est multiple de 16
                # pour l'alignement mémoire optimal avec AVX-512
                n_vectors = len(vectors)

                if n_vectors <= 256:
                    # Petits lots: lire en une fois pour éviter l'overhead
                    vectors_array = self[vectors]
                    if not vectors_array.flags.c_contiguous:
                        vectors_array = np.ascontiguousarray(vectors_array, dtype=np.float32)
                    return np.dot(vectors_array, query)
                else:
                    # Pour les grandes listes, traiter par lots optimisés pour SIMD
                    # 256 est un bon compromis (registres AVX2/AVX-512, cache L1/L2)
                    batch_size = 256  # Multiple de 16 pour SIMD optimal
                    n_batches = (n_vectors + batch_size - 1) // batch_size

                    # Préallouer avec alignement mémoire optimal
                    results = np.empty(n_vectors, dtype=np.float32)

                    for i in range(n_batches):
                        start_idx = i * batch_size
                        end_idx = min((i + 1) * batch_size, n_vectors)
                        batch_indices = vectors[start_idx:end_idx]

                        # Lecture optimisée des vecteurs
                        batch_vectors = self[batch_indices]
                        # Assurer contiguïté pour SIMD
                        if not batch_vectors.flags.c_contiguous:
                            batch_vectors = np.ascontiguousarray(batch_vectors, dtype=np.float32)

                        # Dot product vectorisé (exploite AVX2/AVX-512)
                        results[start_idx:end_idx] = np.dot(batch_vectors, query)

                    return results
    
    def close(self) -> None:
        """Libère les ressources, y compris le cache."""
        if self.mode == "mmap" and self.mmap_obj is not None:
            # Afficher les statistiques finales du cache
            if self.cache_enabled:
                self._update_cache_stats(force=True)
                print(f"  → Cache final: {len(self.cache):,} vecteurs, {self.cache_hits:,} hits, {self.cache_misses:,} misses")
                
                # Vider le cache
                self.cache.clear()
                self.cache = None
            
            # Fermer mmap
            self.mmap_obj.close()
            self.mmap_file.close()
            self.mmap_obj = None
            self.mmap_file = None

def read_vectors(file_path: str = None, mode: str = "ram", cache_size_mb: Optional[int] = None, vectors: Optional[np.ndarray] = None) -> VectorReader:
    """
    Fonction utilitaire pour lire des vecteurs depuis un fichier ou un tableau numpy.

    Args:
        file_path: Chemin vers le fichier binaire contenant les vecteurs (None si vectors est fourni)
        mode: Mode de lecture - "ram" (défaut) ou "mmap"
        cache_size_mb: Taille du cache en mégaoctets pour le mode mmap (si None, utilise la valeur de ConfigManager)
        vectors: Tableau numpy de vecteurs à utiliser directement (None si file_path est fourni)

    Returns:
        VectorReader: Instance de lecteur de vecteurs
    """
    if vectors is not None:
        # Créer un VectorReader avec les vecteurs fournis
        reader = VectorReader.__new__(VectorReader)
        reader.file_path = None
        reader.mode = "ram"
        reader.n, reader.d = vectors.shape
        reader.header_size = 16
        reader.vector_size = reader.d * 4
        reader.vectors = vectors
        reader.mmap_file = None
        reader.mmap_obj = None
        reader.cache_enabled = False
        reader.cache = None
        reader.cache_hits = 0
        reader.cache_misses = 0
        reader.max_cache_size = 0
        reader.cache_capacity = 0
        return reader
    elif file_path is not None:
        return VectorReader(file_path, mode, cache_size_mb)
    else:
        raise ValueError("Soit file_path soit vectors doit être fourni")

def load_tree(file_path: str, mmap_tree: bool = False) -> K16Tree:
    """
    Charge la structure plate optimisée de l'arbre depuis le fichier précompilé.

    Args:
        file_path: Chemin du fichier binaire de l'arbre ou du fichier plat (.flat.npy)
        mmap_tree: Si True, charge la structure plate en mode mmap pour économiser la RAM

    Returns:
        K16Tree: Arbre K16 chargé avec la structure plate
    """
    # Déterminer le chemin du fichier plat
    if file_path.endswith('.flat') or file_path.endswith('.flat.npy'):
        flat_path = file_path
    else:
        flat_path = os.path.splitext(file_path)[0] + '.flat.npy'

    print(f"⏳ Chargement de la structure plate optimisée depuis {flat_path}...")
    start_time = time.time()

    # Essayer de charger l'arbre
    try:
        if mmap_tree:
            try:
                flat_tree = TreeFlat.load(flat_path, mmap_mode='r')
            except ValueError as e:
                print(f"⚠️ Échec du memory-mapping de l'arbre ({e}), chargement en RAM.")
                flat_tree = TreeFlat.load(flat_path)
        else:
            flat_tree = TreeFlat.load(flat_path)
        print(f"✓ Structure TreeFlat compressée activée")
    except Exception as e:
        print(f"⚠️ Erreur lors du chargement de l'arbre: {e}")
        raise

    tree = K16Tree(None)
    tree.flat_tree = flat_tree
    elapsed = time.time() - start_time
    print(f"✓ Structure plate chargée en {elapsed:.2f}s")
    return tree
```

### k16/utils/cli_get_data.py

```python
"""
Module pour le téléchargement et la préparation des données.
Fournit des fonctions pour télécharger le dataset Natural Questions et générer les embeddings.
"""

import os
import sys
import time
import datetime
import struct
import numpy as np
import argparse
from typing import List, Dict, Any
from tqdm.auto import tqdm

from k16.utils.config import ConfigManager

def format_time(seconds: float) -> str:
    """Formate le temps en heures, minutes, secondes."""
    return str(datetime.timedelta(seconds=int(seconds)))

def write_vectors(vecs: np.ndarray, path: str):
    """
    Écrit les vecteurs dans un binaire : header (n, d) + float32 data.
    
    Args:
        vecs: Vecteurs à écrire
        path: Chemin du fichier de sortie
    """
    n, d = vecs.shape
    start_time = time.time()
    print(f"⏳ Écriture de {n:,} vecteurs (dim {d}) vers {path}...")
    
    # S'assurer que le répertoire existe
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    
    # Convertir en float32 et s'assurer que les vecteurs sont normalisés
    vecs_float32 = vecs.astype(np.float32)
    
    # Format exact attendu par build_candidates
    with open(path, "wb") as f:
        # Utiliser QQ (uint64_t) pour assurer la compatibilité avec endianness explicite
        f.write(struct.pack("<QQ", n, d))
        f.write(vecs_float32.tobytes())
    
    elapsed = time.time() - start_time
    print(f"✓ {n:,} vecteurs (dim {d}) écrits dans {path} [terminé en {elapsed:.2f}s]")

def get_data_command(args: argparse.Namespace) -> int:
    """
    Commande pour télécharger et préparer les données.
    
    Args:
        args: Arguments de ligne de commande
        
    Returns:
        int: Code de retour (0 pour succès, autre pour erreur)
    """
    # Initialisation du gestionnaire de configuration
    config_manager = ConfigManager(args.config)
    
    # Récupération des paramètres pour la préparation des données
    prepare_data_config = config_manager.get_section("prepare_data")
    files_config = config_manager.get_section("files")
    
    # Enregistrer le temps de départ pour calculer la durée totale
    total_start_time = time.time()
    
    try:
        print(f"📥 Téléchargement et préparation des données...")
        print(f"  - Fichier QA: {args.out_text}")
        print(f"  - Fichier vecteurs: {args.out_vec}")
        print(f"  - Modèle: {args.model}")
        print(f"  - Taille de batch: {args.batch_size}")
        
        # 1. Télécharger NQ‑open (train)
        print("⏳ Téléchargement et préparation des données Natural Questions (open)...")
        print("  Cela peut prendre plusieurs minutes, veuillez patienter...")
        download_start = time.time()
        
        # Import dynamique pour éviter des dépendances inutiles
        try:
            from datasets import load_dataset
        except ImportError:
            print("⚠️ Bibliothèque 'datasets' non installée. Installation en cours...")
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets"])
            from datasets import load_dataset
            
        ds = load_dataset("nq_open", split="train")
        download_time = time.time() - download_start
        print(f"✓ Téléchargement terminé en {format_time(download_time)} - {len(ds):,} exemples chargés")
        
        # 2. Création du fichier QA
        print(f"⏳ Création du fichier QA {args.out_text}...")
        qa_start = time.time()
        os.makedirs(os.path.dirname(args.out_text) if os.path.dirname(args.out_text) else ".", exist_ok=True)
        
        lines = []
        with tqdm(total=len(ds), desc="Extraction Q&R") as pbar:
            for ex in ds:
                lines.append(f"{ex['question'].strip()} ||| {ex['answer'][0].strip()}")
                pbar.update(1)
        
        with open(args.out_text, "w", encoding="utf-8") as f_out:
            for i, ln in enumerate(tqdm(lines, desc="Écriture vers fichier")):
                f_out.write(ln.replace("\n", " ") + "\n")
                if (i+1) % 10000 == 0:
                    print(f"  → {i+1:,}/{len(lines):,} lignes écrites ({((i+1)/len(lines))*100:.1f}%)")
        
        qa_time = time.time() - qa_start
        print(f"✓ qa.txt écrit : {len(lines):,} lignes → {args.out_text} [terminé en {format_time(qa_time)}]")
        
        # 3. Embeddings
        # Vérifier si le fichier d'embeddings existe déjà
        recalculate = True
        if os.path.exists(args.out_vec):
            if not args.force:
                # Demander à l'utilisateur s'il veut recalculer les embeddings
                print(f"\nLe fichier d'embeddings {args.out_vec} existe déjà.")
                while True:
                    response = input("Voulez-vous recalculer les embeddings ? (o/n): ").lower()
                    if response in ['o', 'oui', 'y', 'yes']:
                        recalculate = True
                        break
                    elif response in ['n', 'non', 'no']:
                        recalculate = False
                        break
                    else:
                        print("Réponse non reconnue. Veuillez répondre par 'o' (oui) ou 'n' (non).")
            else:
                print(f"⚠️ Remplacement forcé du fichier d'embeddings existant: {args.out_vec}")
        
        encode_time = 0
        if recalculate:
            # Calculer les embeddings
            print(f"⏳ Chargement du modèle {args.model}...")
            encode_start = time.time()
            
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                print("⚠️ Bibliothèque 'sentence-transformers' non installée. Installation en cours...")
                import subprocess
                subprocess.check_call([sys.executable, "-m", "pip", "install", "sentence-transformers"])
                from sentence_transformers import SentenceTransformer
            
            model = SentenceTransformer(args.model)
            print(f"✓ Modèle chargé en {time.time() - encode_start:.2f}s")
            
            print(f"⏳ Encodage avec {args.model}...")
            total_batches = (len(lines) + args.batch_size - 1) // args.batch_size
            print(f"  → Encodage de {len(lines):,} lignes en {total_batches:,} batches de taille {args.batch_size}...")
            
            vecs = []
            encoded_examples = 0
            encode_start_time = time.time()
            
            for i in tqdm(range(0, len(lines), args.batch_size), desc="Batches", total=total_batches):
                batch = [f"passage: {t}" for t in lines[i : i + args.batch_size]]
                batch_size = len(batch)
                encoded_examples += batch_size
                
                batch_start = time.time()
                batch_vecs = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
                batch_time = time.time() - batch_start
                
                current_time = time.time() - encode_start_time
                examples_per_sec = encoded_examples / current_time if current_time > 0 else 0
                remaining = (len(lines) - encoded_examples) / examples_per_sec if examples_per_sec > 0 else 0
                
                vecs.append(batch_vecs)
                
                if (i + args.batch_size) % (args.batch_size * 10) == 0 or (i + batch_size) >= len(lines):
                    print(f"  → Batch {(i // args.batch_size) + 1}/{total_batches}: {batch_size} exemples en {batch_time:.2f}s "
                          f"({batch_size/batch_time:.1f} ex/s)")
                    print(f"  → Progrès: {encoded_examples:,}/{len(lines):,} exemples "
                          f"({encoded_examples/len(lines)*100:.1f}%) - Vitesse: {examples_per_sec:.1f} ex/s")
                    print(f"  → Temps écoulé: {format_time(current_time)} - Temps restant estimé: {format_time(remaining)}")
            
            vecs = np.vstack(vecs)
            encode_time = time.time() - encode_start_time
            print(f"✓ Encodage terminé en {format_time(encode_time)} - {len(lines):,} exemples @ {len(lines)/encode_time:.1f} ex/s")
            
            # Écrire les embeddings
            write_vectors(vecs, args.out_vec)
        else:
            print(f"Utilisation du fichier d'embeddings existant: {args.out_vec}")
        
        total_time = time.time() - total_start_time
        print("\n✓ Traitement terminé.")
        print(f"  - Configuration  : {args.config}")
        print(f"  - QA             : {args.out_text}")
        print(f"  - Embeddings     : {args.out_vec}")
        print(f"  - Modèle         : {args.model}")
        print(f"  - Batch size     : {args.batch_size}")
        print(f"  - Temps total    : {format_time(total_time)}")
        print(f"    ├─ Téléchargement : {format_time(download_time)} ({download_time/total_time*100:.1f}%)")
        print(f"    ├─ Création QA    : {format_time(qa_time)} ({qa_time/total_time*100:.1f}%)")
        if encode_time > 0:
            print(f"    └─ Encodage       : {format_time(encode_time)} ({encode_time/total_time*100:.1f}%)")
        else:
            print(f"    └─ Encodage       : (utilisé fichier existant)")
        
        # Instructions pour les étapes suivantes
        print("\nPour construire l'arbre avec ces vecteurs :")
        print(f"  python -m k16.cli build {args.out_vec} --config {args.config}")
        
    except Exception as e:
        print(f"\n❌ Erreur: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0
```

### k16/utils/optimization.py

```python
"""
Module d'optimisations pour K16.
Configure et gère les optimisations numériques (SIMD, Numba JIT, etc.).
"""

import os
import numpy as np
import platform
from typing import Dict, Any, Optional, List

# Vérifier si Numba est disponible
try:
    import numba
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

def configure_simd():
    """
    Configure l'environnement pour utiliser les optimisations SIMD.
    Force l'utilisation des instructions SIMD dans NumPy.
    """
    # Forcer l'utilisation des instructions SIMD
    os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())
    os.environ["MKL_NUM_THREADS"] = str(os.cpu_count())
    os.environ["OPENBLAS_NUM_THREADS"] = str(os.cpu_count())
    
    # Optimisations d'alignement mémoire pour SIMD
    # Améliore significativement les performances des dot products
    try:
        # Alignement mémoire pour vectorisation optimale
        np.config.add_option_enable_numpy_api(False)

        # Sur certaines plateformes, ces options peuvent être disponibles
        try:
            # Pour Intel MKL
            np.config.add_option_mkl_vml_accuracy("high")
            np.config.add_option_mkl_enable_instructions("AVX2")
            # Si AVX-512 est disponible
            if "avx512" in platform.processor().lower():
                np.config.add_option_mkl_enable_instructions("AVX512")
        except Exception:
            pass
    except Exception:
        pass

def check_simd_support():
    """
    Vérifier les extensions SIMD supportées par la configuration NumPy.
    
    Returns:
        List[str]: Liste des extensions SIMD disponibles.
    """
    simd_extensions = []
    try:
        config_info = np.__config__.show()
        if "SIMD Extensions" in config_info:
            print("✓ Extensions SIMD disponibles pour NumPy:")
            capture = False
            for line in config_info.split("\n"):
                if "SIMD Extensions" in line:
                    capture = True
                elif capture and line.startswith("  "):
                    ext = line.strip()
                    simd_extensions.append(ext)
                    print(f"  - {ext}")
                elif capture and not line.startswith("  "):
                    break
        return simd_extensions
    except Exception:
        print("✓ NumPy configuré pour utiliser les instructions SIMD disponibles")
        return simd_extensions

def check_numba_support():
    """
    Vérifie si Numba est disponible et configuré correctement.
    
    Returns:
        bool: True si Numba est disponible et configuré correctement.
    """
    if NUMBA_AVAILABLE:
        print("✓ Numba JIT est disponible pour l'optimisation")
        return True
    else:
        print("⚠️ Numba JIT n'est pas disponible. Les performances seront réduites.")
        return False

def optimize_functions():
    """
    Configure toutes les optimisations numériques pour K16.
    
    Returns:
        Dict[str, Any]: Un dictionnaire contenant les informations sur les optimisations.
    """
    # Configuration SIMD
    configure_simd()
    
    # Vérifier le support
    simd_extensions = check_simd_support()
    numba_available = check_numba_support()
    
    return {
        "simd": {
            "available": len(simd_extensions) > 0,
            "extensions": simd_extensions
        },
        "numba": {
            "available": numba_available
        }
    }
```

### k16/utils/config.py

```python
"""
Module de gestion de la configuration pour K16.
Centralise le chargement et l'accès à la configuration YAML.
"""

import os
import yaml

# Chemin par défaut du fichier de configuration
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "config.yaml")

# Configuration par défaut
DEFAULT_CONFIG = {
    "general": {
        "debug": False
    },
    "build_tree": {
        "max_depth": 6,
        "k": 16,
        "k_adaptive": False,  # Valeur fixe k=16 par défaut
        "k_min": 2,
        "k_max": 32,
        "max_leaf_size": 100,
        "max_data": 256,  # Multiple de 16 pour optimisation SIMD
        "max_workers": 8,
        "use_gpu": True,
        "prune_unused": False,  # Paramètre obsolète
        # Le pruning des feuilles inutilisées est maintenant automatique et ne peut plus être désactivé
        "hnsw_batch_size": 1000,
        "grouping_batch_size": 5000,
        "hnsw_m": 16,
        "hnsw_ef_construction": 200,
    },
    "flat_tree": {
        "max_dims": 512,  # Multiple de 16 pour optimisation SIMD
        "reduction_method": "variance"  # ou "directional"
    },
    "search": {
        "k": 100,
        "queries": 100,
        "mode": "ram",
        "cache_size_mb": 500,
        "use_faiss": True
    },
    "prepare_data": {
        "model": "intfloat/multilingual-e5-large",
        "batch_size": 128,
        "normalize": True
    },
    "files": {
        "vectors_dir": ".",
        "trees_dir": ".",
        "default_qa": "qa.txt",
        "default_vectors": "vectors.bin",
        "default_tree": "tree.bin"
    }
}

class ConfigManager:
    """Gestionnaire de configuration pour K16."""
    
    def __init__(self, config_path=None):
        """
        Initialise le gestionnaire de configuration.
        
        Paramètres :
            config_path: Chemin vers le fichier de configuration YAML.
                         Si None, utilise le chemin par défaut.
        """
        self.config_path = config_path or DEFAULT_CONFIG_PATH
        self.config = self.load_config()
        
    def load_config(self):
        """
        Charge la configuration depuis le fichier YAML.
        
        Retourne :
            Dict: La configuration chargée, ou la configuration par défaut en cas d'erreur.
        """
        try:
            with open(self.config_path, "r") as f:
                config = yaml.safe_load(f)
            
            # Vérifier et compléter la configuration
            self._ensure_complete_config(config)
            
            return config
        except Exception as e:
            print(f"⚠️ Erreur lors du chargement de la configuration: {str(e)}")
            print(f"⚠️ Utilisation des paramètres par défaut")
            return DEFAULT_CONFIG.copy()
    
    def _ensure_complete_config(self, config):
        """
        S'assure que la configuration contient toutes les sections nécessaires.
        Complète avec les valeurs par défaut si nécessaire.
        
        Paramètres :
            config: Configuration à vérifier et compléter.
        """
        for section, default_values in DEFAULT_CONFIG.items():
            if section not in config:
                config[section] = default_values.copy()
            else:
                for key, value in default_values.items():
                    if key not in config[section]:
                        config[section][key] = value
    
    def get_section(self, section):
        """
        Récupère une section complète de la configuration.
        
        Paramètres :
            section: Nom de la section à récupérer.

        Retourne :
            Dict: La section demandée, ou un dictionnaire vide si la section n'existe pas.
        """
        return self.config.get(section, {})
    
    def get(self, section, key, default=None):
        """
        Récupère une valeur spécifique de la configuration.
        
        Paramètres :
            section: La section contenant la clé.
            key: La clé à récupérer.
            default: Valeur par défaut si la clé n'existe pas.

        Retourne :
            La valeur associée à la clé, ou la valeur par défaut si la clé n'existe pas.
        """
        section_data = self.get_section(section)
        return section_data.get(key, default)
    
    def get_file_path(self, file_key, default=None):
        """
        Construit le chemin complet vers un fichier spécifié dans la configuration.
        
        Paramètres :
            file_key: Clé du fichier dans la section 'files'.
            default: Valeur par défaut si la clé n'existe pas.

        Retourne :
            Le chemin complet vers le fichier.
        """
        files_section = self.get_section("files")
        
        if file_key.startswith("default_"):
            # Pour les fichiers par défaut, construire le chemin complet
            file_name = files_section.get(file_key, default)
            
            # Déterminer le répertoire approprié
            if "vectors" in file_key:
                dir_key = "vectors_dir"
            elif "tree" in file_key:
                dir_key = "trees_dir"
            else:
                dir_key = "vectors_dir"  # Par défaut
            
            dir_path = files_section.get(dir_key, ".")
            return os.path.join(dir_path, file_name)
        else:
            # Pour les autres clés, retourner directement la valeur
            return files_section.get(file_key, default)
    
    def reload(self, config_path=None):
        """
        Recharge la configuration depuis un nouveau fichier.
        
        Paramètres :
            config_path: Nouveau chemin de configuration. Si None, utilise le chemin actuel.
        """
        if config_path:
            self.config_path = config_path
        self.config = self.load_config()
        
    def __str__(self):
        """Représentation de la configuration pour le débogage."""
        return f"Configuration chargée depuis: {self.config_path}"

# Fonction utilitaire pour charger une configuration
def load_config(config_path=None):
    """
    Fonction utilitaire pour charger rapidement une configuration.
    
    Paramètres :
        config_path: Chemin vers le fichier de configuration YAML.
                     Si None, utilise le chemin par défaut.
    
    Retourne :
        ConfigManager: Instance du gestionnaire de configuration.
    """
    return ConfigManager(config_path)
```

### k16/utils/cli_search.py

```python
"""
Module pour la recherche interactive en ligne de commande.
Fournit une interface pour rechercher des questions similaires dans un terminal.
"""

import os
import time
import numpy as np
from typing import List, Dict, Any, Tuple
import argparse

from k16.utils.config import ConfigManager
from k16.io.reader import read_vectors, load_tree
from k16.search.searcher import search, Searcher

def format_results(results: List[Dict], timings: Dict[str, float]) -> str:
    """
    Formate les résultats de recherche pour l'affichage en terminal.
    
    Args:
        results: Liste des résultats de recherche
        timings: Dictionnaire des temps d'exécution
        
    Returns:
        str: Résultats formatés
    """
    output = []
    
    # Afficher les métriques
    output.append("\n🕒 Temps:")
    output.append(f"  → Encodage      : {timings['encode']*1000:.2f} ms")
    output.append(f"  → Recherche arbre: {timings['tree_search']*1000:.2f} ms")
    output.append(f"  → Filtrage      : {timings['filter']*1000:.2f} ms")
    search_only = timings['tree_search'] + timings['filter']
    total_time = timings['encode'] + search_only
    output.append(f"  → Recherche totale: {search_only*1000:.2f} ms")
    output.append(f"  → Temps total    : {total_time*1000:.2f} ms")
    
    # Afficher les résultats
    output.append("\n📋 Résultats:")
    for i, result in enumerate(results, 1):
        output.append(f"\n{i}. {result['question']}")
        output.append(f"   → {result['answer']}")
    
    return "\n".join(output)

def search_once(model, searcher, qa_lines: List[str], query: str, k: int = 10) -> Tuple[List[Dict], Dict[str, float]]:
    """
    Effectue une seule recherche.

    Args:
        model: Modèle d'embeddings
        searcher: Chercheur K16
        qa_lines: Lignes de questions-réponses
        query: Question à rechercher
        k: Nombre de résultats à retourner

    Returns:
        Tuple[List[Dict], Dict[str, float]]: Résultats et timings
    """
    # Encoder la requête
    encode_start = time.time()
    query_vector = model.encode(f"query: {query}", normalize_embeddings=True)
    encode_time = time.time() - encode_start

    # Recherche avec l'arbre
    tree_search_start = time.time()
    tree_candidates = searcher.search_tree(query_vector)
    tree_search_time = time.time() - tree_search_start

    # Filtrer pour obtenir les k meilleurs
    filter_start = time.time()
    indices = searcher.filter_candidates(tree_candidates, query_vector, k)
    filter_time = time.time() - filter_start

    # Récupérer les résultats
    results = []
    for idx in indices:
        if idx < len(qa_lines):
            parts = qa_lines[idx].strip().split(" ||| ")
            if len(parts) == 2:
                question, answer = parts
                results.append({
                    "question": question,
                    "answer": answer,
                    "index": idx
                })

    timings = {
        "encode": encode_time,
        "tree_search": tree_search_time,
        "filter": filter_time
    }

    return results, timings

def search_interactive(model, searcher, qa_lines: List[str], k: int = 10) -> Tuple[List[Dict], Dict[str, float]]:
    """
    Effectue une recherche interactive.

    Args:
        model: Modèle d'embeddings
        searcher: Chercheur K16
        qa_lines: Lignes de questions-réponses
        k: Nombre de résultats à retourner

    Returns:
        Tuple[List[Dict], Dict[str, float]]: Résultats et timings
    """
    try:
        query = input("\nVotre question (q pour quitter): ")

        if query.lower() in ['q', 'quit', 'exit']:
            return None, None

        return search_once(model, searcher, qa_lines, query, k)

    except EOFError:
        print("\nMode non-interactif détecté. Voici un exemple de recherche:")
        example_query = "Qui a inventé la théorie de la relativité?"
        print(f"Question: {example_query}")

        results, timings = search_once(model, searcher, qa_lines, example_query, k)
        return results, timings

def search_command(args: argparse.Namespace) -> int:
    """
    Commande pour effectuer une recherche interactive dans un terminal.

    Args:
        args: Arguments de ligne de commande

    Returns:
        int: Code de retour (0 pour succès, autre pour erreur)
    """
    # Initialisation du gestionnaire de configuration
    config_manager = ConfigManager(args.config)

    # Récupération des paramètres
    search_config = config_manager.get_section("search")
    files_config = config_manager.get_section("files")
    build_config = config_manager.get_section("build_tree")
    prepare_config = config_manager.get_section("prepare_data")

    try:
        print(f"🔍 Mode recherche K16 interactive...")
        print(f"  - Vecteurs: {args.vectors_file}")
        print(f"  - Arbre: {args.tree_file}")
        print(f"  - QA: {args.qa_file}")
        print(f"  - Mode: {args.mode}")
        print(f"  - K (nombre de résultats): {args.k}")
        print(f"  - Type de recherche: {args.search_type}")
        print(f"  - Largeur de faisceau: {args.beam_width}")

        # Vérifier que les fichiers existent
        if not os.path.exists(args.vectors_file):
            print(f"❌ Fichier de vecteurs introuvable: {args.vectors_file}")
            print(f"   Utilisez la commande 'getData' pour télécharger et préparer les données:")
            print(f"   python -m k16.cli getData --config {args.config}")
            return 1

        if not os.path.exists(args.tree_file) and not os.path.exists(args.tree_file.replace(".bsp", ".flat.npy")):
            print(f"❌ Fichier d'arbre introuvable: {args.tree_file}")
            print(f"   Utilisez la commande 'build' pour construire l'arbre:")
            print(f"   python -m k16.cli build {args.vectors_file} --config {args.config}")
            return 1

        if not os.path.exists(args.qa_file):
            print(f"❌ Fichier QA introuvable: {args.qa_file}")
            print(f"   Utilisez la commande 'getData' pour télécharger et préparer les données:")
            print(f"   python -m k16.cli getData --config {args.config}")
            return 1

        # Charger les vecteurs
        print(f"⏳ Chargement des vecteurs depuis {args.vectors_file}...")
        vectors_reader = read_vectors(
            file_path=args.vectors_file,
            mode=args.mode,
            cache_size_mb=args.cache_size
        )
        print(f"✓ Vecteurs chargés: {len(vectors_reader):,} vecteurs de dimension {vectors_reader.d}")

        # Charger l'arbre
        print(f"⏳ Chargement de l'arbre depuis {args.tree_file}...")
        tree = load_tree(args.tree_file, mmap_tree=(args.mode == "mmap"))
        print(f"✓ Arbre chargé")

        # Charger les questions et réponses
        print(f"⏳ Chargement des questions et réponses depuis {args.qa_file}...")
        with open(args.qa_file, "r", encoding="utf-8") as f:
            qa_lines = f.readlines()
        print(f"✓ {len(qa_lines):,} questions-réponses chargées")

        # Créer le chercheur
        from k16.search.searcher import Searcher
        searcher = Searcher(
            k16tree=tree,
            vectors_reader=vectors_reader,
            use_faiss=args.use_faiss,
            search_type=args.search_type,
            beam_width=args.beam_width
        )

        # Charger le modèle d'embeddings
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            print("⚠️ Bibliothèque 'sentence-transformers' non installée. Installation en cours...")
            import subprocess, sys
            subprocess.check_call([sys.executable, "-m", "pip", "install", "sentence-transformers"])
            from sentence_transformers import SentenceTransformer

        print(f"⏳ Chargement du modèle {args.model}...")
        model = SentenceTransformer(args.model)
        print(f"✓ Modèle chargé")

        # Statistiques de l'arbre
        stats = tree.flat_tree.get_statistics()
        compression_stats = stats.get('compression', {})

        print("\n📊 Statistiques de l'arbre:")
        print(f"  → Nœuds: {stats.get('n_nodes', '?'):,}")
        print(f"  → Feuilles: {stats.get('n_leaves', '?'):,}")
        print(f"  → Profondeur: {stats.get('max_depth', '?')}")

        # Lancement de l'interface interactive
        print(f"\n💬 Interface de recherche interactive K16")
        print(f"  → Tapez votre question et appuyez sur Entrée")
        print(f"  → Tapez 'q' pour quitter")

        # En mode interactif, continuer jusqu'à ce que l'utilisateur quitte
        try:
            while True:
                results, timings = search_interactive(model, searcher, qa_lines, k=args.k)

                if results is None:
                    print("\n👋 Au revoir!")
                    break

                # Afficher les résultats
                print(format_results(results, timings))
        except KeyboardInterrupt:
            print("\n👋 Recherche interrompue. Au revoir!")

    except Exception as e:
        print(f"\n❌ Erreur: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

    return 0
```

### k16/utils/cli_api.py

```python
"""
Module pour exposer les fonctionnalités de K16 via une API REST.
Utilise FastAPI pour exposer un endpoint de recherche.
"""

import os
import time
import argparse
import numpy as np
from typing import Dict, Any, List, Optional

from k16.utils.config import ConfigManager
from k16.io.reader import read_vectors, load_tree
from k16.search.searcher import Searcher

# Variables globales pour stocker les ressources
model = None
searcher = None
qa_lines = None
vectors_reader = None
tree = None

def load_resources(config_manager):
    """Charge les ressources nécessaires (modèle, vecteurs, arbre, qa) à partir de la configuration."""
    global model, searcher, qa_lines, vectors_reader, tree
    
    # Récupération des paramètres
    search_config = config_manager.get_section("search")
    files_config = config_manager.get_section("files")
    prepare_config = config_manager.get_section("prepare_data")
    
    # Chemins des fichiers
    vectors_path = os.path.join(files_config["vectors_dir"], files_config["default_vectors"])
    tree_path = os.path.join(files_config["trees_dir"], files_config["default_tree"])
    qa_path = os.path.join(files_config["vectors_dir"], files_config.get("default_qa", "qa.txt"))
    
    # Vérifier que les fichiers existent
    if not os.path.exists(vectors_path):
        print(f"❌ Fichier de vecteurs introuvable: {vectors_path}")
        return False
        
    if not os.path.exists(tree_path) and not os.path.exists(tree_path.replace(".bsp", ".flat.npy")):
        print(f"❌ Fichier d'arbre introuvable: {tree_path}")
        return False
        
    if not os.path.exists(qa_path):
        print(f"❌ Fichier QA introuvable: {qa_path}")
        return False
    
    # Charger les vecteurs
    print(f"⏳ Chargement des vecteurs depuis {vectors_path}...")
    vectors_reader = read_vectors(
        file_path=vectors_path,
        mode=search_config["mode"],
        cache_size_mb=search_config["cache_size_mb"]
    )
    print(f"✓ Vecteurs chargés: {len(vectors_reader):,} vecteurs de dimension {vectors_reader.d}")
    
    # Charger l'arbre
    print(f"⏳ Chargement de l'arbre depuis {tree_path}...")
    tree = load_tree(tree_path, mmap_tree=(search_config["mode"] == "mmap"))
    print(f"✓ Arbre chargé")
    
    # Charger les questions et réponses
    print(f"⏳ Chargement des questions et réponses depuis {qa_path}...")
    with open(qa_path, "r", encoding="utf-8") as f:
        qa_lines = f.readlines()
    print(f"✓ {len(qa_lines):,} questions-réponses chargées")
    
    # Créer le chercheur
    searcher = Searcher(
        k16tree=tree,
        vectors_reader=vectors_reader,
        use_faiss=search_config["use_faiss"],
        search_type=search_config["search_type"],
        beam_width=search_config.get("beam_width", 3)
    )
    
    # Charger le modèle d'embeddings
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("⚠️ Bibliothèque 'sentence-transformers' non installée. Installation en cours...")
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "sentence-transformers"])
        from sentence_transformers import SentenceTransformer
    
    model_name = prepare_config.get("model", "intfloat/multilingual-e5-large")
    print(f"⏳ Chargement du modèle {model_name}...")
    model = SentenceTransformer(model_name)
    print(f"✓ Modèle chargé")
    
    # Statistiques de l'arbre
    stats = tree.flat_tree.get_statistics()
    print("\n📊 Statistiques de l'arbre:")
    print(f"  → Nœuds: {stats.get('n_nodes', '?'):,}")
    print(f"  → Feuilles: {stats.get('n_leaves', '?'):,}")
    print(f"  → Profondeur: {stats.get('max_depth', '?')}")
    
    return True

def setup_app():
    """Configure et retourne l'application FastAPI."""
    # Importer les dépendances nécessaires
    try:
        from fastapi import FastAPI, HTTPException, Query
        from fastapi.middleware.cors import CORSMiddleware
        from pydantic import BaseModel
    except ImportError:
        print("⚠️ Les bibliothèques 'fastapi' et 'uvicorn' sont nécessaires pour l'API. Installation en cours...")
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "fastapi", "uvicorn[standard]"])
        from fastapi import FastAPI, HTTPException, Query
        from fastapi.middleware.cors import CORSMiddleware
        from pydantic import BaseModel

    # Modèles Pydantic pour la validation des données
    class SearchQuery(BaseModel):
        query: str
        k: Optional[int] = 10

    class SearchResult(BaseModel):
        question: str
        answer: str
        similarity: float
        index: int

    class SearchResponse(BaseModel):
        results: List[SearchResult]
        timings: Dict[str, float]
        stats: Dict[str, Any]
    
    # Application FastAPI
    app = FastAPI(
        title="K16 Search API",
        description="API for fast vector search using K16 hierarchical tree",
        version="1.0.0",
    )

    # Configuration CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Ajuster en production!
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/")
    async def root():
        """Endpoint racine avec informations de base."""
        return {
            "name": "K16 Search API",
            "version": "1.0.0",
            "description": "API for fast vector search using K16 hierarchical tree",
            "endpoints": {
                "/search": "Search for similar questions",
                "/stats": "Get statistics about the loaded tree",
                "/health": "Check API health status"
            }
        }

    @app.get("/health")
    async def health():
        """Vérification de l'état de santé de l'API."""
        if not model or not searcher or not qa_lines:
            return {
                "status": "error",
                "message": "Resources not fully loaded"
            }
        return {
            "status": "ok",
            "loaded_resources": {
                "model": model.__class__.__name__,
                "vectors": len(vectors_reader) if vectors_reader else 0,
                "qa_lines": len(qa_lines) if qa_lines else 0,
                "tree_loaded": searcher is not None
            }
        }

    @app.get("/stats")
    async def get_stats():
        """Récupère les statistiques de l'arbre et des vecteurs."""
        if not searcher or not tree:
            raise HTTPException(status_code=503, detail="Tree not loaded")
        
        stats = tree.flat_tree.get_statistics()
        
        return {
            "tree": {
                "nodes": stats.get("n_nodes", 0),
                "leaves": stats.get("n_leaves", 0),
                "max_depth": stats.get("max_depth", 0),
                "compression": stats.get("compression", {})
            },
            "vectors": {
                "count": len(vectors_reader) if vectors_reader else 0,
                "dimensions": vectors_reader.d if vectors_reader else 0
            },
            "qa": {
                "count": len(qa_lines) if qa_lines else 0
            },
            "search": {
                "type": searcher.search_type,
                "beam_width": searcher.beam_width if searcher.search_type == "beam" else None
            }
        }

    @app.post("/search", response_model=SearchResponse)
    async def search(query_data: SearchQuery):
        """
        Recherche des questions similaires à la requête.
        
        Args:
            query_data: Contient la requête et le nombre de résultats souhaités
            
        Returns:
            Les résultats de la recherche et les informations de timing
        """
        if not model or not searcher or not qa_lines:
            raise HTTPException(status_code=503, detail="Resources not fully loaded")
        
        # Encoder la requête
        encode_start = time.time()
        query_vector = model.encode(f"query: {query_data.query}", normalize_embeddings=True)
        encode_time = time.time() - encode_start
        
        # Recherche avec l'arbre
        tree_search_start = time.time()
        tree_candidates = searcher.search_tree(query_vector)
        tree_search_time = time.time() - tree_search_start
        
        # Filtrer pour obtenir les k meilleurs
        filter_start = time.time()
        indices_with_scores = []
        
        # Utiliser VectorReader optimisé
        top_indices = searcher.filter_candidates(tree_candidates, query_vector, query_data.k)
        top_vectors = vectors_reader[top_indices]
        scores = [np.dot(query_vector, top_vectors[i]) for i in range(len(top_indices))]
        
        # Trier par score décroissant
        sorted_pairs = sorted(zip(top_indices, scores), key=lambda x: x[1], reverse=True)
        indices_with_scores = [(idx, score) for idx, score in sorted_pairs]
        
        filter_time = time.time() - filter_start
        
        # Récupérer les résultats
        results = []
        for idx, score in indices_with_scores:
            if idx < len(qa_lines):
                parts = qa_lines[idx].strip().split(" ||| ")
                if len(parts) == 2:
                    question, answer = parts
                    results.append(SearchResult(
                        question=question,
                        answer=answer,
                        similarity=float(score),
                        index=idx
                    ))
        
        # Calculer les statistiques et les temps
        timings = {
            "encode_ms": encode_time * 1000,
            "tree_search_ms": tree_search_time * 1000,
            "filter_ms": filter_time * 1000,
            "total_ms": (encode_time + tree_search_time + filter_time) * 1000
        }
        
        stats = {
            "candidates_count": len(tree_candidates)
        }
        
        return SearchResponse(
            results=results,
            timings=timings,
            stats=stats
        )
    
    return app

def api_command(args: argparse.Namespace) -> int:
    """
    Commande pour lancer l'API FastAPI.
    
    Args:
        args: Arguments de ligne de commande
        
    Returns:
        int: Code de retour (0 pour succès, autre pour erreur)
    """
    # Initialisation du gestionnaire de configuration
    config_manager = ConfigManager(args.config)
    
    try:
        # Vérifier que FastAPI est installé
        try:
            import uvicorn
        except ImportError:
            print("⚠️ Les bibliothèques 'fastapi' et 'uvicorn' sont nécessaires pour l'API. Installation en cours...")
            import subprocess, sys
            subprocess.check_call([sys.executable, "-m", "pip", "install", "fastapi", "uvicorn[standard]"])
            import uvicorn
            
        # Récupérer les paramètres API depuis le fichier config
        api_config = config_manager.get_section("api")

        # Définir les valeurs par défaut si la section API n'existe pas ou est incomplète
        default_host = "127.0.0.1"
        default_port = 8000
        default_reload = False
        
        host = args.host if args.host else api_config.get("host", default_host)
        port = args.port if args.port else api_config.get("port", default_port)
        reload = args.reload if args.reload is not None else api_config.get("reload", default_reload)
        
        print(f"🌐 Démarrage de l'API K16...")
        print(f"  - Configuration: {args.config}")
        print(f"  - Adresse: {host}:{port}")
        print(f"  - Rechargement auto: {'Activé' if reload else 'Désactivé'}")
        
        # Charger les ressources
        if not load_resources(config_manager):
            return 1
        
        # Configurer l'application et remplacer l'app globale
        global app
        app = setup_app()

        # Démarrer l'API
        print(f"\n🚀 Démarrage du serveur API sur http://{host}:{port}")
        print(f"  → Documentation Swagger: http://{host}:{port}/docs")
        print(f"  → Documentation ReDoc: http://{host}:{port}/redoc")
        print(f"  → Pour arrêter le serveur: CTRL+C\n")

        # Démarrer Uvicorn
        if reload:
            # En mode reload, on doit utiliser un chemin d'importation
            import inspect
            module_path = inspect.getmodule(api_command).__name__
            uvicorn.run(
                f"{module_path}:app",
                host=host,
                port=port,
                reload=reload,
                log_level="info"
            )
        else:
            # En mode normal, on peut passer l'app directement
            uvicorn.run(
                app,
                host=host,
                port=port,
                log_level="info"
            )
        
    except Exception as e:
        print(f"\n❌ Erreur: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

# Variable globale pour l'application qui sera configurée lors de l'exécution
# Ne pas initialiser ici pour éviter les erreurs d'importation si FastAPI n'est pas installé
app = None
```

### k16/utils/cli_build.py

```python
"""
Module pour la construction d'arbres K16.
Fournit des fonctions pour construire des arbres optimisés.
"""

import os
import time
import datetime
import argparse
from typing import Any

from k16.utils.config import ConfigManager
from k16.builder.builder import build_optimized_tree

def format_time(seconds: float) -> str:
    """Formate le temps en heures, minutes, secondes."""
    return str(datetime.timedelta(seconds=int(seconds)))

def build_command(args: argparse.Namespace) -> int:
    """
    Commande pour construire un arbre optimisé.
    
    Args:
        args: Arguments de ligne de commande
        
    Returns:
        int: Code de retour (0 pour succès, autre pour erreur)
    """
    # Initialisation du gestionnaire de configuration
    config_manager = ConfigManager(args.config)

    # Récupération des paramètres pour la construction de l'arbre
    build_config = config_manager.get_section("build_tree")
    files_config = config_manager.get_section("files")
    flat_tree_config = config_manager.get_section("flat_tree")

    # Enregistrer le temps de départ pour calculer la durée totale
    total_start_time = time.time()

    try:
        print(f"🚀 Construction d'un arbre K16 optimisé...")
        print(f"  - Vecteurs: {args.vectors_file}")
        print(f"  - Sortie: {args.tree_file}")
        print(f"  - Profondeur max: {args.max_depth}")
        print(f"  - Taille max feuille: {args.max_leaf_size}")
        print(f"  - Max data: {args.max_data}")
        print(f"  - Dimensions réduites: {args.max_dims}")
        print(f"  - HNSW: {'Activé' if args.hnsw else 'Désactivé'}")
        print(f"  - K adaptatif: {'Activé' if args.k_adaptive else 'Désactivé'}")

        # Construction de l'arbre optimisé en une seule fonction
        flat_tree = build_optimized_tree(
            vectors=args.vectors_file,
            output_file=args.tree_file,
            max_depth=args.max_depth,
            max_leaf_size=args.max_leaf_size,
            max_data=args.max_data,
            max_dims=args.max_dims,
            use_hnsw=args.hnsw,
            k=args.k,
            k_adaptive=args.k_adaptive,
            verbose=True
        )

        total_time = time.time() - total_start_time
        print(f"\n✓ Construction de l'arbre optimisé terminée en {format_time(total_time)}")

        # Instructions pour l'utilisation du script de test
        print("\nPour tester la recherche dans cet arbre :")
        print(f"  python -m k16.cli test {args.vectors_file} {args.tree_file} --k 100")
        print(f"  ou, en utilisant la configuration :")
        print(f"  python -m k16.cli test --config {args.config}")
        print(f"\nPour faire des recherches interactives :")
        print(f"  python -m k16.cli search --config {args.config}")

    except Exception as e:
        print(f"\n❌ Erreur: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

    return 0
```

### k16/builder/clustering.py

```python
"""
Module de clustering pour K16.
Implémente les algorithmes de clustering utilisés pour construire l'arbre K16.
Optimisé pour les instructions SIMD AVX2/AVX-512.
"""

import numpy as np
import time
import platform
import os
from typing import Tuple, List, Dict, Any, Optional, Union
import multiprocessing
from joblib import Parallel, delayed
from tqdm.auto import tqdm

import faiss

try:
    from kneed import KneeLocator
    KNEEDLE_AVAILABLE = True
except ImportError:
    KNEEDLE_AVAILABLE = False
    print("⚠️ Module kneed n'est pas installé. La méthode du coude utilisera une approche simplifiée.")

from k16.core.tree import TreeNode
from k16.utils.optimization import configure_simd

# Détection des capacités SIMD
def optimize_for_simd():
    """Configure l'environnement pour maximiser l'utilisation des instructions SIMD."""
    # Appliquer les optimisations de base
    configure_simd()
    
    # Détection des capacités SIMD
    cpu_info = platform.processor().lower()
    has_avx2 = "avx2" in cpu_info
    has_avx512 = "avx512" in cpu_info

    # Afficher les informations sur les optimisations SIMD détectées
    if has_avx512:
        print("✓ Optimisation pour AVX-512 (512 bits)")
    elif has_avx2:
        print("✓ Optimisation pour AVX2 (256 bits)")
    else:
        print("✓ Optimisation pour SSE (128 bits)")

    # Vérifier si on a un nombre de clusters optimal pour SIMD
    if platform.machine() in ('x86_64', 'AMD64', 'x86'):
        print("✓ Architecture x86_64 détectée - k=16 est optimal pour SIMD")

    return has_avx2, has_avx512

# Appliquer les optimisations SIMD au démarrage
HAS_AVX2, HAS_AVX512 = optimize_for_simd()

def kmeans_faiss(vectors: np.ndarray, k: int, gpu: bool = False, niter: int = 5, remove_empty: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    K-means sphérique optimisé pour les embeddings normalisés.
    Utilise la similarité cosinus (produit scalaire) au lieu de la distance euclidienne.

    Args:
        vectors: Les vecteurs à clusteriser (shape: [n, d])
        k: Nombre de clusters
        gpu: Utiliser le GPU si disponible
        niter: Nombre d'itérations
        remove_empty: Si True, supprime les clusters vides et ajuste k en conséquence

    Returns:
        Tuple[np.ndarray, np.ndarray]: (centroids, labels)
    """
    # S'assurer que les vecteurs sont en float32
    vectors = vectors.astype(np.float32)

    # Vérifier si les vecteurs sont normalisés
    norms = np.linalg.norm(vectors, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        # Normaliser si nécessaire
        vectors = vectors / norms[:, np.newaxis]

    # Dimension des vecteurs
    d = vectors.shape[1]
    n = vectors.shape[0]

    # Ne pas créer plus de clusters que de points
    if k > n:
        k = n

    # Utiliser l'implémentation K-means optimisée de FAISS
    try:
        # Créer l'objet K-means FAISS
        kmeans = faiss.Kmeans(d, k, niter=niter, verbose=False, spherical=True, gpu=gpu)

        # Entraîner le modèle
        kmeans.train(vectors)

        # Obtenir les centroïdes
        centroids = kmeans.centroids.copy()

        # Assigner les points aux clusters
        index = faiss.IndexFlatIP(d)
        if gpu:
            try:
                res = faiss.StandardGpuResources()
                index = faiss.GpuIndexFlatIP(res, d)
            except:
                pass  # Fallback au CPU

        index.add(centroids)
        _, labels = index.search(vectors, 1)
        labels = labels.reshape(-1)

        # Gérer les clusters vides si demandé
        if remove_empty:
            cluster_sizes = np.bincount(labels, minlength=k)
            empty_clusters = np.where(cluster_sizes == 0)[0]

            if len(empty_clusters) > 0 and len(empty_clusters) < k - 1:
                print(f"  ⚠️ Suppression de {len(empty_clusters)} clusters vides...")

                # Garder seulement les centroïdes des clusters non vides
                valid_indices = np.where(cluster_sizes > 0)[0]
                new_k = len(valid_indices)

                if new_k >= 2:
                    # Filtrer les centroïdes
                    final_centroids = centroids[valid_indices]

                    # Ré-étiqueter les points
                    index.reset()
                    index.add(final_centroids)
                    _, final_labels = index.search(vectors, 1)
                    final_labels = final_labels.reshape(-1)

                    print(f"  ✓ K réduit de {k} à {new_k} pour éviter les clusters vides")
                    return final_centroids, final_labels

        return centroids, labels

    except Exception as e:
        # Fallback vers sklearn si FAISS échoue
        print(f"⚠️ FAISS K-means failed ({e}), using sklearn fallback")
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import normalize

        # Normaliser les vecteurs
        vectors_norm = normalize(vectors, norm='l2')

        # K-means avec sklearn
        kmeans = KMeans(n_clusters=k, max_iter=niter, n_init=1, random_state=42)
        labels = kmeans.fit_predict(vectors_norm)

        # Vérifier les clusters vides
        if remove_empty:
            cluster_sizes = np.bincount(labels, minlength=k)
            empty_clusters = np.where(cluster_sizes == 0)[0]

            if len(empty_clusters) > 0 and len(empty_clusters) < k-1:
                print(f"  ⚠️ Suppression de {len(empty_clusters)} clusters vides (version CPU)...")

                # Garder seulement les centroïdes des clusters non vides
                valid_indices = np.where(cluster_sizes > 0)[0]
                new_k = len(valid_indices)

                # Relabel les points
                new_labels = np.zeros_like(labels)
                for new_idx, old_idx in enumerate(valid_indices):
                    new_labels[labels == old_idx] = new_idx

                # Extraire les centroïdes valides
                valid_centroids = kmeans.cluster_centers_[valid_indices]

                # Normaliser les centroïdes
                centroids = normalize(valid_centroids, norm='l2')
                labels = new_labels

                print(f"  ✓ K réduit de {k} à {new_k} pour éviter les clusters vides")
                return centroids, labels

        # Normaliser les centroïdes
        centroids = normalize(kmeans.cluster_centers_, norm='l2')

    return centroids, labels


def spherical_kmeans_plusplus(vectors: np.ndarray, k: int) -> np.ndarray:
    """
    Initialisation K-means++ adaptée pour le clustering sphérique.
    Sélectionne intelligemment les centroïdes initiaux.

    Args:
        vectors: Vecteurs normalisés
        k: Nombre de clusters

    Returns:
        np.ndarray: Centroïdes initiaux
    """
    n, d = vectors.shape

    # Si k = n, simplement retourner tous les vecteurs comme centroïdes
    if k >= n:
        # Copier les vecteurs pour éviter les modifications accidentelles
        return vectors.copy()

    centroids = np.zeros((k, d), dtype=np.float32)

    # Choisir le premier centroïde aléatoirement
    centroids[0] = vectors[np.random.randint(0, n)]

    # Utiliser FAISS pour accélérer
    index = faiss.IndexFlatIP(d)

    for i in range(1, k):
        # Calculer les distances au centroïde le plus proche
        index.reset()
        index.add(centroids[:i])
        similarities, _ = index.search(vectors, 1)

        # Convertir similarités en distances angulaires
        similarities = np.clip(similarities.reshape(-1), -1, 1)
        angular_distances = np.arccos(similarities)

        # Probabilité proportionnelle au carré de la distance
        probabilities = angular_distances ** 2
        # Éviter la division par zéro si toutes les distances sont nulles
        if np.sum(probabilities) > 0:
            probabilities = probabilities / np.sum(probabilities)
        else:
            # Si toutes les distances sont nulles, choisir uniformément
            probabilities = np.ones(n) / n

        # Sélectionner le prochain centroïde
        cumulative_probs = np.cumsum(probabilities)
        r = np.random.rand()

        # S'assurer que l'indice est valide
        if cumulative_probs[-1] > 0:
            idx = np.searchsorted(cumulative_probs, r)
            if idx >= n:  # Pour éviter les erreurs d'indexation
                idx = n - 1
        else:
            # En cas de problème avec les probabilités, choisir un indice aléatoire
            idx = np.random.randint(0, n)

        centroids[i] = vectors[idx]

    return centroids

def find_optimal_k(vectors: np.ndarray, k_min: int = 2, k_max: int = 32, gpu: bool = False) -> int:
    """
    Trouve le nombre optimal de clusters k en utilisant la méthode du coude.
    Adapté pour le k-means sphérique avec distance angulaire.

    Args:
        vectors: Les vecteurs à clusteriser
        k_min: Nombre minimum de clusters à tester
        k_max: Nombre maximum de clusters à tester
        gpu: Utiliser le GPU si disponible

    Returns:
        int: Nombre optimal de clusters k
    """
    # Limiter k_max par rapport au nombre de vecteurs
    k_max = min(k_max, len(vectors) // 2) if len(vectors) > 2 else k_min
    k_range = range(k_min, k_max + 1)

    # Si nous avons trop peu de vecteurs, retourner le minimum
    if len(k_range) <= 1:
        return k_min

    inertias = []

    # Calculer l'inertie pour différentes valeurs de k
    for k in tqdm(k_range, desc="Recherche du k optimal"):
        centroids, labels = kmeans_faiss(vectors, k, gpu=gpu, niter=5)  # Moins d'itérations pour la recherche

        # Calculer l'inertie adaptée au clustering sphérique
        inertia = 0
        for i in range(k):
            cluster_vectors = vectors[labels == i]
            if len(cluster_vectors) > 0:
                # Calculer les similarités avec le centroïde
                similarities = np.dot(cluster_vectors, centroids[i])
                # Convertir en distances angulaires et sommer
                angular_distances = np.arccos(np.clip(similarities, -1, 1))
                inertia += np.sum(angular_distances ** 2)

        inertias.append(inertia)

    # Utiliser KneeLocator si disponible, sinon utiliser une approche simplifiée
    if KNEEDLE_AVAILABLE:
        try:
            kneedle = KneeLocator(
                list(k_range), inertias,
                curve="convex", direction="decreasing", S=1.0
            )

            # Si un coude est trouvé, l'utiliser
            if kneedle.knee is not None:
                optimal_k = kneedle.knee
                return optimal_k
        except:
            pass  # Si KneeLocator échoue, utiliser l'approche simplifiée

    # Approche simplifiée: analyse des différences
    inertia_diffs = [inertias[i-1] - inertias[i] for i in range(1, len(inertias))]

    # Normaliser les différences
    if max(inertia_diffs) > 0:
        normalized_diffs = [d / max(inertia_diffs) for d in inertia_diffs]

        # Trouver l'indice où la différence normalisée devient inférieure à 0.2 (heuristique)
        for i, diff in enumerate(normalized_diffs):
            if diff < 0.2:
                return k_range[i]

    # Par défaut, retourner la médiane des k testés
    return (k_min + k_max) // 2

def select_closest_vectors(vectors: np.ndarray, all_indices: List[int], centroid: np.ndarray, max_data: int) -> np.ndarray:
    """
    Sélectionne les max_data vecteurs les plus proches du centroïde.

    Args:
        vectors: Tous les vecteurs disponibles
        all_indices: Indices de tous les vecteurs disponibles
        centroid: Centroïde de référence
        max_data: Nombre maximum de vecteurs à sélectionner

    Returns:
        np.ndarray: Tableau des indices des max_data vecteurs les plus proches du centroïde
    """
    # Calculer la similarité entre chaque vecteur et le centroïde
    similarities = np.dot(vectors, centroid)

    # Trier les indices par similarité décroissante
    sorted_indices = np.argsort(-similarities)

    # Sélectionner les max_data indices les plus proches
    selected_count = min(max_data, len(sorted_indices))
    selected_local_indices = sorted_indices[:selected_count]

    # Convertir les indices locaux en indices globaux
    if isinstance(all_indices, list):
        all_indices = np.array(all_indices, dtype=np.int32)
    
    selected_global_indices = all_indices[selected_local_indices]

    return selected_global_indices

def select_closest_natural_vectors(leaf_vectors: np.ndarray, leaf_indices: Union[List[int], np.ndarray],
                                  global_vectors: np.ndarray, global_indices: Union[List[int], np.ndarray],
                                  centroid: np.ndarray, max_data: int, sibling_indices: Optional[List[np.ndarray]] = None) -> np.ndarray:
    """
    Sélectionne d'abord les vecteurs qui tombent naturellement dans la feuille,
    puis complète avec les frères/sœurs, et enfin avec les plus proches au global si nécessaire.

    Args:
        leaf_vectors: Vecteurs qui tombent naturellement dans cette feuille
        leaf_indices: Indices des vecteurs qui tombent naturellement dans cette feuille
        global_vectors: Tous les vecteurs disponibles
        global_indices: Tous les indices disponibles
        centroid: Centroïde de référence
        max_data: Nombre maximum de vecteurs à sélectionner
        sibling_indices: Liste des indices des frères/sœurs (autres clusters du même niveau)

    Returns:
        np.ndarray: Tableau des indices des max_data vecteurs les plus proches du centroïde
    """
    # Conversion en arrays numpy si nécessaire
    if isinstance(leaf_indices, list):
        leaf_indices = np.array(leaf_indices, dtype=np.int32)
    if isinstance(global_indices, list):
        global_indices = np.array(global_indices, dtype=np.int32)
    
    # D'abord, prioriser les vecteurs qui tombent naturellement dans la feuille
    leaf_indices_set = set(leaf_indices.tolist())

    # Si la feuille contient déjà assez de vecteurs, pas besoin d'en chercher d'autres
    if len(leaf_indices) >= max_data:
        # Calculer la similarité entre les vecteurs de la feuille et le centroïde
        similarities = np.dot(leaf_vectors, centroid)
        # Trier les indices par similarité décroissante
        sorted_local_indices = np.argsort(-similarities)
        # Prendre les max_data plus proches
        selected_count = min(max_data, len(sorted_local_indices))
        return leaf_indices[sorted_local_indices[:selected_count]]

    # Sinon, compléter avec les frères/sœurs d'abord, puis au global
    result = leaf_indices.tolist()  # Commencer avec les vecteurs naturels
    used_indices = set(leaf_indices.tolist())  # Maintenir un set de tous les indices utilisés

    # Étape 1: Ajouter des vecteurs des frères/sœurs si disponibles
    if sibling_indices is not None:
        sibling_candidates = []
        for sibling_idx_array in sibling_indices:
            if len(sibling_idx_array) > 0:
                sibling_candidates.extend(sibling_idx_array.tolist())

        if sibling_candidates:
            # Calculer la similarité des vecteurs frères/sœurs avec le centroïde
            sibling_candidates = np.array(sibling_candidates, dtype=np.int32)
            sibling_vectors = global_vectors[sibling_candidates]
            sibling_similarities = np.dot(sibling_vectors, centroid)
            sibling_sorted_indices = np.argsort(-sibling_similarities)

            # Ajouter les meilleurs frères/sœurs qui ne sont pas déjà utilisés
            for idx in sibling_sorted_indices:
                global_idx = sibling_candidates[idx]
                if global_idx not in used_indices:
                    result.append(global_idx)
                    used_indices.add(global_idx)  # Mettre à jour le set d'exclusion
                    if len(result) >= max_data:
                        return np.array(result, dtype=np.int32)

    # Étape 2: Si encore pas assez, compléter avec les plus proches au global
    # Calculer la similarité entre tous les vecteurs et le centroïde
    similarities = np.dot(global_vectors, centroid)

    # Trier les indices par similarité décroissante
    sorted_indices = np.argsort(-similarities)

    # Ajouter les vecteurs les plus proches qui ne sont pas déjà utilisés
    for i in sorted_indices:
        global_idx = global_indices[i]
        if global_idx not in used_indices:
            result.append(global_idx)
            used_indices.add(global_idx)  # Mettre à jour le set d'exclusion
            if len(result) >= max_data:
                break

    return np.array(result, dtype=np.int32)

def build_tree_node(vectors: np.ndarray, current_indices: Union[List[int], np.ndarray], global_vectors: np.ndarray, global_indices: Union[List[int], np.ndarray],
                   level: int, max_depth: int, k_adaptive: bool, k: int, k_min: int, k_max: int,
                   max_leaf_size: int, max_data: int, use_gpu: bool = False, sibling_indices: Optional[List[np.ndarray]] = None) -> TreeNode:
    """
    Construit un nœud de l'arbre optimisé.

    Args:
        vectors: Les vecteurs assignés à ce nœud
        current_indices: Les indices des vecteurs assignés à ce nœud
        global_vectors: Tous les vecteurs de la collection complète
        global_indices: Tous les indices de la collection complète
        level: Niveau actuel dans l'arbre
        max_depth: Profondeur maximale de l'arbre
        k_adaptive: Utiliser la méthode du coude pour déterminer k
        k: Nombre fixe de clusters (si k_adaptive=False)
        k_min, k_max: Limites pour k adaptatif
        max_leaf_size: Taille maximale d'une feuille pour l'arrêt de la subdivision
        max_data: MAX_DATA - Nombre de vecteurs les plus proches à stocker dans chaque feuille
        use_gpu: Utiliser le GPU pour K-means

    Returns:
        TreeNode: Un nœud de l'arbre
    """
    # Conversion en arrays numpy si nécessaire
    if isinstance(global_indices, list):
        global_indices = np.array(global_indices, dtype=np.int32)
    if isinstance(current_indices, list):
        current_indices = np.array(current_indices, dtype=np.int32)

    # Vérifier si nous avons assez de vecteurs pour continuer
    if len(vectors) == 0:
        # Créer un nœud vide avec un centroïde aléatoire normalisé
        random_vector = np.random.randn(global_vectors.shape[1])
        random_centroid = random_vector / np.linalg.norm(random_vector)
        empty_node = TreeNode(level=level)
        empty_node.centroid = random_centroid
        return empty_node

    # Calculer le centroïde de ce nœud (normalisé pour le produit scalaire)
    centroid = np.mean(vectors, axis=0)
    # Normaliser le centroïde pour la similarité cosinus
    norm = np.linalg.norm(centroid)
    if norm > 0:  # Éviter la division par zéro
        centroid = centroid / norm
    else:
        # Si le centroïde est un vecteur nul (cas rare), créer un vecteur aléatoire normalisé
        random_vector = np.random.randn(vectors.shape[1])
        centroid = random_vector / np.linalg.norm(random_vector)

    # Créer une feuille si les conditions sont remplies
    if level >= max_depth or len(vectors) <= max_leaf_size:
        leaf = TreeNode(level=level)
        leaf.centroid = centroid

        # Sélectionner les max_data vecteurs les plus proches du centroïde
        # Priorité aux vecteurs qui tombent naturellement dans cette feuille, puis frères/sœurs, puis global
        leaf.set_indices(select_closest_natural_vectors(
            vectors, current_indices,
            global_vectors, global_indices,
            centroid, max_data, sibling_indices
        ))

        return leaf

    # Déterminer le nombre de clusters à utiliser
    if k_adaptive:
        if len(vectors) > 10000:
            sample_size = min(10000, len(vectors))
            sample_indices = np.random.choice(len(vectors), sample_size, replace=False)
            sample_vectors = vectors[sample_indices]
            used_k = find_optimal_k(sample_vectors, k_min, k_max, gpu=use_gpu)
        else:
            used_k = find_optimal_k(vectors, k_min, k_max, gpu=use_gpu)
    else:
        used_k = min(k, len(vectors))  # Ne pas créer plus de clusters que de vecteurs

    # Afficher la progression seulement pour les premiers niveaux
    if level <= 2:
        print(f"  → Niveau {level+1}: Clustering K-means avec k={used_k} sur {len(vectors):,} vecteurs...")

    # Appliquer K-means pour obtenir k clusters avec FAISS
    # En activant la suppression des clusters vides
    centroids, labels = kmeans_faiss(vectors, used_k, gpu=use_gpu, remove_empty=True)

    # Ajuster used_k au nombre réel de clusters après suppression des vides
    used_k = len(centroids)

    # Vérifier que les centroïdes sont normalisés
    for i in range(len(centroids)):
        norm = np.linalg.norm(centroids[i])
        if norm > 0:  # Éviter la division par zéro
            centroids[i] = centroids[i] / norm

    # Créer le noeud actuel avec son centroïde
    node = TreeNode(level=level)
    node.centroid = centroid

    # Créer des groupes pour chaque cluster en utilisant numpy
    cluster_indices = [[] for _ in range(used_k)]
    cluster_vectors = [[] for _ in range(used_k)]

    # Assigner chaque vecteur à son cluster
    for i, cluster_idx in enumerate(labels):
        cluster_vectors[cluster_idx].append(vectors[i])
        cluster_indices[cluster_idx].append(current_indices[i])
    
    # Convertir les listes en tableaux numpy
    cluster_vectors_np = [np.array(vec_list) if vec_list else np.array([]) for vec_list in cluster_vectors]
    cluster_indices_np = [np.array(idx_list, dtype=np.int32) if idx_list else np.array([], dtype=np.int32) for idx_list in cluster_indices]

    # Construire les enfants pour chaque cluster (récursivement)
    for i in range(used_k):
        if len(cluster_vectors_np[i]) > 0:
            # Préparer les indices des frères/sœurs (tous les autres clusters du même niveau)
            sibling_indices_list = [cluster_indices_np[j] for j in range(used_k) if j != i and len(cluster_indices_np[j]) > 0]

            # Pour les nœuds internes, passer les vecteurs et indices actuels
            child = build_tree_node(
                cluster_vectors_np[i],
                cluster_indices_np[i],
                global_vectors,
                global_indices,
                level + 1,
                max_depth,
                k_adaptive,
                k,
                k_min,
                k_max,
                max_leaf_size,
                max_data,
                use_gpu,
                sibling_indices_list
            )
            node.add_child(child)
        else:
            # Si un cluster est vide, créer une feuille basée sur le centroïde
            empty_centroid = centroids[i]
            empty_node = TreeNode(level=level + 1)
            empty_node.centroid = empty_centroid

            # Sélectionner les vecteurs globaux les plus proches de ce centroïde vide
            similarities = np.dot(global_vectors, empty_centroid)
            sorted_indices = np.argsort(-similarities)

            # Sélectionner jusqu'à max_data indices les plus proches
            selected_count = min(max_data, len(sorted_indices))
            empty_node.set_indices(global_indices[sorted_indices[:selected_count]])

            node.add_child(empty_node)

    # Bien que node.centroids soit déjà mis à jour dans add_child,
    # on s'assure qu'il contient tous les centroïdes alignés avec children
    node.set_children_centroids()
    
    return node

def process_cluster(i: int, vectors: np.ndarray, cluster_indices: np.ndarray, cluster_vectors: np.ndarray,
                   global_vectors: np.ndarray, global_indices: np.ndarray, level: int, max_depth: int,
                   k_adaptive: bool, k: int, k_min: int, k_max: int, max_leaf_size: int, max_data: int,
                   centroids: np.ndarray, use_gpu: bool = False) -> Tuple[int, Optional[TreeNode]]:
    """
    Fonction pour traiter un cluster en parallèle.

    Args:
        i: Indice du cluster
        vectors: Tous les vecteurs
        cluster_indices: Indices des vecteurs dans ce cluster
        cluster_vectors: Vecteurs dans ce cluster
        global_vectors: Tous les vecteurs de la collection complète
        global_indices: Tous les indices de la collection complète
        level: Niveau actuel dans l'arbre
        max_depth: Profondeur maximale de l'arbre
        k_adaptive: Utiliser la méthode du coude pour déterminer k
        k: Nombre fixe de clusters (si k_adaptive=False)
        k_min, k_max: Limites pour k adaptatif
        max_leaf_size: Taille maximale d'une feuille pour l'arrêt de la subdivision
        max_data: MAX_DATA - Nombre de vecteurs les plus proches à stocker dans chaque feuille
        centroids: Les centroïdes de tous les clusters
        use_gpu: Utiliser le GPU pour K-means

    Returns:
        Tuple[int, Optional[TreeNode]]: (indice du cluster, nœud construit)
    """
    if len(cluster_vectors) > 0:
        return i, build_tree_node(
            cluster_vectors,
            cluster_indices,
            global_vectors,
            global_indices,
            level,
            max_depth,
            k_adaptive,
            k,
            k_min,
            k_max,
            max_leaf_size,
            max_data,
            use_gpu
        )
    else:
        # Si un cluster est vide, créer une feuille basée sur le centroïde
        # Normaliser le centroïde pour la similarité cosinus
        centroid = centroids[i]
        centroid = centroid / np.linalg.norm(centroid)

        # Sélectionner les vecteurs globaux les plus proches de ce centroïde
        similarities = np.dot(global_vectors, centroid)
        sorted_indices = np.argsort(-similarities)

        # Créer une feuille avec le centroïde normalisé
        leaf = TreeNode(level=level)
        leaf.centroid = centroid

        # Sélectionner jusqu'à max_data indices les plus proches
        selected_count = min(max_data, len(sorted_indices))
        leaf.set_indices(global_indices[sorted_indices[:selected_count]])

        return i, leaf

def build_tree(vectors: np.ndarray, max_depth: int = 6, k: int = 16, k_adaptive: bool = False, 
              k_min: int = 2, k_max: int = 32, max_leaf_size: int = 100, max_data: int = 3000, 
              max_workers: Optional[int] = None, use_gpu: bool = False) -> TreeNode:
    """
    Construit un arbre à profondeur variable avec 'k' branches à chaque niveau,
    et pré-calcule les max_data indices les plus proches pour chaque feuille.

    Args:
        vectors: Les vecteurs à indexer
        max_depth: Profondeur maximale de l'arbre
        k: Nombre de branches par nœud (si k_adaptive=False)
        k_adaptive: Utiliser la méthode du coude pour déterminer k automatiquement
        k_min: Nombre minimum de clusters pour k adaptatif
        k_max: Nombre maximum de clusters pour k adaptatif
        max_leaf_size: Taille maximale d'une feuille pour l'arrêt de la subdivision
        max_data: MAX_DATA - Nombre de vecteurs les plus proches à stocker dans chaque feuille
        max_workers: Nombre maximum de processus parallèles
        use_gpu: Utiliser le GPU pour K-means si disponible

    Returns:
        TreeNode: Racine de l'arbre
    """
    if max_workers is None:
        max_workers = multiprocessing.cpu_count()

    # Vérifier si le GPU est vraiment disponible
    gpu_available = False
    if use_gpu:
        try:
            res = faiss.StandardGpuResources()
            # Test simple de création d'un index GPU
            test_index = faiss.GpuIndexFlatIP(res, 128)
            gpu_available = True
            print("✓ GPU FAISS détecté et disponible")
        except (AttributeError, RuntimeError) as e:
            print(f"⚠️ GPU demandé mais non disponible: {e}. Utilisation du CPU à la place.")
            use_gpu = False

    gpu_str = "GPU" if use_gpu else "CPU"
    print(f"⏳ Construction de l'arbre optimisé avec FAISS sur {gpu_str} (max {max_depth} niveaux) avec {'k adaptatif' if k_adaptive else f'k={k}'},")
    print(f"   max_leaf_size={max_leaf_size}, MAX_DATA={max_data}, max_workers={max_workers}")

    start_time = time.time()

    # Statistiques sur les feuilles
    leaf_sizes = []
    leaf_depths = []

    # Initialiser les indices globaux comme un tableau numpy
    global_indices = np.arange(len(vectors), dtype=np.int32)

    # Utiliser k=16 par défaut avec ajustement intelligent
    # pour les petits datasets ou les niveaux profonds
    vectors_count = len(vectors)
    min_points_per_cluster = max(1, max_leaf_size // 2)

    # Détermine le nombre optimal de clusters (k)
    if k_adaptive:
        # Mode adaptatif traditionnel (coude)
        if vectors_count > 10000:
            sample_size = 10000
            sample_indices = np.random.choice(vectors_count, sample_size, replace=False)
            sample_vectors = vectors[sample_indices]
            root_k = find_optimal_k(sample_vectors, k_min, k_max, gpu=use_gpu)
        else:
            root_k = find_optimal_k(vectors, k_min, k_max, gpu=use_gpu)
    else:
        # Ajustement automatique - k est réduit si le nombre de points est trop petit
        # k=16 est optimal pour SIMD, mais pas si on a trop peu de points
        if vectors_count < k * min_points_per_cluster:
            # Calculer un k qui garantit au moins min_points_per_cluster points par cluster
            adjusted_k = max(2, min(k, vectors_count // min_points_per_cluster))
            if adjusted_k != k:
                print(f"  → Ajustement automatique: k={k} → k={adjusted_k} (car {vectors_count} points < {k}×{min_points_per_cluster})")
            root_k = adjusted_k
        else:
            # Utiliser k=16 standard (optimal pour SIMD)
            root_k = k

    # Appliquer K-means pour le premier niveau avec FAISS
    print(f"  → Niveau 1: Clustering K-means avec k={root_k} sur {len(vectors):,} vecteurs...")
    centroids, labels = kmeans_faiss(vectors, root_k, gpu=use_gpu, remove_empty=True)

    # Ajuster root_k après suppression des clusters vides
    root_k = len(centroids)

    # Créer le noeud racine (avec centroïde normalisé)
    root_centroid = np.mean(centroids, axis=0)
    root_centroid = root_centroid / np.linalg.norm(root_centroid)
    root = TreeNode(level=0)
    root.centroid = root_centroid

    # Créer des groupes pour chaque cluster en utilisant numpy
    # Utiliser une liste de tableaux pour améliorer l'efficacité
    cluster_vectors = [[] for _ in range(root_k)]
    cluster_indices = [[] for _ in range(root_k)]

    # Assigner chaque vecteur à son cluster
    for i, cluster_idx in enumerate(labels):
        cluster_vectors[cluster_idx].append(vectors[i])
        cluster_indices[cluster_idx].append(global_indices[i])
    
    # Convertir en tableaux numpy
    cluster_vectors_np = [np.array(vec_list) if vec_list else np.array([]) for vec_list in cluster_vectors]
    cluster_indices_np = [np.array(idx_list, dtype=np.int32) if idx_list else np.array([], dtype=np.int32) for idx_list in cluster_indices]

    # Construction parallèle des sous-arbres de premier niveau
    print(f"  → Construction parallèle des sous-arbres avec {max_workers} workers...")

    # Créer la liste des arguments pour chaque cluster
    tasks = []
    for i in range(root_k):
        tasks.append((
            i,
            vectors,  # Pas utilisé directement dans le traitement des clusters
            cluster_indices_np[i] if i < len(cluster_indices_np) else np.array([], dtype=np.int32),
            cluster_vectors_np[i] if i < len(cluster_vectors_np) else np.array([]),
            vectors,  # Vecteurs globaux pour pré-calculer les indices proches
            global_indices,  # Indices globaux pour pré-calculer les indices proches
            1,  # Niveau des sous-arbres (1)
            max_depth,
            k_adaptive,
            k,
            k_min,
            k_max,
            max_leaf_size,
            max_data,
            centroids,  # Passer les centroïdes pour traiter les clusters vides
            use_gpu
        ))

    # Exécuter les tâches en parallèle avec joblib
    results = Parallel(n_jobs=max_workers, verbose=10)(
        delayed(process_cluster)(*task) for task in tasks
    )

    # Reconstruire l'arbre à partir des résultats
    for i, child in results:
        if child is not None:
            root.add_child(child)
        else:
            # Ce cas ne devrait pas arriver avec le nouveau process_cluster
            # mais gardons-le par précaution
            empty_node = TreeNode(level=1)
            empty_node.centroid = centroids[i]
            root.add_child(empty_node)
    
    # S'assurer que root.centroids est bien initialisé
    root.set_children_centroids()
    
    # Collecter des statistiques sur les feuilles
    def collect_leaf_stats(node, level=0):
        if not node.children:  # C'est une feuille
            leaf_sizes.append(len(node.indices))
            leaf_depths.append(level)
        else:
            for child in node.children:
                collect_leaf_stats(child, level + 1)
    
    collect_leaf_stats(root)
    
    elapsed = time.time() - start_time

    # Afficher des statistiques sur les feuilles
    if leaf_sizes:
        avg_size = sum(leaf_sizes) / len(leaf_sizes)
        min_size = min(leaf_sizes)
        max_size = max(leaf_sizes)
        avg_depth = sum(leaf_depths) / len(leaf_depths)
        min_depth = min(leaf_depths)
        max_depth_seen = max(leaf_depths)

        print(f"✓ Construction de l'arbre optimisé terminée en {elapsed:.2f}s")
        print(f"  → Statistiques des feuilles:")
        print(f"     - Nombre de feuilles   : {len(leaf_sizes):,}")
        print(f"     - Taille moyenne       : {avg_size:.1f} vecteurs")
        print(f"     - Taille minimale      : {min_size} vecteurs")
        print(f"     - Taille maximale      : {max_size} vecteurs")
        print(f"     - Profondeur moyenne   : {avg_depth:.1f}")
        print(f"     - Profondeur minimale  : {min_depth}")
        print(f"     - Profondeur maximale  : {max_depth_seen}")
    else:
        print(f"✓ Construction de l'arbre optimisé terminée en {elapsed:.2f}s")

    return root
```

### k16/builder/builder.py

```python
"""
Constructeur simplifié d'arbres K16 optimisés.
Une seule fonction qui fait tout: construction, réduction, HNSW, optimisation.
"""

import os
from typing import Optional, Union, Dict, Any
import numpy as np

from k16.utils.config import ConfigManager
from k16.core.tree import K16Tree, TreeNode
from k16.core.flat_tree import TreeFlat
from k16.io.reader import VectorReader
from k16.utils.optimization import configure_simd

def build_optimized_tree(
    vectors: Union[np.ndarray, str],
    output_file: str,
    config: Optional[Dict[str, Any]] = None,
    max_depth: Optional[int] = None,
    max_leaf_size: Optional[int] = None,
    max_data: Optional[int] = None,
    max_dims: Optional[int] = None,
    use_hnsw: Optional[bool] = None,
    k: Optional[int] = None,
    k_adaptive: Optional[bool] = None,
    verbose: bool = True
) -> TreeFlat:
    """
    Construit un arbre K16 optimisé en une seule fonction.

    Cette fonction fait tout:
    1. Chargement des vecteurs si nécessaire
    2. Construction de l'arbre hiérarchique avec k=16 par défaut
    3. Réduction dimensionnelle
    4. Amélioration HNSW si activée
    5. Optimisation des centroïdes (économie ~56% d'espace)
    6. Sauvegarde de l'arbre

    Args:
        vectors: Soit un tableau numpy de vecteurs, soit un chemin vers un fichier de vecteurs
        output_file: Chemin vers le fichier de sortie pour sauvegarder l'arbre
        config: Configuration personnalisée (facultatif, sinon utilise config.yaml)
        max_depth: Profondeur maximale de l'arbre (facultatif)
        max_leaf_size: Taille maximale des feuilles (facultatif)
        max_data: Nombre de vecteurs à stocker par feuille (facultatif)
        max_dims: Nombre de dimensions à conserver (facultatif)
        use_hnsw: Activer l'amélioration HNSW (facultatif)
        k: Nombre de branches par nœud (16 par défaut)
        k_adaptive: Utiliser k adaptatif (désactivé par défaut)
        verbose: Afficher les messages de progression

    Returns:
        L'arbre optimisé construit
    """
    # S'assurer que les optimisations SIMD sont configurées
    configure_simd()
    
    # 1. Charger la configuration
    if config is None:
        config_manager = ConfigManager()
        build_config = config_manager.get_section("build_tree")
        flat_config = config_manager.get_section("flat_tree")
    else:
        build_config = config.get("build_tree", {})
        flat_config = config.get("flat_tree", {})
    
    # 2. Utiliser les paramètres explicites ou les valeurs de configuration
    max_depth = max_depth if max_depth is not None else build_config.get("max_depth", 32)
    max_leaf_size = max_leaf_size if max_leaf_size is not None else build_config.get("max_leaf_size", 50)
    max_data = max_data if max_data is not None else build_config.get("max_data", 256)  # Optimisé pour SIMD (multiple de 16)
    max_dims = max_dims if max_dims is not None else flat_config.get("max_dims", 512)  # Optimisé pour SIMD (multiple de 16)
    k_adaptive = k_adaptive if k_adaptive is not None else build_config.get("k_adaptive", False)
    use_hnsw = use_hnsw if use_hnsw is not None else build_config.get("use_hnsw_improvement", True)

    # Autres paramètres de configuration
    k = k if k is not None else build_config.get("k", 16)  # Valeur fixe k=16 par défaut
    k_min = build_config.get("k_min", 2)
    k_max = build_config.get("k_max", 32)
    max_workers = build_config.get("max_workers", None)
    reduction_method = flat_config.get("reduction_method", "directional")
    
    # 3. Préparer les vecteurs
    if isinstance(vectors, str):
        if verbose:
            print(f"⏳ Chargement des vecteurs depuis {vectors}...")
        vectors_reader = VectorReader(file_path=vectors, mode="ram")
        if verbose:
            print(f"✓ {len(vectors_reader):,} vecteurs chargés (dim {vectors_reader.d})")
        vectors_data = vectors_reader.vectors
    else:
        vectors_data = vectors
        vectors_reader = None
    
    # 4. Créer le répertoire de sortie si nécessaire
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    
    # 5. Construction de l'arbre hiérarchique avec k-means
    if verbose:
        print(f"⏳ Construction de l'arbre K16 avec max_depth={max_depth}, "
              f"max_leaf_size={max_leaf_size}, max_data={max_data}, "
              f"{'k adaptatif' if k_adaptive else f'k={k}'}")
              
    # Importation locale pour éviter les dépendances circulaires
    from k16.builder.clustering import build_tree as clustering_build_tree
    
    tree = clustering_build_tree(
        vectors=vectors_data,
        max_depth=max_depth,
        k=k,
        k_adaptive=k_adaptive,
        k_min=k_min,
        k_max=k_max,
        max_leaf_size=max_leaf_size,
        max_data=max_data,
        max_workers=max_workers
    )
    
    # 6. Calcul de la réduction dimensionnelle et conversion en K16Tree
    k16tree = K16Tree(tree)
    
    if verbose:
        print(f"⏳ Calcul de la réduction dimensionnelle (max_dims={max_dims}, method={reduction_method})...")
    
    k16tree.compute_dimensional_reduction(max_dims=max_dims, method=reduction_method)
    
    # 7. Conversion en structure plate
    if verbose:
        print(f"⏳ Conversion en structure plate...")
    
    flat_tree = TreeFlat.from_tree(k16tree)
    k16tree.flat_tree = flat_tree
    
    # 8. Amélioration optionnelle avec HNSW
    if use_hnsw:
        if verbose:
            print(f"⏳ Amélioration avec HNSW...")
        
        if vectors_reader is None:
            # Créer un lecteur de vecteurs si on n'en a pas déjà un
            vectors_reader = VectorReader(vectors=vectors_data)
            
        k16tree = k16tree.improve_with_hnsw(vectors_reader, max_data)
        flat_tree = k16tree.flat_tree
        
        if verbose:
            print(f"✓ Amélioration HNSW terminée!")
    elif verbose:
        print(f"ℹ️ Amélioration HNSW désactivée")
    
    # 9. Sauvegarde de l'arbre (avec optimisation automatique)
    if output_file:
        # Assurer l'extension correcte
        if not (output_file.endswith(".flat.npy") or output_file.endswith(".flat")):
            output_file = os.path.splitext(output_file)[0] + ".flat.npy"
            
        if verbose:
            print(f"⏳ Sauvegarde de l'arbre vers {output_file}...")
            
        flat_tree.save(output_file)  # L'optimisation est faite automatiquement
        
        if verbose:
            print(f"✓ Arbre optimisé sauvegardé dans {output_file}")
    
    # 10. Nettoyer
    if vectors_reader is not None and isinstance(vectors, str):
        vectors_reader.close()
    
    return flat_tree

def build_tree(
    vectors: np.ndarray,
    max_depth: int = 6, 
    max_leaf_size: int = 100, 
    max_data: int = 256,
    max_dims: int = 512,
    use_hnsw: bool = True,
    k: int = 16,
    k_adaptive: bool = False,
    output_file: Optional[str] = None
) -> K16Tree:
    """
    Fonction simplifiée pour construire un arbre K16.
    
    Args:
        vectors: Vecteurs à indexer
        max_depth: Profondeur maximale de l'arbre
        max_leaf_size: Taille maximale d'une feuille pour l'arrêt de la subdivision
        max_data: Nombre de vecteurs à stocker par feuille
        max_dims: Nombre de dimensions à conserver pour la réduction dimensionnelle
        use_hnsw: Activer l'amélioration HNSW
        k: Nombre de branches par nœud
        k_adaptive: Utiliser k adaptatif
        output_file: Chemin vers le fichier de sortie (facultatif)
        
    Returns:
        K16Tree: Arbre K16 construit
    """
    # Utiliser build_optimized_tree avec des paramètres par défaut
    flat_tree = build_optimized_tree(
        vectors=vectors,
        output_file=output_file or "",  # Chaîne vide pour ne pas sauvegarder si None
        max_depth=max_depth,
        max_leaf_size=max_leaf_size,
        max_data=max_data,
        max_dims=max_dims,
        use_hnsw=use_hnsw,
        k=k,
        k_adaptive=k_adaptive,
        verbose=True
    )
    
    # Créer un K16Tree à partir de l'arbre plat
    tree = K16Tree(None)
    tree.flat_tree = flat_tree
    
    return tree
```

### k16/core/tree.py

```python
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
```

### k16/core/flat_tree.py

```python
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
```

