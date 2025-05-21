#!/usr/bin/env python3
"""
Script de test pour l'arbre plat optimisé.
Compare les performances de recherche entre l'arbre normal et l'arbre plat.
"""

import os
import sys
import time
import numpy as np
from typing import Dict, List, Any
from argparse import ArgumentParser
from tqdm.auto import tqdm

# Ajouter le répertoire parent au chemin
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importer les modules de K16
from lib.config import ConfigManager
from lib.io import VectorReader, TreeIO
from lib.search import Searcher

# Importer l'arbre plat
from flat_tree import TreeFlat

def evaluate_flat_tree(tree_path: str, vectors_path: str, mode: str = "ram", 
                     k: int = 10, queries: int = 100, beam_width: int = 1):
    """
    Évalue les performances de l'arbre plat comparé à l'arbre classique.
    
    Args:
        tree_path: Chemin vers le fichier d'arbre
        vectors_path: Chemin vers le fichier de vecteurs
        mode: Mode de chargement des vecteurs ("ram" ou "mmap")
        k: Nombre de voisins à rechercher
        queries: Nombre de requêtes à effectuer
        beam_width: Largeur du faisceau pour la recherche
    """
    print(f"🔍 K16 Flat Tree - Mode: {mode.upper()}, k={k}")
    print(f"  - Configuration: {tree_path}")
    print(f"  - Vecteurs: {vectors_path}")
    print(f"  - Paramètres: mode={mode}, beam_width={beam_width}, k={k}, queries={queries}")
    
    # Charger les vecteurs
    print(f"⏳ Chargement des vecteurs depuis {vectors_path} en mode {mode}...")
    start_time = time.time()
    
    vectors_reader = VectorReader(vectors_path, mode=mode)
    
    load_time = time.time() - start_time
    print(f"✓ {len(vectors_reader):,} vecteurs (dim {vectors_reader.d}) prêts en mode {mode} [terminé en {load_time:.2f}s]")
    # Afficher la mémoire utilisée (approximative)
    memory_usage = vectors_reader.vectors.nbytes / (1024**2) if mode == "ram" else "Mmap + Cache"
    print(f"  → Mémoire utilisée: {memory_usage if isinstance(memory_usage, str) else f'{memory_usage:.1f} MB'}")
    
    # Charger l'arbre standard
    print(f"⏳ Chargement de l'arbre depuis {tree_path}...")
    start_time = time.time()
    
    tree_io = TreeIO()
    standard_tree = tree_io.load_as_k16tree(tree_path)
    
    load_time = time.time() - start_time
    print(f"✓ Arbre chargé depuis {tree_path} en {load_time:.2f}s")
    
    # Convertir l'arbre en structure plate
    print("⏳ Conversion de l'arbre en structure plate optimisée...")
    start_time = time.time()
    
    flat_tree = TreeFlat.from_tree(standard_tree)
    
    convert_time = time.time() - start_time
    flat_stats = flat_tree.get_statistics()
    print(f"✓ Arbre converti en structure plate en {convert_time:.2f}s")
    print(f"  → Statistiques: {flat_tree.n_nodes:,} nœuds, {flat_tree.n_leaves:,} feuilles")
    
    # Initialiser le chercheur standard
    searcher_standard = Searcher(
        standard_tree.root,
        vectors_reader,
        use_faiss=True,
        search_type="beam" if beam_width > 1 else "single",
        beam_width=beam_width
    )
    
    # Générer des requêtes aléatoires
    print(f"\n⏳ Génération de {queries} requêtes aléatoires...")
    
    # Récupérer des vecteurs aléatoires de la base de données
    query_indices = np.random.choice(len(vectors_reader), queries, replace=False)
    query_vectors = vectors_reader[query_indices]
    
    # Trouver la vérité terrain et comparer les résultats
    print(f"\n⏳ Évaluation avec {queries} requêtes aléatoires...")

    standard_times = []
    standard_candidates = []
    flat_times = []
    flat_candidates = []
    recalls_standard = []
    recalls_flat = []

    for i, query in enumerate(tqdm(query_vectors, desc="Évaluation")):
        # Recherche standard
        start_time = time.time()
        standard_results = searcher_standard.search_k_nearest(query, k)
        standard_time = time.time() - start_time
        standard_times.append(standard_time)
        standard_candidates.append(len(standard_results))

        # Recherche flat tree
        start_time = time.time()
        flat_results = flat_tree.search_tree(query, beam_width)
        flat_time = time.time() - start_time
        flat_times.append(flat_time)
        flat_candidates.append(len(flat_results))

        # Trouver la vérité terrain avec recherche naïve
        ground_truth = searcher_standard.brute_force_search(query, k)

        # Calculer le recall pour standard
        intersection = set(standard_results).intersection(set(ground_truth))
        recall_standard = len(intersection) / k if k > 0 else 0
        recalls_standard.append(recall_standard)

        # Calculer le recall pour flat
        intersection = set(flat_results).intersection(set(ground_truth))
        recall_flat = len(intersection) / k if k > 0 else 0
        recalls_flat.append(recall_flat)

        # Afficher les progrès pour certaines requêtes
        if (i + 1) % 10 == 0 or (i + 1) == queries:
            print(f"  → Requête {i+1}/{queries}: Standard recall = {recall_standard:.4f}, Flat recall = {recall_flat:.4f}")
    
    # Calculer les moyennes et autres statistiques
    avg_standard_time = np.mean(standard_times)
    avg_flat_time = np.mean(flat_times)
    speedup = avg_standard_time / avg_flat_time if avg_flat_time > 0 else 0
    
    avg_candidates_standard = np.mean(standard_candidates) if standard_candidates else 0
    avg_candidates_flat = np.mean(flat_candidates) if flat_candidates else 0
    
    avg_recall_standard = np.mean(recalls_standard)
    avg_recall_flat = np.mean(recalls_flat)
    
    # Afficher les résultats
    print("\n✓ Résultats de l'évaluation comparative:")
    print(f"  - Nombre de requêtes     : {queries}")
    print(f"  - k (voisins demandés)   : {k}")
    print(f"  - Mode                   : {mode.upper()}")
    print(f"  - Beam width             : {beam_width}")
    
    print("\nStructure standard:")
    print(f"  - Temps moyen (ms)       : {avg_standard_time*1000:.2f} ms")
    print(f"  - Recall moyen           : {avg_recall_standard:.4f} ({avg_recall_standard*100:.2f}%)")
    print(f"  - Candidats moyens       : {avg_candidates_standard:.1f}")
    
    print("\nStructure plate:")
    print(f"  - Temps moyen (ms)       : {avg_flat_time*1000:.2f} ms")
    print(f"  - Recall moyen           : {avg_recall_flat:.4f} ({avg_recall_flat*100:.2f}%)")
    print(f"  - Candidats moyens       : {avg_candidates_flat:.1f}")
    
    print(f"\nAccélération avec structure plate: {speedup:.2f}x")
    
    if avg_recall_flat > avg_recall_standard:
        print(f"Amélioration du recall: +{(avg_recall_flat - avg_recall_standard)*100:.2f}%")
    else:
        print(f"Différence de recall: {(avg_recall_flat - avg_recall_standard)*100:.2f}%")

def main():
    parser = ArgumentParser(description="Test des performances de l'arbre plat optimisé")
    parser.add_argument("--vectors", help="Chemin vers le fichier de vecteurs")
    parser.add_argument("--tree", help="Chemin vers le fichier d'arbre")
    parser.add_argument("--mode", choices=["ram", "mmap"], default="ram", help="Mode de chargement des vecteurs (défaut: ram)")
    parser.add_argument("--k", type=int, default=100, help="Nombre de voisins à rechercher (défaut: 100)")
    parser.add_argument("--queries", type=int, default=100, help="Nombre de requêtes aléatoires (défaut: 100)")
    parser.add_argument("--beam-width", type=int, default=1, help="Largeur du faisceau pour la recherche (défaut: 1)")
    
    args = parser.parse_args()
    
    # Charger la configuration
    config = ConfigManager()
    files_config = config.get_section("files")
    
    # Déterminer les chemins de fichiers
    vectors_path = args.vectors or os.path.join(files_config["vectors_dir"], files_config["default_vectors"])
    tree_path = args.tree or os.path.join(files_config["trees_dir"], files_config["default_tree"])
    
    # Exécuter l'évaluation
    evaluate_flat_tree(
        tree_path=tree_path,
        vectors_path=vectors_path,
        mode=args.mode,
        k=args.k,
        queries=args.queries,
        beam_width=args.beam_width
    )

if __name__ == "__main__":
    main()