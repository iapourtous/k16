"""
Module pour les tests de performance.
Fournit des outils pour évaluer les performances de K16.
"""

import os
import time
import datetime
import numpy as np
import argparse
from typing import List, Dict, Any, Tuple

from k16.utils.config import ConfigManager
from k16.io.reader import read_vectors, load_tree
from k16.search.searcher import Searcher

def format_time(seconds: float) -> str:
    """Formate le temps en heures, minutes, secondes."""
    return str(datetime.timedelta(seconds=int(seconds)))

def test_command(args: argparse.Namespace) -> int:
    """
    Commande pour tester les performances de recherche dans un arbre.

    Args:
        args: Arguments de ligne de commande

    Returns:
        int: Code de retour (0 pour succès, autre pour erreur)
    """
    # Initialisation du gestionnaire de configuration
    config_manager = ConfigManager(args.config)

    # Récupération des paramètres pour la recherche
    search_config = config_manager.get_section("search")
    files_config = config_manager.get_section("files")

    try:
        evaluate_msg = "avec évaluation des performances" if args.evaluate else "sans évaluation"
        print(f"🔍 Test de l'arbre K16 {evaluate_msg}...")
        print(f"  - Vecteurs: {args.vectors_file}")
        print(f"  - Arbre: {args.tree_file}")
        print(f"  - Mode: {args.mode}")
        print(f"  - K (nombre de voisins): {args.k}")
        print(f"  - Type de recherche: {args.search_type}")
        print(f"  - Largeur de faisceau: {args.beam_width}")
        print(f"  - Requêtes de test: {args.queries}")

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

        # Sélection des requêtes de test
        print(f"⏳ Sélection de {args.queries} requêtes aléatoires...")
        np.random.seed(42)  # Pour des résultats reproductibles

        if args.queries < len(vectors_reader):
            query_indices = np.random.choice(len(vectors_reader), args.queries, replace=False)
        else:
            query_indices = np.arange(len(vectors_reader))

        query_vectors = vectors_reader[query_indices]
        print(f"✓ {len(query_indices)} requêtes sélectionnées")

        # Créer un chercheur pour l'évaluation
        searcher = Searcher(
            k16tree=tree,
            vectors_reader=vectors_reader,
            use_faiss=args.use_faiss,
            search_type=args.search_type,
            beam_width=args.beam_width
        )

        # Évaluer les performances avec recall et accélération
        print(f"⏳ Évaluation des performances avec k={args.k}, mode={args.search_type}...")
        results = searcher.evaluate_search(query_vectors, k=args.k)

        # Afficher un récapitulatif des résultats
        print(f"\n✓ Évaluation terminée")
        print(f"  → Recall: {results['avg_recall']*100:.2f}%")
        print(f"  → Accélération: {results['speedup']:.2f}x plus rapide que la recherche naïve")
        print(f"  → Temps moyen (arbre): {results['avg_tree_time']*1000:.2f} ms")
        print(f"  → Temps moyen (filtrage): {results['avg_filter_time']*1000:.2f} ms")
        print(f"  → Temps moyen total: {results['avg_total_time']*1000:.2f} ms")
        print(f"  → Temps moyen (naïf): {results['avg_naive_time']*1000:.2f} ms")
        print(f"  → Nombre moyen de candidats: {results['avg_candidates']:.1f}")

        # Conseils d'optimisation
        print(f"\n💡 Conseils d'optimisation:")
        print(f"  → Utilisez le mode 'ram' pour la vitesse maximale si la mémoire le permet")
        print(f"  → Utilisez le mode 'mmap' pour économiser la mémoire avec peu d'impact sur les performances")
        print(f"  → Augmentez max_data lors de la construction pour améliorer le recall")

    except Exception as e:
        print(f"\n❌ Erreur: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

    return 0