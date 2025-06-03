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