#!/usr/bin/env python3
"""
Recherche ultra-rapide de voisins dans un arbre K16 optimisé.
Cette version utilise les modules optimisés de la bibliothèque K16.
Supporte deux modes de chargement des embeddings: RAM et mmap.
Utilise un fichier de configuration YAML central.

Usage:
    # Mode RAM (vecteurs en mémoire)
    python search_new.py vectors.bin tree.bin --k 100 --mode ram

    # Mode mmap (vecteurs mappés)
    python search_new.py vectors.bin tree.bin --k 100 --mode mmap

    # Utilisation de la configuration
    python search_new.py --config config.yaml
"""

import os
import sys
import argparse
import random
import time
import datetime

# Add the parent directory to sys.path to make the lib modules importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importation des modules de la bibliothèque K16
from lib.config import ConfigManager
from lib.io import VectorReader, TreeIO
from lib.search import Searcher

def format_time(seconds):
    """Formate le temps en heures, minutes, secondes."""
    return str(datetime.timedelta(seconds=int(seconds)))

def main():
    # Initialisation du gestionnaire de configuration
    config_manager = ConfigManager()
    
    # Récupération des paramètres pour la recherche
    search_config = config_manager.get_section("search")
    files_config = config_manager.get_section("files")
    
    # Définir les chemins par défaut
    default_vectors_path = os.path.join(files_config["vectors_dir"], files_config["default_vectors"])
    default_tree_path = os.path.join(files_config["trees_dir"], files_config["default_tree"])
    
    # Analyse des arguments de la ligne de commande
    parser = argparse.ArgumentParser(description="Recherche ultra-rapide de voisins dans un arbre K16 optimisé")
    parser.add_argument("vectors_file", nargs="?", default=default_vectors_path,
                        help=f"Fichier binaire contenant les vecteurs embeddings (par défaut: {default_vectors_path})")
    parser.add_argument("tree_file", nargs="?", default=default_tree_path,
                        help=f"Fichier contenant l'arbre K16 optimisé (par défaut: {default_tree_path})")
    parser.add_argument("--config", default=config_manager.config_path,
                        help=f"Chemin vers le fichier de configuration (par défaut: {config_manager.config_path})")
    parser.add_argument("--k", type=int, default=search_config["k"],
                        help=f"Nombre de voisins à rechercher (par défaut: {search_config['k']})")
    parser.add_argument("--queries", type=int, default=search_config["queries"],
                        help=f"Nombre de requêtes aléatoires à tester (par défaut: {search_config['queries']})")
    parser.add_argument("--mode", choices=["ram", "mmap"], default=search_config["mode"],
                        help=f"Mode de chargement des vecteurs (par défaut: {search_config['mode']})")
    parser.add_argument("--cache-size", type=int, default=search_config.get("cache_size_mb", 500),
                        help=f"Taille du cache en mégaoctets pour le mode mmap (par défaut: {search_config.get('cache_size_mb', 500)})")
    parser.add_argument("--use-faiss", action="store_true", default=search_config.get("use_faiss", True),
                        help=f"Utiliser FAISS pour la recherche (par défaut: {search_config.get('use_faiss', True)})")
    parser.add_argument("--no-faiss", action="store_false", dest="use_faiss",
                        help="Ne pas utiliser FAISS pour la recherche")
    
    args = parser.parse_args()
    
    # Recharger la configuration si un fichier spécifique est fourni
    if args.config != config_manager.config_path:
        config_manager = ConfigManager(args.config)
        print(f"✓ Configuration chargée depuis: {args.config}")
    
    # Enregistrer le temps de départ pour calculer la durée totale
    total_start_time = time.time()
    
    print(f"\n🔍 K16 Search - Mode: {args.mode.upper()}, k={args.k}")
    print(f"  - Configuration: {args.config}")
    print(f"  - Vecteurs: {args.vectors_file}")
    print(f"  - Arbre: {args.tree_file}")
    cache_info = f", Cache: {args.cache_size} MB" if args.mode.lower() == "mmap" else ""
    faiss_info = ", FAISS actif" if args.use_faiss else ", FAISS inactif"
    print(f"  - Paramètres: mode={args.mode}{cache_info}{faiss_info}, k={args.k}, queries={args.queries}")
    
    try:
        # Initialisation du lecteur de vecteurs
        vectors_reader = VectorReader(
            file_path=args.vectors_file,
            mode=args.mode,
            cache_size_mb=args.cache_size
        )
        
        # Chargement de l'arbre
        tree, _ = TreeIO.load_tree(args.tree_file)  # Unpack the tuple to get just the tree

        # Initialisation du chercheur avec les paramètres de configuration
        search_config = config_manager.get_section("search")
        build_config = config_manager.get_section("build_tree")
        searcher = Searcher(
            tree,
            vectors_reader,
            use_faiss=args.use_faiss,
            search_type=search_config.get("search_type", "single"),
            beam_width=search_config.get("beam_width", 3),
            max_data=build_config.get("max_data", 4000)
        )
        
        # Générer des requêtes aléatoires
        print(f"\n⏳ Génération de {args.queries} requêtes aléatoires...")
        query_indices = random.sample(range(len(vectors_reader)), args.queries)
        queries = vectors_reader[query_indices]
        
        # Évaluer la recherche
        results = searcher.evaluate_search(queries, k=args.k)
        
        total_time = time.time() - total_start_time
        print(f"\n✓ Évaluation terminée en {format_time(total_time)}")
        print(f"\n💡 Conseil: Pour optimiser la recherche, ajustez les paramètres dans {args.config}:")
        print(f"   - Utilisez le mode 'ram' pour la vitesse maximale si la mémoire le permet")
        print(f"   - Utilisez le mode 'mmap' pour économiser la mémoire avec peu d'impact sur les performances")
        print(f"   - Augmentez max_data lors de la construction pour améliorer le recall")
        
    except Exception as e:
        print(f"\n❌ Erreur: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        # Libérer les ressources
        if 'vectors_reader' in locals():
            vectors_reader.close()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())