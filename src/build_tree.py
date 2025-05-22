#!/usr/bin/env python3
"""
Construction d'un arbre optimisé pour la recherche rapide d'embeddings similaires.
Utilise les modules optimisés de la bibliothèque K16.
Cette version pré-calcule directement les MAX_DATA indices les plus proches pour chaque feuille.
Utilise un fichier de configuration YAML central.

Utilisation :
    python build_tree_new.py [--mmap-tree] vectors.bin tree.bin
    python build_tree_new.py [--mmap-tree] --config /path/to/config.yaml
"""

import os
import sys
import argparse
import time
import datetime

# Add the parent directory to sys.path to make the lib modules importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importation des modules de la bibliothèque K16
from lib.config import ConfigManager
from lib.io import VectorReader, TreeIO
from lib.clustering import build_tree

def format_time(seconds):
    """Formate le temps en heures, minutes, secondes."""
    return str(datetime.timedelta(seconds=int(seconds)))

def main():
    # Initialisation du gestionnaire de configuration
    config_manager = ConfigManager()
    
    # Récupération des paramètres pour la construction de l'arbre
    build_config = config_manager.get_section("build_tree")
    files_config = config_manager.get_section("files")
    
    # Définir les chemins par défaut
    default_vectors_path = os.path.join(files_config["vectors_dir"], files_config["default_vectors"])
    default_tree_path = os.path.join(files_config["trees_dir"], files_config["default_tree"])
    
    # Analyse des arguments de la ligne de commande
    parser = argparse.ArgumentParser(description="Construction d'un arbre optimisé pour la recherche rapide d'embeddings")
    parser.add_argument("vectors_file", nargs="?", default=default_vectors_path,
                        help=f"Fichier binaire contenant les vecteurs embeddings (par défaut: {default_vectors_path})")
    parser.add_argument("tree_file", nargs="?", default=default_tree_path,
                        help=f"Fichier de sortie pour l'arbre (par défaut: {default_tree_path})")
    parser.add_argument("--config", default=config_manager.config_path,
                        help=f"Chemin vers le fichier de configuration (par défaut: {config_manager.config_path})")
    parser.add_argument("--max_depth", type=int, default=build_config["max_depth"],
                        help=f"Profondeur maximale de l'arbre (par défaut: {build_config['max_depth']})")
    parser.add_argument("--k", type=int, default=build_config["k"],
                        help=f"Nombre de branches par nœud (par défaut: {build_config['k']}, ignoré si --k_adaptive est utilisé)")
    parser.add_argument("--k_adaptive", action="store_true", default=build_config["k_adaptive"],
                        help="Utiliser la méthode du coude pour déterminer k automatiquement")
    parser.add_argument("--k_min", type=int, default=build_config["k_min"],
                        help=f"Nombre minimum de clusters pour k adaptatif (par défaut: {build_config['k_min']})")
    parser.add_argument("--k_max", type=int, default=build_config["k_max"],
                        help=f"Nombre maximum de clusters pour k adaptatif (par défaut: {build_config['k_max']})")
    parser.add_argument("--max_leaf_size", type=int, default=build_config["max_leaf_size"],
                        help=f"Taille maximale d'une feuille pour l'arrêt de la subdivision (par défaut: {build_config['max_leaf_size']})")
    parser.add_argument("--max_data", type=int, default=build_config["max_data"],
                        help=f"MAX_DATA: Nombre de vecteurs à stocker dans chaque feuille (par défaut: {build_config['max_data']})")
    parser.add_argument("--max_workers", type=int, default=build_config["max_workers"],
                        help=f"Nombre maximum de processus parallèles (par défaut: {build_config['max_workers']})")
    parser.add_argument("--gpu", action="store_true", default=build_config["use_gpu"],
                        help="Utiliser le GPU pour K-means si disponible")
    parser.add_argument("--mmap-tree", action="store_true", default=False,
                        help="Sauvegarder la structure plate en format répertoire pour 'mmap+' (memory-mapping de l'arbre)")
    
    args = parser.parse_args()
    
    # Recharger la configuration si un fichier spécifique est fourni
    if args.config != config_manager.config_path:
        config_manager = ConfigManager(args.config)
        print(f"✓ Configuration chargée depuis: {args.config}")
    
    # Enregistrer le temps de départ pour calculer la durée totale
    total_start_time = time.time()
    
    try:
        # Créer les répertoires de sortie si nécessaires
        os.makedirs(os.path.dirname(os.path.abspath(args.tree_file)), exist_ok=True)
        
        # Chargement des vecteurs
        # Utiliser le mode RAM pour la construction de l'arbre pour des performances optimales
        vectors_reader = VectorReader(args.vectors_file, mode="ram")
        
        # Construction de l'arbre optimisé
        tree = build_tree(
            vectors=vectors_reader.vectors,
            max_depth=args.max_depth,
            k=args.k,
            k_adaptive=args.k_adaptive,
            k_min=args.k_min,
            k_max=args.k_max,
            max_leaf_size=args.max_leaf_size,
            max_data=args.max_data,
            max_workers=args.max_workers,
            use_gpu=args.gpu
        )
        
        # L'optimisation des indices avec HNSW a été retirée car elle n'améliore pas le recall

        # Conversion et sauvegarde directe en structure plate
        flat_path = args.tree_file
        if not (flat_path.endswith(".flat.npy") or flat_path.endswith(".flat")):
            flat_path = os.path.splitext(flat_path)[0] + ".flat.npy"
        print(f"⏳ Conversion et sauvegarde de la structure plate vers {flat_path}...")
        from lib.flat_tree import TreeFlat
        from lib.tree import K16Tree
        k16tree = K16Tree(tree)
        flat_tree = TreeFlat.from_tree(k16tree)
        if args.mmap_tree:
            flat_tree.save(flat_path, mmap_dir=True)
            print(f"✓ Structure plate sauvegardée vers {os.path.splitext(flat_path)[0]}/")
        else:
            flat_tree.save(flat_path)
            print(f"✓ Structure plate sauvegardée vers {flat_path}")

        total_time = time.time() - total_start_time
        print("\n✓ Construction de l'arbre optimisé terminée.")
        print(f"  - Configuration  : {args.config}")
        print(f"  - Vecteurs       : {args.vectors_file}")
        print(f"  - Structure plate: {flat_path}")
        print(f"  - Paramètres     : profondeur_max={args.max_depth}, {'k adaptatif' if args.k_adaptive else f'k={args.k}'}")
        print(f"                     max_leaf_size={args.max_leaf_size}, MAX_DATA={args.max_data}")
        print(f"                     max_workers={args.max_workers}, gpu={args.gpu}")
        print(f"  - Temps total    : {format_time(total_time)}")

        # Instructions pour l'utilisation du script de recherche
        print("\nPour tester la recherche dans cet arbre :")
        print(f"  python src/test.py {args.vectors_file} {flat_path} --k 100")
        print(f"  ou, en utilisant la configuration :")
        print(f"  python src/test.py --config {args.config}")
        
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