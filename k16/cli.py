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