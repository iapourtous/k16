#!/usr/bin/env python3
"""
Analyse détaillée d'un arbre K16.
Affiche des statistiques et métriques sur la structure de l'arbre.

Usage:
    python analyze_tree.py tree.bin
    python analyze_tree.py --config config.yaml
"""

import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import argparse
import yaml
import time
from collections import defaultdict, Counter
try:
    import seaborn as sns
except ImportError:
    print("⚠️ La bibliothèque seaborn n'est pas installée. Les visualisations seront simplifiées.")
    sns = None

from tqdm.auto import tqdm

# Définition de la classe TreeNode pour assurer la compatibilité lors du chargement
class TreeNode:
    """Noeud dans l'arbre K16 optimisé."""
    def __init__(self, centroid, level=0):
        self.centroid = centroid  # Centroïde du nœud
        self.level = level        # Niveau dans l'arbre
        self.children = []        # Pour les nœuds internes: liste des noeuds enfants
        self.indices = []         # Pour les feuilles: liste pré-calculée des MAX_DATA indices les plus proches

# Chemin par défaut du fichier de configuration
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")

# Chargement de la configuration
def load_config(config_path=None):
    """Charge la configuration depuis le fichier YAML."""
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"⚠️ Erreur lors du chargement de la configuration: {str(e)}")
        print(f"⚠️ Utilisation des paramètres par défaut")
        return {
            "files": {
                "trees_dir": ".",
                "default_tree": "tree.bin"
            }
        }

# Chargement de la configuration
CONFIG = load_config()

# Définir les chemins par défaut à partir de la configuration
default_tree_path = os.path.join(CONFIG["files"]["trees_dir"], CONFIG["files"]["default_tree"])

def load_tree(path):
    """Charge l'arbre depuis un fichier binaire."""
    start_time = time.time()
    print(f"⏳ Chargement de l'arbre depuis {path}...")

    try:
        with open(path, "rb") as f:
            tree = pickle.load(f)

        elapsed = time.time() - start_time
        print(f"✓ Arbre chargé depuis {path} en {elapsed:.2f}s")
        return tree
    except Exception as e:
        print(f"❌ Erreur lors du chargement de l'arbre: {str(e)}")
        print(f"Essayez de créer l'arbre d'abord avec:")
        print(f"  python src/build_tree.py --config config.yaml")
        sys.exit(1)

def analyze_tree(tree):
    """
    Analyse détaillée de l'arbre K16.
    Collecte et calcule diverses métriques sur la structure de l'arbre.
    
    Args:
        tree: L'arbre à analyser
        
    Returns:
        Un dictionnaire de métriques et statistiques
    """
    print("⏳ Analyse de l'arbre en cours...")
    start_time = time.time()
    
    stats = {
        # Métriques globales
        "total_nodes": 0,
        "internal_nodes": 0,
        "leaf_nodes": 0,
        "max_depth": 0,
        "avg_depth": 0,
        "min_leaf_depth": float('inf'),
        
        # Statistiques de branches
        "branch_counts": [],  # Nombre de branches par niveau
        "avg_branches": 0,    # Branches moyennes par nœud interne
        
        # Statistiques de feuilles
        "leaf_sizes": [],     # Nombre d'indices par feuille
        "leaf_depths": [],    # Profondeur des feuilles
        "avg_leaf_size": 0,
        "min_leaf_size": float('inf'),
        "max_leaf_size": 0,
        "total_indices": 0,
        
        # Distribution des branches
        "branching_factors": defaultdict(int),  # Distribution des facteurs de branchement
        
        # Métriques d'équilibre
        "balance_factor": 0,  # Ratio profondeur min/max
        "depth_distribution": defaultdict(int),  # Distribution des profondeurs des feuilles
    }
    
    # Parcourir l'arbre pour collecter les statistiques
    def traverse(node, depth=0):
        nonlocal stats
        
        stats["total_nodes"] += 1
        
        # Mettre à jour la profondeur maximale
        stats["max_depth"] = max(stats["max_depth"], depth)
        
        if not node.children:  # C'est une feuille
            stats["leaf_nodes"] += 1
            stats["leaf_depths"].append(depth)
            stats["min_leaf_depth"] = min(stats["min_leaf_depth"], depth)
            stats["depth_distribution"][depth] += 1
            
            # Statistiques sur les indices dans les feuilles
            leaf_size = len(node.indices)
            stats["leaf_sizes"].append(leaf_size)
            stats["min_leaf_size"] = min(stats["min_leaf_size"], leaf_size)
            stats["max_leaf_size"] = max(stats["max_leaf_size"], leaf_size)
            stats["total_indices"] += leaf_size
        else:  # C'est un nœud interne
            stats["internal_nodes"] += 1
            
            # Facteur de branchement
            branch_count = len(node.children)
            stats["branching_factors"][branch_count] += 1
            
            # Stocker le nombre de branches par niveau
            if depth >= len(stats["branch_counts"]):
                stats["branch_counts"].append(branch_count)
            else:
                stats["branch_counts"][depth] += branch_count
            
            # Continuer avec les enfants
            for child in node.children:
                traverse(child, depth + 1)
    
    # Parcourir l'arbre récursivement
    traverse(tree)
    
    # Calculer les moyennes et autres métriques dérivées
    if stats["leaf_nodes"] > 0:
        stats["avg_leaf_size"] = sum(stats["leaf_sizes"]) / stats["leaf_nodes"]
        stats["avg_depth"] = sum(stats["leaf_depths"]) / stats["leaf_nodes"]
        
        # Facteur d'équilibre
        if stats["max_depth"] > 0:
            stats["balance_factor"] = stats["min_leaf_depth"] / stats["max_depth"]
    
    if stats["internal_nodes"] > 0:
        total_branches = sum(stats["branching_factors"][k] * k for k in stats["branching_factors"])
        stats["avg_branches"] = total_branches / stats["internal_nodes"]
    
    elapsed = time.time() - start_time
    print(f"✓ Analyse terminée en {elapsed:.2f}s")
    return stats

def generate_visualizations(stats, output_dir="."):
    """
    Génère des visualisations des statistiques de l'arbre.
    
    Args:
        stats: Les statistiques de l'arbre
        output_dir: Le répertoire de sortie pour les graphiques
    """
    print("⏳ Génération des visualisations...")
    
    # Configurer le style des graphiques
    sns.set(style="whitegrid")
    
    # 1. Distribution des profondeurs des feuilles
    plt.figure(figsize=(10, 6))
    depths = sorted(stats["depth_distribution"].keys())
    counts = [stats["depth_distribution"][d] for d in depths]
    
    plt.bar(depths, counts, color='skyblue')
    plt.title('Distribution des profondeurs des feuilles')
    plt.xlabel('Profondeur')
    plt.ylabel('Nombre de feuilles')
    plt.xticks(depths)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.savefig(os.path.join(output_dir, "leaf_depth_distribution.png"), dpi=300, bbox_inches='tight')
    
    # 2. Distribution des tailles des feuilles
    plt.figure(figsize=(10, 6))
    
    # Utiliser une échelle logarithmique si nécessaire
    use_log = max(stats["leaf_sizes"]) / min(stats["leaf_sizes"] if min(stats["leaf_sizes"]) > 0 else 1) > 100
    
    if use_log:
        plt.hist(stats["leaf_sizes"], bins=30, color='lightgreen', log=True)
        plt.title('Distribution des tailles des feuilles (échelle log)')
    else:
        plt.hist(stats["leaf_sizes"], bins=30, color='lightgreen')
        plt.title('Distribution des tailles des feuilles')
    
    plt.xlabel('Nombre d\'indices par feuille')
    plt.ylabel('Nombre de feuilles')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.savefig(os.path.join(output_dir, "leaf_size_distribution.png"), dpi=300, bbox_inches='tight')
    
    # 3. Distribution des facteurs de branchement
    plt.figure(figsize=(10, 6))
    factors = sorted(stats["branching_factors"].keys())
    counts = [stats["branching_factors"][f] for f in factors]
    
    plt.bar(factors, counts, color='salmon')
    plt.title('Distribution des facteurs de branchement')
    plt.xlabel('Nombre de branches par nœud')
    plt.ylabel('Nombre de nœuds')
    plt.xticks(factors)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.savefig(os.path.join(output_dir, "branching_factors.png"), dpi=300, bbox_inches='tight')
    
    # 4. Graphique en radar des caractéristiques de l'arbre
    if stats["avg_branches"] > 0 and stats["max_depth"] > 0 and stats["max_leaf_size"] > 0:
        plt.figure(figsize=(10, 10))
        
        # Normaliser les valeurs pour le graphique en radar
        max_norm = max(stats["max_depth"], stats["avg_branches"], stats["avg_leaf_size"])
        
        # Caractéristiques pour le radar
        categories = ['Profondeur max', 'Branches moyennes', 'Taille feuille moyenne', 
                      'Équilibre', 'Cohérence branches']
        
        # Valeurs normalisées
        values = [
            stats["max_depth"] / max_norm,
            stats["avg_branches"] / max_norm,
            stats["avg_leaf_size"] / stats["max_leaf_size"],
            stats["balance_factor"],
            1 - (max(stats["branching_factors"].values()) - min(stats["branching_factors"].values())) / max(stats["branching_factors"].values()) if len(stats["branching_factors"]) > 0 else 0
        ]
        
        # Créer le graphique en radar
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
        values += values[:1]  # Fermer le polygone
        angles += angles[:1]  # Fermer le polygone
        
        ax = plt.subplot(111, polar=True)
        ax.plot(angles, values, linewidth=2, linestyle='solid')
        ax.fill(angles, values, alpha=0.25)
        ax.set_thetagrids(np.degrees(angles[:-1]), categories)
        ax.set_ylim(0, 1)
        plt.title('Caractéristiques structurelles de l\'arbre')
        
        plt.savefig(os.path.join(output_dir, "tree_characteristics_radar.png"), dpi=300, bbox_inches='tight')
    
    print(f"✓ Visualisations générées dans le répertoire {output_dir}")

def print_stats(stats):
    """
    Affiche les statistiques de l'arbre de manière formatée.
    
    Args:
        stats: Les statistiques de l'arbre
    """
    width = 50  # Largeur du tableau ASCII
    
    print("\n" + "="*width)
    print(" "*15 + "STATISTIQUES DE L'ARBRE K16")
    print("="*width)
    
    # Statistiques générales
    print("\n--- STRUCTURE GÉNÉRALE " + "-"*(width-20))
    print(f"Nombre total de nœuds : {stats['total_nodes']:,}")
    print(f"Nœuds internes        : {stats['internal_nodes']:,}")
    print(f"Feuilles              : {stats['leaf_nodes']:,}")
    print(f"Profondeur maximale   : {stats['max_depth']}")
    print(f"Profondeur moyenne    : {stats['avg_depth']:.2f}")
    print(f"Profondeur min feuille: {stats['min_leaf_depth']}")
    print(f"Facteur d'équilibre   : {stats['balance_factor']:.4f}")
    
    # Statistiques de branchement
    print("\n--- STATISTIQUES DE BRANCHEMENT " + "-"*(width-30))
    print(f"Branches moyennes par nœud : {stats['avg_branches']:.2f}")
    
    # Distribution des facteurs de branchement
    print("\nDistribution des facteurs de branchement :")
    factors = sorted(stats["branching_factors"].keys())
    for factor in factors:
        count = stats["branching_factors"][factor]
        percent = count / stats["internal_nodes"] * 100 if stats["internal_nodes"] > 0 else 0
        print(f"  {factor} branches : {count:,} nœuds ({percent:.1f}%)")
    
    # Statistiques des feuilles
    print("\n--- STATISTIQUES DES FEUILLES " + "-"*(width-30))
    print(f"Nombre total d'indices    : {stats['total_indices']:,}")
    print(f"Taille moyenne des feuilles: {stats['avg_leaf_size']:.2f}")
    print(f"Taille minimale           : {stats['min_leaf_size']}")
    print(f"Taille maximale           : {stats['max_leaf_size']}")
    
    # Distribution des profondeurs des feuilles
    print("\nDistribution des profondeurs des feuilles :")
    depths = sorted(stats["depth_distribution"].keys())
    for depth in depths:
        count = stats["depth_distribution"][depth]
        percent = count / stats["leaf_nodes"] * 100 if stats["leaf_nodes"] > 0 else 0
        print(f"  Profondeur {depth} : {count:,} feuilles ({percent:.1f}%)")
    
    # Histogramme ASCII des tailles des feuilles
    if stats["leaf_sizes"]:
        print("\nHistogramme des tailles des feuilles :")
        # Diviser en 5 groupes
        min_size = min(stats["leaf_sizes"])
        max_size = max(stats["leaf_sizes"])
        step = (max_size - min_size) / 5 if max_size > min_size else 1
        
        for i in range(5):
            lower = min_size + i * step
            upper = min_size + (i + 1) * step
            count = sum(1 for size in stats["leaf_sizes"] if lower <= size < upper)
            percent = count / len(stats["leaf_sizes"]) * 100
            bar = "#" * int(percent / 2)
            print(f"  {lower:.0f}-{upper:.0f} : {bar} {count:,} ({percent:.1f}%)")
    
    print("\n" + "="*width)

def main():
    parser = argparse.ArgumentParser(description="Analyse détaillée d'un arbre K16")
    parser.add_argument("tree_file", nargs="?", default=default_tree_path,
                      help=f"Fichier contenant l'arbre K16 (par défaut: {default_tree_path})")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH,
                      help=f"Chemin vers le fichier de configuration (par défaut: {DEFAULT_CONFIG_PATH})")
    parser.add_argument("--output-dir", default=".",
                      help="Répertoire de sortie pour les visualisations (par défaut: .)")
    parser.add_argument("--no-plots", action="store_true",
                      help="Ne pas générer de visualisations")
    
    args = parser.parse_args()
    
    # Recharger la configuration si un fichier spécifique est fourni
    if args.config != DEFAULT_CONFIG_PATH:
        global CONFIG
        CONFIG = load_config(args.config)
        print(f"✓ Configuration chargée depuis: {args.config}")
    
    # Charger l'arbre
    tree = load_tree(args.tree_file)
    
    # Analyser l'arbre
    stats = analyze_tree(tree)
    
    # Afficher les statistiques
    print_stats(stats)
    
    # Générer des visualisations
    if not args.no_plots:
        try:
            os.makedirs(args.output_dir, exist_ok=True)
            generate_visualizations(stats, args.output_dir)
        except Exception as e:
            print(f"⚠️ Erreur lors de la génération des visualisations: {str(e)}")
    
    print("\n✓ Analyse terminée.")

if __name__ == "__main__":
    main()