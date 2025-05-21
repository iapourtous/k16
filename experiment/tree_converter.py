#!/usr/bin/env python3
"""
Convertisseur d'arbre hiérarchique vers arbre plat.
Utilitaire pour convertir un arbre K16 existant et le sauvegarder au format plat.
"""

import os
import sys
import time
import argparse
import numpy as np

# Ajouter le répertoire parent au chemin
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importer les modules K16
from lib.config import ConfigManager
from lib.io import TreeIO
from flat_tree import TreeFlat

def convert_tree(input_path: str, output_path: str, add_binary: bool = True, n_bits: int = 256):
    """
    Convertit un arbre K16 standard en arbre plat optimisé.
    
    Args:
        input_path: Chemin du fichier d'arbre K16 à convertir
        output_path: Chemin où sauvegarder l'arbre plat
        add_binary: Ajouter des représentations binaires des centroïdes
        n_bits: Nombre de bits pour la représentation binaire
    """
    print(f"🔄 Conversion de l'arbre K16 en structure plate optimisée")
    print(f"  - Source: {input_path}")
    print(f"  - Destination: {output_path}")
    print(f"  - Représentation binaire: {'Activée' if add_binary else 'Désactivée'}")
    if add_binary:
        print(f"  - Bits pour la quantification: {n_bits}")
    
    # Charger l'arbre source
    print(f"⏳ Chargement de l'arbre source...")
    tree_io = TreeIO()
    start_time = time.time()

    tree = tree_io.load_as_k16tree(input_path)
    
    load_time = time.time() - start_time
    print(f"✓ Arbre chargé en {load_time:.2f}s")
    
    # Afficher quelques statistiques sur l'arbre source
    stats = tree.get_statistics()
    print(f"  → Nœuds: {stats['node_count']:,}")
    print(f"  → Feuilles: {stats['leaf_count']:,}")
    print(f"  → Profondeur: {stats['max_depth']}")
    
    # Convertir en structure plate
    print(f"⏳ Conversion en structure plate...")
    start_time = time.time()
    
    flat_tree = TreeFlat.from_tree(tree)
    
    conversion_time = time.time() - start_time
    print(f"✓ Conversion terminée en {conversion_time:.2f}s")
    
    # Ajouter les représentations binaires si demandé
    if add_binary:
        print(f"⏳ Génération des représentations binaires ({n_bits} bits)...")
        start_time = time.time()
        
        flat_tree.add_binary_centroids(n_bits)
        
        binary_time = time.time() - start_time
        print(f"✓ Représentations binaires générées en {binary_time:.2f}s")
    
    # Sauvegarder l'arbre plat
    print(f"⏳ Sauvegarde de l'arbre plat...")
    start_time = time.time()
    
    # Créer le répertoire de destination si nécessaire
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Sauvegarder
    flat_tree.save(output_path)
    
    save_time = time.time() - start_time
    print(f"✓ Arbre plat sauvegardé en {save_time:.2f}s")
    print(f"  → Chemin: {output_path}")
    
    # Afficher quelques statistiques sur l'arbre plat
    flat_stats = flat_tree.get_statistics()
    print(f"\n📊 Statistiques de l'arbre plat:")
    print(f"  - Nœuds: {flat_stats['n_nodes']:,}")
    print(f"  - Feuilles: {flat_stats['n_leaves']:,}")
    print(f"  - Profondeur: {flat_stats['max_depth']}")
    print(f"  - Représentation binaire: {flat_stats['binary_representation']}")
    if flat_stats['binary_representation']:
        print(f"  - Bits: {flat_stats['binary_bits']}")
    
    total_time = load_time + conversion_time + (binary_time if add_binary else 0) + save_time
    print(f"\n✅ Conversion terminée en {total_time:.2f}s total")

def main():
    parser = argparse.ArgumentParser(description="Convertisseur d'arbre K16 en structure plate optimisée")
    parser.add_argument("--input", help="Chemin du fichier d'arbre K16 à convertir")
    parser.add_argument("--output", help="Chemin où sauvegarder l'arbre plat")
    parser.add_argument("--no-binary", action="store_true", help="Ne pas générer de représentations binaires")
    parser.add_argument("--bits", type=int, default=256, help="Nombre de bits pour la représentation binaire (défaut: 256)")
    
    args = parser.parse_args()
    
    # Charger la configuration
    config = ConfigManager()
    files_config = config.get_section("files")
    
    # Déterminer les chemins de fichiers
    input_path = args.input or os.path.join(files_config["trees_dir"], files_config["default_tree"])
    
    if args.output:
        output_path = args.output
    else:
        # Générer un nom basé sur l'entrée
        input_name = os.path.basename(input_path)
        output_name = f"flat_{input_name.split('.')[0]}.flat"
        output_path = os.path.join(files_config["trees_dir"], output_name)
    
    # Exécuter la conversion
    convert_tree(
        input_path=input_path,
        output_path=output_path,
        add_binary=not args.no_binary,
        n_bits=args.bits
    )

if __name__ == "__main__":
    main()