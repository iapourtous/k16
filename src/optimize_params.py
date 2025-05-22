#!/usr/bin/env python3
"""
Script d'optimisation automatique des paramètres K16.
Teste différentes combinaisons de paramètres pour trouver le meilleur compromis vitesse/recall.
Optimisé : construit un arbre une fois puis teste toutes les variantes de beam.
"""

import os
import sys
import json
import time
import itertools
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple

# Add the parent directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.config import ConfigManager
from lib.io import VectorReader, TreeIO
from lib.search import Searcher
from lib.clustering import build_tree

def test_tree_configuration(params: Dict, vectors_reader: VectorReader,
                           test_queries: np.ndarray, k: int = 10) -> List[Dict]:
    """
    Construit un arbre et teste toutes les configurations de beam sur cet arbre.
    """
    print(f"\n🌳 Construction avec : max_depth={params['max_depth']}, "
          f"max_leaf_size={params['max_leaf_size']}, max_data={params['max_data']}")

    # Construction de l'arbre
    print("  🔨 Construction de l'arbre...")
    build_start = time.time()

    tree = build_tree(
        vectors=vectors_reader.vectors,
        max_depth=params['max_depth'],
        k=16,  # Fixé pour ces tests
        k_adaptive=True,
        k_min=2,
        k_max=32,
        max_leaf_size=params['max_leaf_size'],
        max_data=params['max_data'],
        max_workers=12,
        use_gpu=True
    )

    build_time = time.time() - build_start
    print(f"  ✓ Arbre construit en {build_time:.2f}s")

    # Convertir en k16tree
    from lib.tree import K16Tree
    k16tree = K16Tree(tree)

    # Utiliser la structure plate optimisée
    use_flat_tree = params.get('use_flat_tree', True)
    if use_flat_tree:
        print("  🔨 Conversion en structure plate optimisée...")
        flat_start = time.time()
        from lib.flat_tree import TreeFlat
        flat_tree = TreeFlat.from_tree(k16tree)
        k16tree.flat_tree = flat_tree
        flat_time = time.time() - flat_start
        print(f"  ✓ Structure plate générée en {flat_time:.2f}s")

    results = []

    # Test en mode single
    print("  🔍 Test mode single...")
    searcher_single = Searcher(
        k16tree,
        vectors_reader,
        use_faiss=True,
        search_type="single",
        max_data=params['max_data']
    )

    single_results = searcher_single.evaluate_search(test_queries, k=k)

    results.append({
        'params': params.copy(),
        'build_time': build_time,
        'search_type': 'single',
        'use_flat_tree': use_flat_tree,
        'results': single_results
    })

    # Test différentes valeurs de beam_width sur le même arbre
    beam_widths = [2, 3, 4, 6, 8]

    for beam_width in beam_widths:
        print(f"  🔍 Test mode beam (width={beam_width})...")

        beam_params = params.copy()
        beam_params['beam_width'] = beam_width

        searcher_beam = Searcher(
            k16tree,
            vectors_reader,
            use_faiss=True,
            search_type="beam",
            beam_width=beam_width,
            max_data=params['max_data']
        )

        beam_results = searcher_beam.evaluate_search(test_queries, k=k)

        results.append({
            'params': beam_params,
            'build_time': build_time,  # Même temps de construction
            'search_type': 'beam',
            'use_flat_tree': use_flat_tree,
            'results': beam_results
        })

    return results

def optimize_parameters():
    """
    Teste différentes combinaisons de paramètres pour trouver l'optimal.
    """
    # Configuration de base
    config_manager = ConfigManager()
    files_config = config_manager.get_section("files")
    
    # Charger les vecteurs
    vectors_path = os.path.join(files_config["vectors_dir"], files_config["default_vectors"])
    print("📚 Chargement des vecteurs...")
    vectors_reader = VectorReader(vectors_path, mode="ram")
    
    # Générer des requêtes de test
    n_queries = 100
    query_indices = np.random.choice(len(vectors_reader), n_queries, replace=False)
    test_queries = vectors_reader[query_indices]
    
    # Paramètres d'arbre à tester
    tree_params = {
        'max_depth': [32],
        'max_leaf_size': [5, 10, 20, 30, 50],
        'max_data': [100, 200, 300, 400, 500],
        'use_flat_tree': [True]  # Structure plate activée par défaut pour tous les tests
    }
    
    # Générer toutes les combinaisons d'arbres
    param_names = list(tree_params.keys())
    param_values = list(tree_params.values())
    tree_combinations = list(itertools.product(*param_values))
    
    print(f"\n🔬 Test de {len(tree_combinations)} arbres différents...")
    print(f"   Chaque arbre sera testé avec single + 9 beam widths = {len(tree_combinations) * 10} tests totaux")
    
    all_results = []
    
    for i, combination in enumerate(tree_combinations):
        params = dict(zip(param_names, combination))
        
        print(f"\n📊 Arbre {i+1}/{len(tree_combinations)}")
        
        try:
            # Teste toutes les configurations de beam sur cet arbre
            tree_results = test_tree_configuration(params, vectors_reader, test_queries)
            all_results.extend(tree_results)
            
            # Sauvegarder les résultats au fur et à mesure
            with open('optimization_results.json', 'w') as f:
                json.dump(all_results, f, indent=2)
                
        except Exception as e:
            print(f"  ❌ Erreur : {str(e)}")
            continue
    
    # Analyser les résultats
    print("\n📊 Analyse des résultats...")
    analyze_results(all_results)
    
    return all_results

def analyze_results(results: List[Dict]):
    """
    Analyse les résultats et trouve les meilleures configurations.
    """
    # Séparer single et beam, avec structure plate
    single_results = [r for r in results if r['search_type'] == 'single']
    beam_results = [r for r in results if r['search_type'] == 'beam']
    flat_results = [r for r in results if r.get('use_flat_tree', False)]

    # Meilleure performance single avec structure plate
    if single_results:
        best_single_speed = min(single_results,
                               key=lambda x: x['results']['avg_total_time'])
        best_single_recall = max(single_results,
                                key=lambda x: x['results']['avg_recall'])

        flat_status = "structure plate" if best_single_speed.get('use_flat_tree', False) else "structure standard"
        print(f"\n🏆 Meilleur mode SINGLE ({flat_status}) :")
        print(f"  Vitesse maximale : {best_single_speed['params']}")
        print(f"    - Temps : {best_single_speed['results']['avg_total_time']*1000:.2f}ms")
        print(f"    - Recall : {best_single_speed['results']['avg_recall']:.4f}")

        flat_status = "structure plate" if best_single_recall.get('use_flat_tree', False) else "structure standard"
        print(f"  Recall maximal ({flat_status}) : {best_single_recall['params']}")
        print(f"    - Temps : {best_single_recall['results']['avg_total_time']*1000:.2f}ms")
        print(f"    - Recall : {best_single_recall['results']['avg_recall']:.4f}")

    # Meilleure performance beam avec structure plate
    if beam_results:
        best_beam_speed = min(beam_results,
                             key=lambda x: x['results']['avg_total_time'])
        best_beam_recall = max(beam_results,
                              key=lambda x: x['results']['avg_recall'])

        flat_status = "structure plate" if best_beam_speed.get('use_flat_tree', False) else "structure standard"
        print(f"\n🏆 Meilleur mode BEAM ({flat_status}) :")
        print(f"  Vitesse maximale : {best_beam_speed['params']}")
        print(f"    - Temps : {best_beam_speed['results']['avg_total_time']*1000:.2f}ms")
        print(f"    - Recall : {best_beam_speed['results']['avg_recall']:.4f}")

        flat_status = "structure plate" if best_beam_recall.get('use_flat_tree', False) else "structure standard"
        print(f"  Recall maximal ({flat_status}) : {best_beam_recall['params']}")
        print(f"    - Temps : {best_beam_recall['results']['avg_total_time']*1000:.2f}ms")
        print(f"    - Recall : {best_beam_recall['results']['avg_recall']:.4f}")

    # Si nous avons des résultats avec la structure plate
    if flat_results:
        best_flat_speed = min(flat_results,
                             key=lambda x: x['results']['avg_total_time'])
        best_flat_recall = max(flat_results,
                              key=lambda x: x['results']['avg_recall'])

        print("\n🏆 Meilleur avec STRUCTURE PLATE :")
        print(f"  Vitesse maximale : {best_flat_speed['params']}")
        print(f"    - Mode : {best_flat_speed['search_type']}")
        print(f"    - Temps : {best_flat_speed['results']['avg_total_time']*1000:.2f}ms")
        print(f"    - Recall : {best_flat_speed['results']['avg_recall']:.4f}")
        print(f"  Recall maximal : {best_flat_recall['params']}")
        print(f"    - Mode : {best_flat_recall['search_type']}")
        print(f"    - Temps : {best_flat_recall['results']['avg_total_time']*1000:.2f}ms")
        print(f"    - Recall : {best_flat_recall['results']['avg_recall']:.4f}")

    # Meilleur compromis (score = recall / temps)
    all_configs = []
    for r in results:
        score = r['results']['avg_recall'] / r['results']['avg_total_time']
        all_configs.append((
            r['search_type'],
            r['params'],
            score,
            r['results']['avg_recall'],
            r['results']['avg_total_time'],
            r.get('use_flat_tree', False)
        ))

    best_compromise = max(all_configs, key=lambda x: x[2])
    flat_status = "structure plate" if best_compromise[5] else "structure standard"
    print(f"\n🏆 Meilleur compromis (recall/temps) - {flat_status} :")
    print(f"  Mode : {best_compromise[0]}")
    print(f"  Params : {best_compromise[1]}")
    print(f"  Recall : {best_compromise[3]:.4f}")
    print(f"  Temps : {best_compromise[4]*1000:.2f}ms")
    print(f"  Score : {best_compromise[2]:.2f}")

    # Repérer les configurations atteignant un recall de 1.0 (100%)
    perfect_recall_configs = [r for r in results if r['results']['avg_recall'] >= 0.999]
    if perfect_recall_configs:
        fastest_perfect = min(perfect_recall_configs, key=lambda x: x['results']['avg_total_time'])
        flat_status = "structure plate" if fastest_perfect.get('use_flat_tree', False) else "structure standard"
        print(f"\n🏆 Configuration la plus rapide avec recall parfait (>= 0.999) - {flat_status} :")
        print(f"  Mode : {fastest_perfect['search_type']}")
        print(f"  Params : {fastest_perfect['params']}")
        print(f"  Recall : {fastest_perfect['results']['avg_recall']:.4f}")
        print(f"  Temps : {fastest_perfect['results']['avg_total_time']*1000:.2f}ms")
    
    # Créer un graphique des résultats
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(14, 10))
        
        # Couleurs par max_data
        colors = {100: 'blue', 200: 'green', 300: 'cyan', 400: 'red', 500: 'orange', 1000: 'purple', 2000: 'brown'}

        # Deux sous-graphiques
        plt.subplot(2, 1, 1)

        # Graphique principal : vitesse vs recall
        for r in results:
            color = colors.get(r['params']['max_data'], 'black')

            # Différencier structure plate et standard
            use_flat = r.get('use_flat_tree', False)
            if r['search_type'] == 'single':
                marker = 'o' if use_flat else '^'
            else:
                marker = 's' if use_flat else 'd'

            label = f"{r['search_type']}, max_data={r['params']['max_data']}"
            if r['search_type'] == 'beam':
                label += f", width={r['params'].get('beam_width', '?')}"
            label += f", flat={use_flat}"

            plt.scatter(r['results']['avg_total_time']*1000,
                       r['results']['avg_recall'],
                       c=color,
                       marker=marker,
                       s=100,
                       alpha=0.7,
                       label=label)
        
        plt.xlabel('Temps (ms)')
        plt.ylabel('Recall')
        plt.title('K16 : Compromis Vitesse vs Recall')
        plt.grid(True, alpha=0.3)
        
        # Marquer les meilleurs points
        plt.scatter(best_single_speed['results']['avg_total_time']*1000,
                   best_single_speed['results']['avg_recall'],
                   marker='*', s=500, c='red', edgecolors='black',
                   label='Meilleur vitesse (single)')
        
        plt.scatter(best_beam_recall['results']['avg_total_time']*1000,
                   best_beam_recall['results']['avg_recall'],
                   marker='*', s=500, c='green', edgecolors='black',
                   label='Meilleur recall (beam)')
        
        plt.scatter(best_compromise[4]*1000,
                   best_compromise[3],
                   marker='*', s=500, c='gold', edgecolors='black',
                   label='Meilleur compromis')
        
        # Graphique secondaire : impact du beam_width
        plt.subplot(2, 1, 2)
        
        # Grouper par configuration d'arbre
        beam_by_config = {}
        for r in beam_results:
            tree_key = (r['params']['max_depth'], 
                       r['params']['max_leaf_size'], 
                       r['params']['max_data'])
            
            if tree_key not in beam_by_config:
                beam_by_config[tree_key] = []
            
            beam_by_config[tree_key].append((
                r['params']['beam_width'],
                r['results']['avg_recall'],
                r['results']['avg_total_time']
            ))
        
        # Tracer l'impact du beam_width pour chaque config
        for tree_key, beam_data in beam_by_config.items():
            beam_data.sort(key=lambda x: x[0])  # Trier par beam_width
            widths = [x[0] for x in beam_data]
            recalls = [x[1] for x in beam_data]
            
            label = f"depth={tree_key[0]}, leaf={tree_key[1]}, data={tree_key[2]}"
            plt.plot(widths, recalls, 'o-', alpha=0.7, label=label)
        
        plt.xlabel('Beam Width')
        plt.ylabel('Recall')
        plt.title('Impact du Beam Width sur le Recall')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('optimization_results.png', dpi=300, bbox_inches='tight')
        print("\n📈 Graphique sauvegardé : optimization_results.png")
        
    except ImportError:
        print("\n⚠️  matplotlib non installé, pas de graphique généré")

if __name__ == "__main__":
    print("🚀 K16 - Optimisation automatique des paramètres")
    print("=" * 50)
    
    results = optimize_parameters()
    
    print("\n✅ Optimisation terminée!")
    print("Résultats sauvegardés dans : optimization_results.json")