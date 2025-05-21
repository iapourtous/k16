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
    
    results = []
    
    # Test en mode single
    print("  🔍 Test mode single...")
    searcher_single = Searcher(
        tree, 
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
        'results': single_results
    })
    
    # Test différentes valeurs de beam_width sur le même arbre
    beam_widths = [2,4,8,16,18]
    
    for beam_width in beam_widths:
        print(f"  🔍 Test mode beam (width={beam_width})...")
        
        beam_params = params.copy()
        beam_params['beam_width'] = beam_width
        
        searcher_beam = Searcher(
            tree, 
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
        'max_leaf_size': [5,10, 20, 30, 50],
        'max_data': [100,200,300,400,500]
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
    # Séparer single et beam
    single_results = [r for r in results if r['search_type'] == 'single']
    beam_results = [r for r in results if r['search_type'] == 'beam']
    
    # Meilleure performance single
    if single_results:
        best_single_speed = min(single_results, 
                               key=lambda x: x['results']['avg_total_time'])
        best_single_recall = max(single_results, 
                                key=lambda x: x['results']['avg_recall'])
        
        print("\n🏆 Meilleur mode SINGLE :")
        print(f"  Vitesse maximale : {best_single_speed['params']}")
        print(f"    - Temps : {best_single_speed['results']['avg_total_time']*1000:.2f}ms")
        print(f"    - Recall : {best_single_speed['results']['avg_recall']:.4f}")
        print(f"  Recall maximal : {best_single_recall['params']}")
        print(f"    - Temps : {best_single_recall['results']['avg_total_time']*1000:.2f}ms")
        print(f"    - Recall : {best_single_recall['results']['avg_recall']:.4f}")
    
    # Meilleure performance beam
    if beam_results:
        best_beam_speed = min(beam_results, 
                             key=lambda x: x['results']['avg_total_time'])
        best_beam_recall = max(beam_results, 
                              key=lambda x: x['results']['avg_recall'])
        
        print("\n🏆 Meilleur mode BEAM :")
        print(f"  Vitesse maximale : {best_beam_speed['params']}")
        print(f"    - Temps : {best_beam_speed['results']['avg_total_time']*1000:.2f}ms")
        print(f"    - Recall : {best_beam_speed['results']['avg_recall']:.4f}")
        print(f"  Recall maximal : {best_beam_recall['params']}")
        print(f"    - Temps : {best_beam_recall['results']['avg_total_time']*1000:.2f}ms")
        print(f"    - Recall : {best_beam_recall['results']['avg_recall']:.4f}")
    
    # Meilleur compromis (score = recall / temps)
    all_configs = []
    for r in results:
        score = r['results']['avg_recall'] / r['results']['avg_total_time']
        all_configs.append((
            r['search_type'], 
            r['params'], 
            score, 
            r['results']['avg_recall'], 
            r['results']['avg_total_time']
        ))
    
    best_compromise = max(all_configs, key=lambda x: x[2])
    print(f"\n🏆 Meilleur compromis (recall/temps) :")
    print(f"  Mode : {best_compromise[0]}")
    print(f"  Params : {best_compromise[1]}")
    print(f"  Recall : {best_compromise[3]:.4f}")
    print(f"  Temps : {best_compromise[4]*1000:.2f}ms")
    print(f"  Score : {best_compromise[2]:.2f}")
    
    # Créer un graphique des résultats
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(14, 10))
        
        # Couleurs par max_data
        colors = {200: 'blue', 500: 'green', 1000: 'red', 2000: 'orange'}
        
        # Deux sous-graphiques
        plt.subplot(2, 1, 1)
        
        # Graphique principal : vitesse vs recall
        for r in results:
            color = colors.get(r['params']['max_data'], 'black')
            marker = 'o' if r['search_type'] == 'single' else 's'
            
            label = f"{r['search_type']}, max_data={r['params']['max_data']}"
            if r['search_type'] == 'beam':
                label += f", width={r['params']['beam_width']}"
            
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