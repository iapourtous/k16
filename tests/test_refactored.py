#!/usr/bin/env python3
"""
Script de test pour vérifier que la version refactorisée fonctionne correctement.
"""

import os
import sys
import time
import numpy as np

# Importer les modules refactorisés
from k16.utils.config import ConfigManager
from k16.utils.optimization import configure_simd, check_simd_support, check_numba_support
from k16.core.tree import K16Tree
from k16.core.flat_tree import TreeFlat
from k16.io.reader import read_vectors, load_tree
from k16.io.writer import write_vectors, write_tree
from k16.search.searcher import search
from k16.builder.builder import build_optimized_tree

def test_optimization():
    """Teste les fonctions d'optimisation."""
    print("\n--- Test des optimisations ---")
    configure_simd()
    check_simd_support()
    check_numba_support()
    print("✓ Test des optimisations OK")

def test_config():
    """Teste la gestion de la configuration."""
    print("\n--- Test de la configuration ---")
    config_manager = ConfigManager()
    build_config = config_manager.get_section("build_tree")
    print(f"✓ Configuration chargée: max_depth={build_config['max_depth']}, k={build_config['k']}")
    print("✓ Test de la configuration OK")

def test_vector_io():
    """Teste la lecture/écriture de vecteurs."""
    print("\n--- Test de la lecture/écriture de vecteurs ---")
    
    # Créer des vecteurs de test
    n_vectors = 1000
    dims = 128
    vectors = np.random.randn(n_vectors, dims).astype(np.float32)
    vectors = vectors / np.linalg.norm(vectors, axis=1)[:, np.newaxis]
    
    # Écrire les vecteurs
    test_file = "test_vectors.bin"
    write_vectors(vectors, test_file)
    
    # Lire les vecteurs
    vectors_reader = read_vectors(test_file)
    
    # Vérifier les dimensions
    assert len(vectors_reader) == n_vectors
    assert vectors_reader.d == dims
    
    # Vérifier quelques vecteurs
    for i in range(0, n_vectors, 100):
        np.testing.assert_allclose(vectors[i], vectors_reader[i], rtol=1e-5)
    
    print(f"✓ Test de la lecture/écriture de vecteurs OK ({n_vectors} vecteurs, {dims} dimensions)")
    
    # Nettoyer
    os.remove(test_file)

def test_tree_build_and_search():
    """Teste la construction d'arbre et la recherche."""
    print("\n--- Test de la construction d'arbre et de la recherche ---")
    
    # Créer des vecteurs de test
    n_vectors = 1000
    dims = 128
    vectors = np.random.randn(n_vectors, dims).astype(np.float32)
    vectors = vectors / np.linalg.norm(vectors, axis=1)[:, np.newaxis]
    
    # Paramètres
    max_depth = 3
    k = 4
    max_leaf_size = 50
    max_data = 100
    
    # Construire l'arbre
    start_time = time.time()
    tree = build_optimized_tree(
        vectors=vectors,
        output_file="",  # Ne pas sauvegarder
        max_depth=max_depth,
        k=k,
        max_leaf_size=max_leaf_size,
        max_data=max_data,
        use_hnsw=False,  # Désactiver HNSW pour le test rapide
        verbose=True
    )
    build_time = time.time() - start_time
    
    print(f"✓ Arbre construit en {build_time:.2f}s")
    
    # Créer un K16Tree avec l'arbre plat
    k16tree = K16Tree(None)
    k16tree.flat_tree = tree
    
    # Créer un lecteur de vecteurs
    vectors_reader = read_vectors(vectors=vectors)
    
    # Effectuer une recherche
    query = vectors[0]  # Utiliser le premier vecteur comme requête
    start_time = time.time()
    results, scores = search(
        tree=k16tree,
        vectors_reader=vectors_reader,
        query=query,
        k=10,
        search_type="beam",
        beam_width=3
    )
    search_time = time.time() - start_time
    
    print(f"✓ Recherche effectuée en {search_time*1000:.2f}ms")
    print(f"✓ Nombre de résultats: {len(results)}")
    print(f"✓ Premier résultat: {results[0]} (score: {scores[0]:.4f})")
    
    # Vérifier que le premier résultat est bien le vecteur lui-même
    assert results[0] == 0, "Le premier résultat devrait être le vecteur de requête lui-même"
    
    print("✓ Test de la construction d'arbre et de la recherche OK")

def main():
    """Fonction principale pour exécuter les tests."""
    print("=== Tests de la version refactorisée de K16 ===")
    
    try:
        test_optimization()
        test_config()
        test_vector_io()
        test_tree_build_and_search()
        
        print("\n✅ TOUS LES TESTS ONT RÉUSSI")
        return 0
    except Exception as e:
        print(f"\n❌ ERREUR: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())