#!/usr/bin/env python3
"""
Script de test pour la fonction de recherche interactive.
Ce script simule un environnement interactif pour tester la commande search.
"""

import os
import sys
import time
from sentence_transformers import SentenceTransformer

# Ajouter le répertoire principal au chemin d'importation
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from k16.utils.config import ConfigManager
from k16.io.reader import read_vectors, load_tree
from k16.search.searcher import Searcher

def main():
    """Fonction principale pour tester la recherche interactive."""
    # Initialisation du gestionnaire de configuration
    config_manager = ConfigManager()
    
    # Récupération des chemins par défaut
    files_config = config_manager.get_section("files")
    prepare_config = config_manager.get_section("prepare_data")
    search_config = config_manager.get_section("search")
    
    vectors_path = os.path.join(files_config["vectors_dir"], files_config["default_vectors"])
    tree_path = os.path.join(files_config["trees_dir"], files_config["default_tree"])
    qa_path = os.path.join(files_config["vectors_dir"], files_config.get("default_qa", "qa.txt"))
    
    # Vérifier que les fichiers existent
    if not os.path.exists(vectors_path):
        print(f"❌ Fichier de vecteurs introuvable: {vectors_path}")
        return 1
    
    if not os.path.exists(tree_path) and not os.path.exists(tree_path.replace(".bsp", ".flat.npy")):
        print(f"❌ Fichier d'arbre introuvable: {tree_path}")
        return 1
    
    if not os.path.exists(qa_path):
        print(f"❌ Fichier QA introuvable: {qa_path}")
        return 1
    
    try:
        # Charger les vecteurs
        print(f"⏳ Chargement des vecteurs depuis {vectors_path}...")
        vectors_reader = read_vectors(
            file_path=vectors_path,
            mode="ram",
            cache_size_mb=search_config.get("cache_size_mb", 500)
        )
        print(f"✓ Vecteurs chargés: {len(vectors_reader):,} vecteurs de dimension {vectors_reader.d}")
        
        # Charger l'arbre
        print(f"⏳ Chargement de l'arbre depuis {tree_path}...")
        tree = load_tree(tree_path, mmap_tree=False)
        print(f"✓ Arbre chargé")
        
        # Charger les questions et réponses
        print(f"⏳ Chargement des questions et réponses depuis {qa_path}...")
        with open(qa_path, "r", encoding="utf-8") as f:
            qa_lines = f.readlines()
        print(f"✓ {len(qa_lines):,} questions-réponses chargées")
        
        # Créer le chercheur
        searcher = Searcher(
            k16tree=tree,
            vectors_reader=vectors_reader,
            use_faiss=search_config.get("use_faiss", True),
            search_type=search_config.get("search_type", "single"),
            beam_width=search_config.get("beam_width", 3)
        )
        
        # Charger le modèle d'embeddings
        model_name = prepare_config.get("model", "intfloat/multilingual-e5-large")
        print(f"⏳ Chargement du modèle {model_name}...")
        model = SentenceTransformer(model_name)
        print(f"✓ Modèle chargé")
        
        # Exemple de recherche
        k = 3  # Nombre de résultats à afficher
        test_query = "Qui a inventé la théorie de la relativité?"
        print(f"\nRecherche pour: {test_query}")
        
        # Encoder la requête
        encode_start = time.time()
        query_vector = model.encode(f"query: {test_query}", normalize_embeddings=True)
        encode_time = time.time() - encode_start
        
        # Recherche
        tree_search_start = time.time()
        tree_candidates = searcher.search_tree(query_vector)
        tree_search_time = time.time() - tree_search_start
        
        # Filtrer pour les k meilleurs
        filter_start = time.time()
        indices = searcher.filter_candidates(tree_candidates, query_vector, k)
        filter_time = time.time() - filter_start
        
        # Afficher les résultats
        print("\n📋 Résultats:")
        for i, idx in enumerate(indices, 1):
            if idx < len(qa_lines):
                parts = qa_lines[idx].strip().split(" ||| ")
                if len(parts) == 2:
                    question, answer = parts
                    print(f"\n{i}. {question}")
                    print(f"   → {answer}")
        
        # Afficher les temps
        print("\n⏱️ Temps d'exécution:")
        print(f"  → Encodage: {encode_time*1000:.2f} ms")
        print(f"  → Recherche arbre: {tree_search_time*1000:.2f} ms")
        print(f"  → Filtrage: {filter_time*1000:.2f} ms")
        total_time = encode_time + tree_search_time + filter_time
        print(f"  → Total: {total_time*1000:.2f} ms")
        
        return 0
        
    except Exception as e:
        print(f"❌ Erreur: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())