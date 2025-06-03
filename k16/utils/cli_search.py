"""
Module pour la recherche interactive en ligne de commande.
Fournit une interface pour rechercher des questions similaires dans un terminal.
"""

import os
import time
import numpy as np
from typing import List, Dict, Any, Tuple
import argparse

from k16.utils.config import ConfigManager
from k16.io.reader import read_vectors, load_tree
from k16.search.searcher import search, Searcher

def format_results(results: List[Dict], timings: Dict[str, float]) -> str:
    """
    Formate les résultats de recherche pour l'affichage en terminal.
    
    Args:
        results: Liste des résultats de recherche
        timings: Dictionnaire des temps d'exécution
        
    Returns:
        str: Résultats formatés
    """
    output = []
    
    # Afficher les métriques
    output.append("\n🕒 Temps:")
    output.append(f"  → Encodage      : {timings['encode']*1000:.2f} ms")
    output.append(f"  → Recherche arbre: {timings['tree_search']*1000:.2f} ms")
    output.append(f"  → Filtrage      : {timings['filter']*1000:.2f} ms")
    search_only = timings['tree_search'] + timings['filter']
    total_time = timings['encode'] + search_only
    output.append(f"  → Recherche totale: {search_only*1000:.2f} ms")
    output.append(f"  → Temps total    : {total_time*1000:.2f} ms")
    
    # Afficher les résultats
    output.append("\n📋 Résultats:")
    for i, result in enumerate(results, 1):
        output.append(f"\n{i}. {result['question']}")
        output.append(f"   → {result['answer']}")
    
    return "\n".join(output)

def search_once(model, searcher, qa_lines: List[str], query: str, k: int = 10) -> Tuple[List[Dict], Dict[str, float]]:
    """
    Effectue une seule recherche.

    Args:
        model: Modèle d'embeddings
        searcher: Chercheur K16
        qa_lines: Lignes de questions-réponses
        query: Question à rechercher
        k: Nombre de résultats à retourner

    Returns:
        Tuple[List[Dict], Dict[str, float]]: Résultats et timings
    """
    # Encoder la requête
    encode_start = time.time()
    query_vector = model.encode(f"query: {query}", normalize_embeddings=True)
    encode_time = time.time() - encode_start

    # Recherche avec l'arbre
    tree_search_start = time.time()
    tree_candidates = searcher.search_tree(query_vector)
    tree_search_time = time.time() - tree_search_start

    # Filtrer pour obtenir les k meilleurs
    filter_start = time.time()
    indices = searcher.filter_candidates(tree_candidates, query_vector, k)
    filter_time = time.time() - filter_start

    # Récupérer les résultats
    results = []
    for idx in indices:
        if idx < len(qa_lines):
            parts = qa_lines[idx].strip().split(" ||| ")
            if len(parts) == 2:
                question, answer = parts
                results.append({
                    "question": question,
                    "answer": answer,
                    "index": idx
                })

    timings = {
        "encode": encode_time,
        "tree_search": tree_search_time,
        "filter": filter_time
    }

    return results, timings

def search_interactive(model, searcher, qa_lines: List[str], k: int = 10) -> Tuple[List[Dict], Dict[str, float]]:
    """
    Effectue une recherche interactive.

    Args:
        model: Modèle d'embeddings
        searcher: Chercheur K16
        qa_lines: Lignes de questions-réponses
        k: Nombre de résultats à retourner

    Returns:
        Tuple[List[Dict], Dict[str, float]]: Résultats et timings
    """
    try:
        query = input("\nVotre question (q pour quitter): ")

        if query.lower() in ['q', 'quit', 'exit']:
            return None, None

        return search_once(model, searcher, qa_lines, query, k)

    except EOFError:
        print("\nMode non-interactif détecté. Voici un exemple de recherche:")
        example_query = "Qui a inventé la théorie de la relativité?"
        print(f"Question: {example_query}")

        results, timings = search_once(model, searcher, qa_lines, example_query, k)
        return results, timings

def search_command(args: argparse.Namespace) -> int:
    """
    Commande pour effectuer une recherche interactive dans un terminal.

    Args:
        args: Arguments de ligne de commande

    Returns:
        int: Code de retour (0 pour succès, autre pour erreur)
    """
    # Initialisation du gestionnaire de configuration
    config_manager = ConfigManager(args.config)

    # Récupération des paramètres
    search_config = config_manager.get_section("search")
    files_config = config_manager.get_section("files")
    build_config = config_manager.get_section("build_tree")
    prepare_config = config_manager.get_section("prepare_data")

    try:
        print(f"🔍 Mode recherche K16 interactive...")
        print(f"  - Vecteurs: {args.vectors_file}")
        print(f"  - Arbre: {args.tree_file}")
        print(f"  - QA: {args.qa_file}")
        print(f"  - Mode: {args.mode}")
        print(f"  - K (nombre de résultats): {args.k}")
        print(f"  - Type de recherche: {args.search_type}")
        print(f"  - Largeur de faisceau: {args.beam_width}")

        # Vérifier que les fichiers existent
        if not os.path.exists(args.vectors_file):
            print(f"❌ Fichier de vecteurs introuvable: {args.vectors_file}")
            print(f"   Utilisez la commande 'getData' pour télécharger et préparer les données:")
            print(f"   python -m k16.cli getData --config {args.config}")
            return 1

        if not os.path.exists(args.tree_file) and not os.path.exists(args.tree_file.replace(".bsp", ".flat.npy")):
            print(f"❌ Fichier d'arbre introuvable: {args.tree_file}")
            print(f"   Utilisez la commande 'build' pour construire l'arbre:")
            print(f"   python -m k16.cli build {args.vectors_file} --config {args.config}")
            return 1

        if not os.path.exists(args.qa_file):
            print(f"❌ Fichier QA introuvable: {args.qa_file}")
            print(f"   Utilisez la commande 'getData' pour télécharger et préparer les données:")
            print(f"   python -m k16.cli getData --config {args.config}")
            return 1

        # Charger les vecteurs
        print(f"⏳ Chargement des vecteurs depuis {args.vectors_file}...")
        vectors_reader = read_vectors(
            file_path=args.vectors_file,
            mode=args.mode,
            cache_size_mb=args.cache_size
        )
        print(f"✓ Vecteurs chargés: {len(vectors_reader):,} vecteurs de dimension {vectors_reader.d}")

        # Charger l'arbre
        print(f"⏳ Chargement de l'arbre depuis {args.tree_file}...")
        tree = load_tree(args.tree_file, mmap_tree=(args.mode == "mmap"))
        print(f"✓ Arbre chargé")

        # Charger les questions et réponses
        print(f"⏳ Chargement des questions et réponses depuis {args.qa_file}...")
        with open(args.qa_file, "r", encoding="utf-8") as f:
            qa_lines = f.readlines()
        print(f"✓ {len(qa_lines):,} questions-réponses chargées")

        # Créer le chercheur
        from k16.search.searcher import Searcher
        searcher = Searcher(
            k16tree=tree,
            vectors_reader=vectors_reader,
            use_faiss=args.use_faiss,
            search_type=args.search_type,
            beam_width=args.beam_width
        )

        # Charger le modèle d'embeddings
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            print("⚠️ Bibliothèque 'sentence-transformers' non installée. Installation en cours...")
            import subprocess, sys
            subprocess.check_call([sys.executable, "-m", "pip", "install", "sentence-transformers"])
            from sentence_transformers import SentenceTransformer

        print(f"⏳ Chargement du modèle {args.model}...")
        model = SentenceTransformer(args.model)
        print(f"✓ Modèle chargé")

        # Statistiques de l'arbre
        stats = tree.flat_tree.get_statistics()
        compression_stats = stats.get('compression', {})

        print("\n📊 Statistiques de l'arbre:")
        print(f"  → Nœuds: {stats.get('n_nodes', '?'):,}")
        print(f"  → Feuilles: {stats.get('n_leaves', '?'):,}")
        print(f"  → Profondeur: {stats.get('max_depth', '?')}")

        # Lancement de l'interface interactive
        print(f"\n💬 Interface de recherche interactive K16")
        print(f"  → Tapez votre question et appuyez sur Entrée")
        print(f"  → Tapez 'q' pour quitter")

        # En mode interactif, continuer jusqu'à ce que l'utilisateur quitte
        try:
            while True:
                results, timings = search_interactive(model, searcher, qa_lines, k=args.k)

                if results is None:
                    print("\n👋 Au revoir!")
                    break

                # Afficher les résultats
                print(format_results(results, timings))
        except KeyboardInterrupt:
            print("\n👋 Recherche interrompue. Au revoir!")

    except Exception as e:
        print(f"\n❌ Erreur: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

    return 0