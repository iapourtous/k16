"""
Module pour exposer les fonctionnalités de K16 via une API REST.
Utilise FastAPI pour exposer un endpoint de recherche.
"""

import os
import time
import argparse
import numpy as np
from typing import Dict, Any, List, Optional

from k16.utils.config import ConfigManager
from k16.io.reader import read_vectors, load_tree
from k16.search.searcher import Searcher

# Variables globales pour stocker les ressources
model = None
searcher = None
qa_lines = None
vectors_reader = None
tree = None

def load_resources(config_manager):
    """Charge les ressources nécessaires (modèle, vecteurs, arbre, qa) à partir de la configuration."""
    global model, searcher, qa_lines, vectors_reader, tree
    
    # Récupération des paramètres
    search_config = config_manager.get_section("search")
    files_config = config_manager.get_section("files")
    prepare_config = config_manager.get_section("prepare_data")
    
    # Chemins des fichiers
    vectors_path = os.path.join(files_config["vectors_dir"], files_config["default_vectors"])
    tree_path = os.path.join(files_config["trees_dir"], files_config["default_tree"])
    qa_path = os.path.join(files_config["vectors_dir"], files_config.get("default_qa", "qa.txt"))
    
    # Vérifier que les fichiers existent
    if not os.path.exists(vectors_path):
        print(f"❌ Fichier de vecteurs introuvable: {vectors_path}")
        return False
        
    if not os.path.exists(tree_path) and not os.path.exists(tree_path.replace(".bsp", ".flat.npy")):
        print(f"❌ Fichier d'arbre introuvable: {tree_path}")
        return False
        
    if not os.path.exists(qa_path):
        print(f"❌ Fichier QA introuvable: {qa_path}")
        return False
    
    # Charger les vecteurs
    print(f"⏳ Chargement des vecteurs depuis {vectors_path}...")
    vectors_reader = read_vectors(
        file_path=vectors_path,
        mode=search_config["mode"],
        cache_size_mb=search_config["cache_size_mb"]
    )
    print(f"✓ Vecteurs chargés: {len(vectors_reader):,} vecteurs de dimension {vectors_reader.d}")
    
    # Charger l'arbre
    print(f"⏳ Chargement de l'arbre depuis {tree_path}...")
    tree = load_tree(tree_path, mmap_tree=(search_config["mode"] == "mmap"))
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
        use_faiss=search_config["use_faiss"],
        search_type=search_config["search_type"],
        beam_width=search_config.get("beam_width", 3)
    )
    
    # Charger le modèle d'embeddings
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("⚠️ Bibliothèque 'sentence-transformers' non installée. Installation en cours...")
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "sentence-transformers"])
        from sentence_transformers import SentenceTransformer
    
    model_name = prepare_config.get("model", "intfloat/multilingual-e5-large")
    print(f"⏳ Chargement du modèle {model_name}...")
    model = SentenceTransformer(model_name)
    print(f"✓ Modèle chargé")
    
    # Statistiques de l'arbre
    stats = tree.flat_tree.get_statistics()
    print("\n📊 Statistiques de l'arbre:")
    print(f"  → Nœuds: {stats.get('n_nodes', '?'):,}")
    print(f"  → Feuilles: {stats.get('n_leaves', '?'):,}")
    print(f"  → Profondeur: {stats.get('max_depth', '?')}")
    
    return True

def setup_app():
    """Configure et retourne l'application FastAPI."""
    # Importer les dépendances nécessaires
    try:
        from fastapi import FastAPI, HTTPException, Query
        from fastapi.middleware.cors import CORSMiddleware
        from pydantic import BaseModel
    except ImportError:
        print("⚠️ Les bibliothèques 'fastapi' et 'uvicorn' sont nécessaires pour l'API. Installation en cours...")
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "fastapi", "uvicorn[standard]"])
        from fastapi import FastAPI, HTTPException, Query
        from fastapi.middleware.cors import CORSMiddleware
        from pydantic import BaseModel

    # Modèles Pydantic pour la validation des données
    class SearchQuery(BaseModel):
        query: str
        k: Optional[int] = 10

    class SearchResult(BaseModel):
        question: str
        answer: str
        similarity: float
        index: int

    class SearchResponse(BaseModel):
        results: List[SearchResult]
        timings: Dict[str, float]
        stats: Dict[str, Any]
    
    # Application FastAPI
    app = FastAPI(
        title="K16 Search API",
        description="API for fast vector search using K16 hierarchical tree",
        version="1.0.0",
    )

    # Configuration CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Ajuster en production!
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/")
    async def root():
        """Endpoint racine avec informations de base."""
        return {
            "name": "K16 Search API",
            "version": "1.0.0",
            "description": "API for fast vector search using K16 hierarchical tree",
            "endpoints": {
                "/search": "Search for similar questions",
                "/stats": "Get statistics about the loaded tree",
                "/health": "Check API health status"
            }
        }

    @app.get("/health")
    async def health():
        """Vérification de l'état de santé de l'API."""
        if not model or not searcher or not qa_lines:
            return {
                "status": "error",
                "message": "Resources not fully loaded"
            }
        return {
            "status": "ok",
            "loaded_resources": {
                "model": model.__class__.__name__,
                "vectors": len(vectors_reader) if vectors_reader else 0,
                "qa_lines": len(qa_lines) if qa_lines else 0,
                "tree_loaded": searcher is not None
            }
        }

    @app.get("/stats")
    async def get_stats():
        """Récupère les statistiques de l'arbre et des vecteurs."""
        if not searcher or not tree:
            raise HTTPException(status_code=503, detail="Tree not loaded")
        
        stats = tree.flat_tree.get_statistics()
        
        return {
            "tree": {
                "nodes": stats.get("n_nodes", 0),
                "leaves": stats.get("n_leaves", 0),
                "max_depth": stats.get("max_depth", 0),
                "compression": stats.get("compression", {})
            },
            "vectors": {
                "count": len(vectors_reader) if vectors_reader else 0,
                "dimensions": vectors_reader.d if vectors_reader else 0
            },
            "qa": {
                "count": len(qa_lines) if qa_lines else 0
            },
            "search": {
                "type": searcher.search_type,
                "beam_width": searcher.beam_width if searcher.search_type == "beam" else None
            }
        }

    @app.post("/search", response_model=SearchResponse)
    async def search(query_data: SearchQuery):
        """
        Recherche des questions similaires à la requête.
        
        Args:
            query_data: Contient la requête et le nombre de résultats souhaités
            
        Returns:
            Les résultats de la recherche et les informations de timing
        """
        if not model or not searcher or not qa_lines:
            raise HTTPException(status_code=503, detail="Resources not fully loaded")
        
        # Encoder la requête
        encode_start = time.time()
        query_vector = model.encode(f"query: {query_data.query}", normalize_embeddings=True)
        encode_time = time.time() - encode_start
        
        # Recherche avec l'arbre
        tree_search_start = time.time()
        tree_candidates = searcher.search_tree(query_vector)
        tree_search_time = time.time() - tree_search_start
        
        # Filtrer pour obtenir les k meilleurs
        filter_start = time.time()
        indices_with_scores = []
        
        # Utiliser VectorReader optimisé
        top_indices = searcher.filter_candidates(tree_candidates, query_vector, query_data.k)
        top_vectors = vectors_reader[top_indices]
        scores = [np.dot(query_vector, top_vectors[i]) for i in range(len(top_indices))]
        
        # Trier par score décroissant
        sorted_pairs = sorted(zip(top_indices, scores), key=lambda x: x[1], reverse=True)
        indices_with_scores = [(idx, score) for idx, score in sorted_pairs]
        
        filter_time = time.time() - filter_start
        
        # Récupérer les résultats
        results = []
        for idx, score in indices_with_scores:
            if idx < len(qa_lines):
                parts = qa_lines[idx].strip().split(" ||| ")
                if len(parts) == 2:
                    question, answer = parts
                    results.append(SearchResult(
                        question=question,
                        answer=answer,
                        similarity=float(score),
                        index=idx
                    ))
        
        # Calculer les statistiques et les temps
        timings = {
            "encode_ms": encode_time * 1000,
            "tree_search_ms": tree_search_time * 1000,
            "filter_ms": filter_time * 1000,
            "total_ms": (encode_time + tree_search_time + filter_time) * 1000
        }
        
        stats = {
            "candidates_count": len(tree_candidates)
        }
        
        return SearchResponse(
            results=results,
            timings=timings,
            stats=stats
        )
    
    return app

def api_command(args: argparse.Namespace) -> int:
    """
    Commande pour lancer l'API FastAPI.
    
    Args:
        args: Arguments de ligne de commande
        
    Returns:
        int: Code de retour (0 pour succès, autre pour erreur)
    """
    # Initialisation du gestionnaire de configuration
    config_manager = ConfigManager(args.config)
    
    try:
        # Vérifier que FastAPI est installé
        try:
            import uvicorn
        except ImportError:
            print("⚠️ Les bibliothèques 'fastapi' et 'uvicorn' sont nécessaires pour l'API. Installation en cours...")
            import subprocess, sys
            subprocess.check_call([sys.executable, "-m", "pip", "install", "fastapi", "uvicorn[standard]"])
            import uvicorn
            
        # Récupérer les paramètres API depuis le fichier config
        api_config = config_manager.get_section("api")

        # Définir les valeurs par défaut si la section API n'existe pas ou est incomplète
        default_host = "127.0.0.1"
        default_port = 8000
        default_reload = False
        
        host = args.host if args.host else api_config.get("host", default_host)
        port = args.port if args.port else api_config.get("port", default_port)
        reload = args.reload if args.reload is not None else api_config.get("reload", default_reload)
        
        print(f"🌐 Démarrage de l'API K16...")
        print(f"  - Configuration: {args.config}")
        print(f"  - Adresse: {host}:{port}")
        print(f"  - Rechargement auto: {'Activé' if reload else 'Désactivé'}")
        
        # Charger les ressources
        if not load_resources(config_manager):
            return 1
        
        # Configurer l'application et remplacer l'app globale
        global app
        app = setup_app()

        # Démarrer l'API
        print(f"\n🚀 Démarrage du serveur API sur http://{host}:{port}")
        print(f"  → Documentation Swagger: http://{host}:{port}/docs")
        print(f"  → Documentation ReDoc: http://{host}:{port}/redoc")
        print(f"  → Pour arrêter le serveur: CTRL+C\n")

        # Démarrer Uvicorn
        if reload:
            # En mode reload, on doit utiliser un chemin d'importation
            import inspect
            module_path = inspect.getmodule(api_command).__name__
            uvicorn.run(
                f"{module_path}:app",
                host=host,
                port=port,
                reload=reload,
                log_level="info"
            )
        else:
            # En mode normal, on peut passer l'app directement
            uvicorn.run(
                app,
                host=host,
                port=port,
                log_level="info"
            )
        
    except Exception as e:
        print(f"\n❌ Erreur: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

# Variable globale pour l'application qui sera configurée lors de l'exécution
# Ne pas initialiser ici pour éviter les erreurs d'importation si FastAPI n'est pas installé
app = None